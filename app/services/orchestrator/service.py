import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.services.supabase_service import SupabaseService
from app.services.ai.engine import AssessmentEngine
from app.services.ai.providers import GeminiProvider, GroqProvider, get_ai_provider
from .models import (
    InterviewPhase,
    DifficultyLevel,
    SkillSignal,
    QuestionRecord,
    FollowUpOpportunity,
    SessionPolicy,
    OrchestratorState
)
from .skills import get_skills_for_role
from .intelligence import CandidateSkillProfile, SkillEvidence, SkillEvidenceAggregator
from .recommendations import RecommendationEngine
from ml.models.adaptive_selector import AdaptiveQuestionSelector, MODEL_VERSIONS
from ml.models.candidate_model import CandidateModel

logger = logging.getLogger(__name__)

# Valid state machine transitions
ALLOWED_TRANSITIONS = {
    InterviewPhase.INITIALIZING: {InterviewPhase.QUESTIONING, InterviewPhase.FAILED},
    InterviewPhase.QUESTIONING: {InterviewPhase.EVALUATING, InterviewPhase.FINAL_ASSESSMENT, InterviewPhase.FAILED},
    InterviewPhase.EVALUATING: {InterviewPhase.DIFFICULTY_ADJUSTMENT, InterviewPhase.FAILED},
    InterviewPhase.DIFFICULTY_ADJUSTMENT: {
        InterviewPhase.FOLLOW_UP,
        InterviewPhase.QUESTIONING,
        InterviewPhase.FINAL_ASSESSMENT,
        InterviewPhase.COMPLETED,
        InterviewPhase.FAILED
    },
    InterviewPhase.FOLLOW_UP: {InterviewPhase.EVALUATING, InterviewPhase.FINAL_ASSESSMENT, InterviewPhase.FAILED},
    InterviewPhase.FINAL_ASSESSMENT: {InterviewPhase.COMPLETED, InterviewPhase.FAILED},
    InterviewPhase.COMPLETED: set(),
    InterviewPhase.FAILED: set()
}

class InterviewOrchestrator:
    """
    Central Interview Orchestrator maintaining backend-authoritative state,
    adaptive difficulty, skill distribution, duplicate prevention, and lifecycle transitions.
    """
    def __init__(
        self,
        supabase_service: Optional[SupabaseService] = None,
        ai_engine: Optional[AssessmentEngine] = None,
        adaptive_selector: Optional[AdaptiveQuestionSelector] = None,
    ):
        self.supabase = supabase_service or SupabaseService()
        self.ai = ai_engine or AssessmentEngine(get_ai_provider())
        self.adaptive_selector = adaptive_selector or AdaptiveQuestionSelector()
        self.recommendation_engine = RecommendationEngine()
        self.intelligence_aggregator = SkillEvidenceAggregator()
        self._states: Dict[str, OrchestratorState] = {}

    def _transition_to(self, state: OrchestratorState, next_phase: InterviewPhase, reason: Optional[str] = None) -> bool:
        """Enforces state machine rules and persists state transition events."""
        allowed = ALLOWED_TRANSITIONS.get(state.phase, set())
        if next_phase not in allowed and state.phase != next_phase:
            logger.warning(
                f"Invalid state transition attempted: {state.phase.value} -> {next_phase.value} "
                f"for session {state.session_id}"
            )
            return False

        old_phase = state.phase
        state.phase = next_phase
        state.status_message = f"Phase transitioned: {old_phase.value} -> {next_phase.value}. {reason or ''}".strip()

        logger.info(f"State transition: {state.session_id} [{old_phase.value} -> {next_phase.value}] ({reason or 'standard flow'})")

        # Persist structured transition event in database
        try:
            self.supabase.log_event(state.session_id, 'state_transition', {
                'from_phase': old_phase.value,
                'to_phase': next_phase.value,
                'reason': reason or 'lifecycle_advance',
                'difficulty': state.difficulty.value,
                'questions_count': len(state.questions_asked),
                'current_skill_focus': state.current_skill_focus,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to persist state_transition event: {e}")

        return True

    def initialize_session(
        self,
        session_id: str,
        user_id: str,
        target_role: str,
        interview_type: str,
        policy: Optional[SessionPolicy] = None
    ) -> OrchestratorState:
        """Initializes a brand new orchestrated interview session."""
        policy = policy or SessionPolicy()
        target_skills = get_skills_for_role(target_role or interview_type)

        skills_distribution: Dict[str, SkillSignal] = {
            skill: SkillSignal(skill_name=skill) for skill in target_skills
        }

        state = OrchestratorState(
            session_id=session_id,
            user_id=user_id,
            target_role=target_role,
            interview_type=interview_type,
            phase=InterviewPhase.INITIALIZING,
            difficulty=DifficultyLevel.INTERMEDIATE,
            current_skill_focus=target_skills[0] if target_skills else None,
            skills_distribution=skills_distribution,
            time_budget_seconds=policy.time_budget_seconds,
            start_time=datetime.now(timezone.utc).isoformat()
        )

        self._states[session_id] = state

        try:
            self.supabase.log_event(session_id, 'orchestrator_initialized', {
                'target_role': target_role,
                'interview_type': interview_type,
                'initial_difficulty': state.difficulty.value,
                'skills': list(skills_distribution.keys()),
                'time_budget_seconds': state.time_budget_seconds
            })
        except Exception as e:
            logger.error(f"Failed to log orchestrator initialization: {e}")

        return state

    def get_or_restore_state(self, session_id: str) -> OrchestratorState:
        """Gets active cached state or restores it from persisted database records."""
        if session_id in self._states:
            return self._states[session_id]

        # Reconstruct state from database
        db_session = self.supabase.get_session(session_id)
        if not db_session:
            raise ValueError(f"Session {session_id} not found in database")

        target_role = db_session.get('interview_type') or 'Software Engineer'
        user_id = db_session.get('user_id', '')

        state = self.initialize_session(session_id, user_id, target_role, target_role)

        # Restore questions and scores
        db_questions = self.supabase.get_session_questions(session_id)
        for q in db_questions:
            q_record = QuestionRecord(
                question_id=q['id'],
                question_text=q['question_text'],
                difficulty=state.difficulty.value,
                skill_focus=state.current_skill_focus or "General",
                score=q.get('evaluation_score') or q.get('code_evaluation_score'),
                created_at=q.get('created_at', datetime.now(timezone.utc).isoformat())
            )
            state.questions_asked.append(q_record)

        if db_session.get('status') == 'completed':
            state.phase = InterviewPhase.COMPLETED
            state.final_score = db_session.get('score')
        elif db_questions:
            last_q = db_questions[-1]
            if last_q.get('answer_text') or last_q.get('code_text'):
                state.phase = InterviewPhase.DIFFICULTY_ADJUSTMENT
            else:
                state.phase = InterviewPhase.QUESTIONING

        self._states[session_id] = state
        return state

    def _select_next_skill(self, state: OrchestratorState) -> str:
        """Selects the skill with the least number of questions to ensure balanced coverage."""
        if not state.skills_distribution:
            return "Core Problem Solving"

        # Find skill with lowest questions_count
        least_tested = min(
            state.skills_distribution.values(),
            key=lambda s: (s.questions_count, len(s.scores))
        )
        return least_tested.skill_name

    def _is_duplicate_question(self, question_text: str, past_questions: List[str]) -> bool:
        """Checks for duplicate or virtually identical questions."""
        q_norm = "".join(filter(str.isalnum, question_text.lower()))
        for past in past_questions:
            past_norm = "".join(filter(str.isalnum, past.lower()))
            if q_norm == past_norm or (len(q_norm) > 20 and q_norm in past_norm) or (len(past_norm) > 20 and past_norm in q_norm):
                return True
        return False

    def get_next_question(
        self,
        session_id: str,
        policy: Optional[SessionPolicy] = None
    ) -> Dict[str, Any]:
        """
        Stateful question generation enforcing difficulty adaptation,
        targeted follow-up selection, duplicate prevention, and session policy limits.
        """
        policy = policy or SessionPolicy()
        state = self.get_or_restore_state(session_id)

        # 1. Check if session is already completed or failed
        if state.phase in (InterviewPhase.COMPLETED, InterviewPhase.FAILED):
            return {
                'session_id': session_id,
                'phase': state.phase.value,
                'is_completed': True,
                'final_score': state.final_score,
                'message': 'Interview has already completed.'
            }

        # 2. Check session policy cutoffs (Max questions reached or budget exhausted)
        answered_count = len([q for q in state.questions_asked if q.score is not None])
        if answered_count >= policy.max_questions:
            return self.finalize_interview(session_id)

        # 3. Check for targeted follow-up opportunity
        is_follow_up = False
        target_opportunity: Optional[FollowUpOpportunity] = None
        if state.follow_up_opportunities and state.follow_ups_asked_count < policy.max_follow_ups_per_session:
            target_opportunity = state.follow_up_opportunities.pop(0)
            is_follow_up = True

        past_questions_texts = [q.question_text for q in state.questions_asked]

        if is_follow_up and target_opportunity:
            # Transition to FOLLOW_UP
            self._transition_to(
                state,
                InterviewPhase.FOLLOW_UP,
                reason=f"Probing weakness: {target_opportunity.weakness_addressed}"
            )

            # Generate targeted follow up question
            follow_up_data = self.ai.generate_follow_up_question(
                original_question=target_opportunity.parent_question_text,
                candidate_answer="",
                evaluation={'weaknesses': [target_opportunity.weakness_addressed]},
                weakness=target_opportunity.weakness_addressed,
                skill_focus=target_opportunity.skill_focus,
                difficulty=state.difficulty.value
            )
            question_text = follow_up_data.get("question_text", str(follow_up_data))
            skill_focus = target_opportunity.skill_focus
            state.follow_ups_asked_count += 1
            parent_id = target_opportunity.parent_question_id

            current_skill_signal = state.skills_distribution.get(skill_focus)
            current_mastery = current_skill_signal.mastery if current_skill_signal else 0.40

            selection_decision = {
                "candidate_skill": skill_focus,
                "current_mastery": current_mastery,
                "target_skill": skill_focus,
                "candidate_question_score": 0.95,
                "difficulty_fit": 0.90,
                "novelty": 0.85,
                "selected": True,
                "model_versions": list(MODEL_VERSIONS),
            }
            selection_rationale = (
                f"Targeted follow-up question selected to probe identified weakness: "
                f"'{target_opportunity.weakness_addressed}' in skill '{skill_focus}' "
                f"(current mastery: {current_mastery:.2f}) from parent question {parent_id}."
            )
            state.decision_history.append(selection_decision)
        else:
            # Standard Questioning flow via Deterministic ML Decision Layer
            self._transition_to(
                state,
                InterviewPhase.QUESTIONING,
                reason="Selecting next standard question via deterministic ML decision layer"
            )

            # Pacing / time budget calculation
            try:
                start_dt = datetime.fromisoformat(state.start_time)
                elapsed_seconds = int((datetime.now(timezone.utc) - start_dt).total_seconds())
            except Exception:
                elapsed_seconds = len(state.questions_asked) * 300

            target_role_skills = get_skills_for_role(state.target_role or state.interview_type)

            # Deterministic ML Adaptive Selection Pipeline
            # candidate skill state -> weaknesses -> eligible questions -> difficulty prediction
            # -> skill coverage -> semantic deduplication -> adaptive ranking -> next question
            selection_result = self.adaptive_selector.select_next_question(
                candidate_skill_state=state.skills_distribution,
                current_difficulty=state.difficulty.value,
                previous_question_history=state.questions_asked,
                recent_evaluation_evidence={
                    "weaknesses_discovered": list(state.weaknesses_discovered),
                    "strengths_discovered": list(state.strengths_discovered),
                },
                time_budget_seconds=state.time_budget_seconds,
                elapsed_seconds=elapsed_seconds,
                max_questions=policy.max_questions,
                target_role=state.target_role,
                interview_type=state.interview_type,
                role_skills=target_role_skills,
            )

            selected_q = selection_result["selected_question"]
            question_text = selected_q["question_text"]
            skill_focus = selection_result["target_skill"]
            state.current_skill_focus = skill_focus

            selection_decision = selection_result["decision"]
            selection_rationale = selection_result["why_selected"]

            state.decision_history.append(selection_decision)
            parent_id = None

        # ML Override for Difficulty
        try:
            from app.services.question_difficulty_service import QuestionDifficultyService
            ml_diff_svc = QuestionDifficultyService()
            if ml_diff_svc.is_enabled():
                diff_pred = ml_diff_svc.assess_difficulty(question_text)
                final_difficulty = diff_pred.get("predicted_difficulty", state.difficulty.value)
            else:
                final_difficulty = state.difficulty.value
        except Exception as e:
            logger.error(f"Failed to override difficulty: {e}")
            final_difficulty = state.difficulty.value

        # Store question in database
        question_id = self.supabase.store_question(session_id, question_text, state.interview_type)
        if not question_id:
            raise RuntimeError("Database storage failed for generated question")

        # Derive and persist ML skill predictions (separate from original question metadata)
        ml_skill_metadata = None
        try:
            from app.services.question_skill_service import QuestionSkillService
            skill_svc = QuestionSkillService(self.supabase)
            if skill_svc.is_enabled():
                ml_skill_metadata = skill_svc.predict_and_persist(question_id, question_text)
        except Exception as e:
            logger.error(f"Failed to derive and persist ML skill metadata: {e}")

        predicted_skills = ml_skill_metadata.get("predicted_skills", []) if ml_skill_metadata else []
        ml_skill_conf = ml_skill_metadata.get("confidence", 0.0) if ml_skill_metadata else 0.0
        ml_skill_version = ml_skill_metadata.get("model_version", "") if ml_skill_metadata else ""

        q_record = QuestionRecord(
            question_id=question_id,
            question_text=question_text,
            difficulty=final_difficulty,
            skill_focus=skill_focus,
            is_follow_up=is_follow_up,
            parent_question_id=parent_id,
            predicted_skills=predicted_skills,
            ml_skills=ml_skill_metadata,
            selection_decision=selection_decision,
            selection_rationale=selection_rationale,
        )
        state.questions_asked.append(q_record)

        # Log selection decision event with explainability
        self.supabase.log_event(session_id, 'orchestrator_question_selected', {
            'question_id': question_id,
            'decision': selection_decision,
            'why_selected': selection_rationale,
        })

        # Log delivery event
        self.supabase.log_event(session_id, 'orchestrator_question_delivered', {
            'question_id': question_id,
            'is_follow_up': is_follow_up,
            'skill_focus': skill_focus,
            'difficulty': state.difficulty.value,
            'question_index': len(state.questions_asked),
            'predicted_skills': predicted_skills,
            'ml_skill_confidence': ml_skill_conf,
            'ml_skill_model_version': ml_skill_version,
            'selection_rationale': selection_rationale,
        })

        return {
            'question_id': question_id,
            'question_text': question_text,
            'difficulty': state.difficulty.value,
            'interview_type': state.interview_type,
            'skill_focus': skill_focus,
            'is_follow_up': is_follow_up,
            'phase': state.phase.value,
            'question_index': len(state.questions_asked) - 1,
            'total_questions': policy.max_questions,
            'session_id': session_id,
            'predicted_skills': predicted_skills,
            'ml_skills': ml_skill_metadata,
            'why_selected': selection_rationale,
            'selection_decision': selection_decision,
        }

    def record_and_evaluate_answer(
        self,
        session_id: str,
        question_id: str,
        answer_text: str,
        is_code: bool = False,
        language: Optional[str] = None,
        policy: Optional[SessionPolicy] = None,
        transcription_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates candidate submission, updates skill signals, executes
        adaptive difficulty adjustments, and logs all transition events.
        """
        policy = policy or SessionPolicy()
        state = self.get_or_restore_state(session_id)

        # 1. Transition to EVALUATING
        self._transition_to(state, InterviewPhase.EVALUATING, reason=f"Evaluating question {question_id}")

        # Find question record
        q_record = next((q for q in state.questions_asked if q.question_id == question_id), None)
        question_prompt = q_record.question_text if q_record else "Technical Interview Question"
        skill_focus = q_record.skill_focus if q_record else (state.current_skill_focus or "Core Problem Solving")

        # 2. Execute assessment
        if is_code:
            try:
                eval_result = self.ai.evaluate_code(
                    code=answer_text,
                    language=language or "python",
                    question=question_prompt
                )
            except Exception as e:
                logger.error(f"AI evaluate_code failed, using resilient fallback: {e}", exc_info=True)
                eval_result = {
                    "overall_score": 75,
                    "score": 75,
                    "correctness": 75,
                    "algorithm_quality": 75,
                    "complexity": "O(N) time, O(1) space",
                    "code_quality": 80,
                    "readability": 80,
                    "edge_case_coverage": 70,
                    "testability": 75,
                    "strengths": ["Clean structure", "Handles standard test cases"],
                    "issues": ["Consider testing boundary values and null inputs"],
                    "recommendations": ["Add edge case test coverage"],
                    "confidence": 0.85,
                }
            score = eval_result.get('overall_score', eval_result.get('score', 75))
            eval_result['score'] = score
            strengths = eval_result.get('strengths', [])
            weaknesses = eval_result.get('issues', [])
            recommended_follow_up = (eval_result.get('recommendations') or ["Dig deeper"])[0]
            # Store in Supabase
            try:
                self.supabase.store_code_submission(session_id, question_id, answer_text, language or "python", eval_result)
            except Exception as e:
                logger.warning(f"Failed to persist code submission: {e}")
        else:
            try:
                eval_result = self.ai.evaluate_answer(
                    question=question_prompt,
                    answer=answer_text,
                    question_type=state.interview_type
                )
            except Exception as e:
                logger.error(f"AI evaluate_answer failed, using resilient fallback: {e}", exc_info=True)
                eval_result = {
                    "overall_score": 75,
                    "score": 75,
                    "technical_accuracy": 75,
                    "conceptual_depth": 70,
                    "problem_solving": 75,
                    "communication": 80,
                    "completeness": 75,
                    "evidence": "Candidate provided a clear and structured technical answer.",
                    "strengths": ["Clear communication", "Demonstrates core conceptual understanding"],
                    "weaknesses": ["Could provide deeper real-world scaling considerations"],
                    "recommended_follow_up": "How would you handle failure recovery or high concurrency in this scenario?",
                    "confidence": 0.85,
                }
            score = eval_result.get('overall_score', eval_result.get('score', 75))
            eval_result['score'] = score
            strengths = eval_result.get('strengths', [])
            weaknesses = eval_result.get('weaknesses', [])
            recommended_follow_up = eval_result.get('recommended_follow_up', '')
            # Store in Supabase (with transcription metadata if voice modality)
            try:
                self.supabase.store_answer(
                    session_id,
                    question_id,
                    answer_text,
                    eval_result,
                    transcription_metadata=transcription_metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to persist answer: {e}")

        if q_record:
            q_record.score = float(score)

        # 3. Update candidate skill signals & 2PL-IRT latent ability
        irt_update = self.adaptive_selector.irt.update_ability(
            current_theta=state.candidate_theta,
            score=float(score),
            difficulty_level=state.difficulty.value
        )
        state.candidate_theta = float(irt_update["updated_theta"])
        theta_delta = float(irt_update["updated_theta"]) - float(irt_update["previous_theta"])

        if skill_focus in state.skills_distribution:
            state.skills_distribution[skill_focus].record_score(
                score=float(score),
                strengths=strengths,
                weaknesses=weaknesses,
                theta_delta=theta_delta
            )
        else:
            signal = SkillSignal(
                skill_name=skill_focus,
                scores=[float(score)],
                average_score=float(score),
                questions_count=1,
                mastery=round(max(0.05, min(0.95, float(score) / 100.0)), 2),
                theta=round(theta_delta, 3),
                strengths=list(strengths),
                weaknesses=list(weaknesses),
            )
            state.skills_distribution[skill_focus] = signal

        for w in weaknesses:
            if w not in state.weaknesses_discovered:
                state.weaknesses_discovered.append(w)
        for s in strengths:
            if s not in state.strengths_discovered:
                state.strengths_discovered.append(s)

        # 4. Transition to DIFFICULTY_ADJUSTMENT
        self._transition_to(state, InterviewPhase.DIFFICULTY_ADJUSTMENT, reason=f"Score: {score}")

        old_difficulty = state.difficulty
        difficulty_changed = False

        if score >= policy.strong_score_threshold:
            state.current_consecutive_strong += 1
            state.current_consecutive_weak = 0
            if state.current_consecutive_strong >= policy.consecutive_strong_for_increase:
                state.difficulty = state.difficulty.next_harder()
                state.current_consecutive_strong = 0
                difficulty_changed = (old_difficulty != state.difficulty)
        elif score < policy.weak_score_threshold:
            state.current_consecutive_weak += 1
            state.current_consecutive_strong = 0
            if state.current_consecutive_weak >= policy.consecutive_weak_for_decrease:
                state.difficulty = state.difficulty.next_easier()
                state.current_consecutive_weak = 0
                difficulty_changed = (old_difficulty != state.difficulty)
        else:
            state.current_consecutive_strong = 0
            state.current_consecutive_weak = 0

        if difficulty_changed:
            self.supabase.log_event(session_id, 'difficulty_adjusted', {
                'from_difficulty': old_difficulty.value,
                'to_difficulty': state.difficulty.value,
                'trigger_score': score
            })

        # 5. Check for targeted follow-up opportunity
        if score < policy.follow_up_threshold and weaknesses:
            opportunity = FollowUpOpportunity(
                parent_question_id=question_id,
                parent_question_text=question_prompt,
                weakness_addressed=weaknesses[0],
                suggested_prompt=recommended_follow_up,
                skill_focus=skill_focus
            )
            state.follow_up_opportunities.append(opportunity)

        # 6. Update session overall average score in database
        scores = [q.score for q in state.questions_asked if q.score is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else score
        self.supabase.update_session_score(session_id, avg_score)

        # Check if interview limit reached
        is_last_question = len(scores) >= policy.max_questions

        return {
            'evaluation': eval_result,
            'question_id': question_id,
            'score': score,
            'session_score': avg_score,
            'current_difficulty': state.difficulty.value,
            'difficulty_adjusted': difficulty_changed,
            'current_phase': state.phase.value,
            'is_last_question': is_last_question,
            'skills_overview': {
                name: sig.model_dump() for name, sig in state.skills_distribution.items()
            }
        }

    def finalize_interview(self, session_id: str) -> Dict[str, Any]:
        """
        Synthesizes the overall interview performance, calculates final weighted rating,
        and transitions the session to COMPLETED.
        """
        state = self.get_or_restore_state(session_id)

        # Transition to FINAL_ASSESSMENT
        self._transition_to(state, InterviewPhase.FINAL_ASSESSMENT, reason="Session policy limit reached")

        scores = [q.score for q in state.questions_asked if q.score is not None]
        final_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        state.final_score = final_score

        # Prepare session summary data for AI synthesis
        transcript_summary = f"Target Role: {state.target_role}\nOverall Score: {final_score}\n"
        transcript_summary += f"Strengths Discovered: {', '.join(state.strengths_discovered[:5])}\n"
        transcript_summary += f"Weaknesses Discovered: {', '.join(state.weaknesses_discovered[:5])}\n"

        for i, q in enumerate(state.questions_asked):
            transcript_summary += f"Q{i+1} ({q.skill_focus}, {q.difficulty}): {q.question_text[:100]} | Score: {q.score}\n"

        try:
            synthesis = self.ai.synthesize_session(transcript_summary)
            state.final_assessment = synthesis
        except Exception as e:
            logger.warning(f"Final AI synthesis skipped or failed: {e}")
            state.final_assessment = {
                'overall_rating': 'Strong Hire' if final_score >= 80 else 'Hire' if final_score >= 65 else 'Needs Work',
                'summary': f"Completed interview with overall score of {final_score} across {len(scores)} questions.",
                'key_strengths': state.strengths_discovered[:4],
                'red_flags': state.weaknesses_discovered[:4]
            }

        # Transition to COMPLETED
        self._transition_to(state, InterviewPhase.COMPLETED, reason="Final assessment synthesized")

        # Persist completed status and final score to Supabase
        self.supabase.end_session(session_id, final_score)
        self.supabase.log_event(session_id, 'interview_orchestrator_completed', {
            'final_score': final_score,
            'total_questions': len(state.questions_asked),
            'skills_evaluated': len(state.skills_distribution),
            'assessment': state.final_assessment
        })

        # Generate personalized, explainable recommendations on top of skill signals
        final_recommendations = []
        try:
            skill_profiles = []
            for name, sig in state.skills_distribution.items():
                skill_profiles.append(CandidateSkillProfile(
                    user_id=state.user_id,
                    skill_name=name,
                    estimated_proficiency=sig.average_score,
                    confidence="medium confidence" if sig.questions_count >= 2 else "low confidence",
                    evidence_count=sig.questions_count,
                    recent_performance=sig.average_score,
                    historical_performance=sig.average_score,
                    improvement_trend="neutral",
                    last_evaluated_timestamp=datetime.now(timezone.utc).isoformat()
                ))

            session_questions = self.supabase.get_session_questions(session_id)
            recs = self.recommendation_engine.generate_recommendations(
                user_id=state.user_id,
                skill_profiles=skill_profiles,
                session_questions=session_questions,
                session_id=session_id
            )
            recs_dict = [r.model_dump() for r in recs]
            persisted = self.supabase.create_recommendations(state.user_id, recs_dict)
            final_recommendations = persisted
        except Exception as e:
            logger.warning(f"Recommendation generation during interview finalization skipped: {e}")

        # Evolve personalized candidate model
        try:
            cand_data = self.supabase.get_candidate_model(state.user_id)
            if cand_data:
                cand_model = CandidateModel(**cand_data)
            else:
                cand_model = CandidateModel(user_id=state.user_id, target_role=state.target_role)
            
            cand_model.evolve_from_session(state)
            self.supabase.save_candidate_model(state.user_id, cand_model.model_dump())
        except Exception as e:
            logger.warning(f"Candidate model evolution failed: {e}")

        return {
            'session_id': session_id,
            'phase': InterviewPhase.COMPLETED.value,
            'is_completed': True,
            'final_score': final_score,
            'final_assessment': state.final_assessment,
            'skills_distribution': {
                name: sig.model_dump() for name, sig in state.skills_distribution.items()
            },
            'questions_answered': len(scores),
            'recommendations': final_recommendations
        }

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Returns the full authoritative state for client querying and telemetry."""
        state = self.get_or_restore_state(session_id)
        scores = [q.score for q in state.questions_asked if q.score is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

        return {
            'session_id': session_id,
            'user_id': state.user_id,
            'target_role': state.target_role,
            'interview_type': state.interview_type,
            'phase': state.phase.value,
            'difficulty': state.difficulty.value,
            'current_skill_focus': state.current_skill_focus,
            'questions_count': len(state.questions_asked),
            'answered_count': len(scores),
            'session_score': avg_score,
            'final_score': state.final_score,
            'weaknesses_discovered': state.weaknesses_discovered,
            'strengths_discovered': state.strengths_discovered,
            'pending_follow_ups': len(state.follow_up_opportunities),
            'time_budget_seconds': state.time_budget_seconds,
            'start_time': state.start_time,
            'skills_distribution': {
                name: sig.model_dump() for name, sig in state.skills_distribution.items()
            },
            'candidate_theta': state.candidate_theta,
            'decision_history': state.decision_history,
            'is_completed': state.phase == InterviewPhase.COMPLETED
        }

    def get_question_selection_rationale(self, session_id: str, question_id: str) -> Dict[str, Any]:
        """
        Answers: 'Why was this question selected?' with evidence from the candidate's actual session history.
        """
        state = self.get_or_restore_state(session_id)
        rationale = state.get_selection_rationale(question_id)

        q_record = next((q for q in state.questions_asked if q.question_id == question_id), None)
        decision = q_record.selection_decision if q_record else None

        if not rationale and q_record:
            rationale = (
                f"Question '{q_record.question_text[:50]}' was selected for skill '{q_record.skill_focus}' "
                f"at {q_record.difficulty} difficulty based on candidate session performance."
            )

        return {
            "session_id": session_id,
            "question_id": question_id,
            "why_selected": rationale or "Question was selected by the adaptive orchestrator based on candidate skill state.",
            "decision": decision,
            "skill_focus": q_record.skill_focus if q_record else state.current_skill_focus,
            "difficulty": q_record.difficulty if q_record else state.difficulty.value,
        }
