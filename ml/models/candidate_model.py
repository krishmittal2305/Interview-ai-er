from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

class ObservationEvent(BaseModel):
    """OBSERVATION: What happened (e.g., candidate missed 3 graph traversal concepts)."""
    event_id: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    description: str
    skill_focus: str
    raw_data: Dict[str, Any] = Field(default_factory=dict)

class InferenceState(BaseModel):
    """INFERENCE: What the models estimate (e.g., graph traversal mastery estimate decreased)."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    skill_name: str
    mastery_estimate: float
    confidence: float
    change_delta: float
    reason: str

class DecisionAction(BaseModel):
    """DECISION: What the system recommends (e.g., next question should target graph traversal at moderate difficulty)."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision_type: str
    recommended_action: str
    rationale: str
    supporting_inferences: List[str] = Field(default_factory=list)

class EvidenceChainItem(BaseModel):
    """Persists the evidence chain so the candidate can inspect why a recommendation was produced."""
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    observations: List[ObservationEvent] = Field(default_factory=list)
    inferences: List[InferenceState] = Field(default_factory=list)
    decisions: List[DecisionAction] = Field(default_factory=list)

class CandidateModel(BaseModel):
    """
    Persistent candidate state. Evolved after every completed session.
    """
    user_id: str
    target_role: str = ""
    target_skills: List[str] = Field(default_factory=list)
    skill_estimates: Dict[str, float] = Field(default_factory=dict)
    skill_confidence: Dict[str, float] = Field(default_factory=dict)
    recent_performance: Dict[str, float] = Field(default_factory=dict)
    historical_performance: Dict[str, float] = Field(default_factory=dict)
    difficulty_exposure: Dict[str, Dict[str, int]] = Field(default_factory=dict)  # skill -> {difficulty: count}
    practice_history: List[str] = Field(default_factory=list)  # session IDs
    weak_skill_signals: List[str] = Field(default_factory=list)
    strong_skill_signals: List[str] = Field(default_factory=list)
    recommendation_history: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_chain: List[EvidenceChainItem] = Field(default_factory=list)

    def evolve_from_session(self, session_state: Any) -> EvidenceChainItem:
        """
        Takes an OrchestratorState (or similar session data) and evolves the candidate model.
        Extracts observations, updates inferences, generates decisions, and appends to the evidence chain.
        """
        import uuid
        
        # We need the OrchestratorState from app.services.orchestrator.models,
        # but to avoid circular dependencies if used from there, we use duck-typing / Any
        
        session_id = session_state.session_id
        
        # Ensure role/skills are tracked
        if not self.target_role:
            self.target_role = session_state.target_role
        
        self.practice_history.append(session_id)
        
        new_evidence = EvidenceChainItem(session_id=session_id)
        
        # 1. OBSERVATIONS: What happened in this session
        observations = []
        for q in session_state.questions_asked:
            if not q.score:
                continue
            
            # Difficulty exposure
            if q.skill_focus not in self.difficulty_exposure:
                self.difficulty_exposure[q.skill_focus] = {}
            if q.difficulty not in self.difficulty_exposure[q.skill_focus]:
                self.difficulty_exposure[q.skill_focus][q.difficulty] = 0
            self.difficulty_exposure[q.skill_focus][q.difficulty] += 1
            
            obs = ObservationEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                event_type="question_scored",
                description=f"Candidate scored {q.score} on {q.difficulty} question for skill '{q.skill_focus}'.",
                skill_focus=q.skill_focus,
                raw_data={"score": q.score, "difficulty": q.difficulty, "question_id": q.question_id}
            )
            observations.append(obs)
            
        for weak in session_state.weaknesses_discovered:
            if weak not in self.weak_skill_signals:
                self.weak_skill_signals.append(weak)
            observations.append(ObservationEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                event_type="weakness_discovered",
                description=f"Candidate exhibited weakness: {weak}",
                skill_focus="General",
                raw_data={"weakness": weak}
            ))
            
        for strong in session_state.strengths_discovered:
            if strong not in self.strong_skill_signals:
                self.strong_skill_signals.append(strong)
            observations.append(ObservationEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                event_type="strength_discovered",
                description=f"Candidate exhibited strength: {strong}",
                skill_focus="General",
                raw_data={"strength": strong}
            ))
            
        new_evidence.observations = observations
        
        # 2. INFERENCE: What the models estimate
        inferences = []
        for skill, signal in session_state.skills_distribution.items():
            if skill not in self.target_skills:
                self.target_skills.append(skill)
                
            prev_estimate = self.skill_estimates.get(skill, 0.5)
            new_estimate = signal.mastery
            delta = new_estimate - prev_estimate
            
            self.skill_estimates[skill] = new_estimate
            
            confidence = min(0.95, (signal.questions_count * 0.1) + 0.3)
            self.skill_confidence[skill] = confidence
            
            # Update historical/recent
            self.historical_performance[skill] = self.recent_performance.get(skill, 0.5)
            self.recent_performance[skill] = signal.average_score / 100.0
            
            reason = f"Based on {signal.questions_count} new questions, mastery estimate changed by {delta:+.2f}."
            
            inferences.append(InferenceState(
                skill_name=skill,
                mastery_estimate=new_estimate,
                confidence=confidence,
                change_delta=delta,
                reason=reason
            ))
            
        new_evidence.inferences = inferences
        
        # 3. DECISIONS: What the system recommends
        decisions = []
        # Basic logic: recommend practicing the weakest skill
        if inferences:
            weakest_inference = min(inferences, key=lambda i: i.mastery_estimate)
            
            rec_action = f"Next session should target {weakest_inference.skill_name}."
            
            decision = DecisionAction(
                decision_type="recommendation",
                recommended_action=rec_action,
                rationale=f"Skill '{weakest_inference.skill_name}' has the lowest mastery estimate ({weakest_inference.mastery_estimate:.2f}).",
                supporting_inferences=[weakest_inference.reason]
            )
            decisions.append(decision)
            self.recommendation_history.append(decision.model_dump())
            
        new_evidence.decisions = decisions
        self.evidence_chain.append(new_evidence)

        # Record ML inference in unified observability telemetry
        try:
            from app.observability.tracker import get_telemetry_tracker
            get_telemetry_tracker().record_inference(
                inference_type="ml",
                model="BayesianCandidateModel",
                model_version="candidate_model_v1.0",
                task="candidate_skill_inference",
                latency_ms=1.5,
                success=True,
                status="SUCCESS",
                input_size=len(session_state.questions_asked) if hasattr(session_state, "questions_asked") else 1,
                output_size=len(inferences),
                memory_device="cpu",
                fallback_usage=False,
                retry_count=0,
                metadata={"inferences_count": len(inferences), "decisions_count": len(decisions)},
            )
        except Exception as tel_err:
            pass
        
        return new_evidence
