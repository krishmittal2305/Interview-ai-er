"""
API blueprint — interview session, evaluation, and code execution routes.

This file is the HTTP transport layer only.
It validates requests, delegates to services, and serializes responses.
No business logic lives here.
"""
from __future__ import annotations

import logging
import uuid

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError

from app.security import (
    require_auth,
    verify_session_ownership,
    unauthorized_response,
    not_found_response,
    SecurityService,
)
from app.schemas import (
    StartSessionRequest,
    SubmitAnswerRequest,
    SubmitVoiceAnswerRequest,
    SubmitCodeRequest,
    FollowUpRequest,
    SecurityCheckRequest,
    success,
    error,
)
from app.services.supabase_service import SupabaseService
from app.ai import AssessmentEngine, GeminiProvider, GroqProvider, get_ai_provider
from app.services.orchestrator import InterviewOrchestrator
from app.infrastructure import CodeSandbox

logger = logging.getLogger(__name__)

interview_bp = Blueprint("interview", __name__)

# Service singletons — initialized once per worker process
_supabase = SupabaseService()
_ai_engine = AssessmentEngine(get_ai_provider())
_security = SecurityService()
_orchestrator = InterviewOrchestrator(_supabase, _ai_engine)
_sandbox = CodeSandbox(timeout_seconds=5)

# Keep these names for backward compatibility with tests
supabase_service = _supabase
ai_engine = _ai_engine
security_service = _security
orchestrator = _orchestrator
sandbox = _sandbox

# Whisper transcription — lazy-loaded (no model loaded until first request)
_transcription_service = None

def _get_transcription_service():
    """Lazy-init the Whisper service so we don't pay cold-start cost at import."""
    global _transcription_service
    if _transcription_service is None:
        from app.services.transcription import WhisperTranscriptionService
        _transcription_service = WhisperTranscriptionService()
    return _transcription_service


def _parse_json(schema_class):
    """Parse and validate request JSON against a Pydantic schema.
    Returns (instance, None) or (None, error_response_tuple)."""
    data = request.get_json(silent=True)
    if not data:
        return None, (jsonify(error("Request body is missing or not JSON", "NO_BODY")), 400)
    try:
        return schema_class(**data), None
    except ValidationError as exc:
        messages = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors())
        return None, (jsonify(error(messages, "VALIDATION_ERROR")), 422)


# ─────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────

@interview_bp.route("/start_session", methods=["POST"])
@require_auth
def start_session():
    """Create a new interview session and initialize the orchestrator."""
    req, err = _parse_json(StartSessionRequest)
    if err:
        return err

    user_id = g.user_id

    # Normalize legacy type names
    type_aliases = {
        "Technical": "Software Engineer",
        "Coding": "Software Engineer",
        "System Design": "Software Engineer",
        "Behavioral": "Software Engineer",
    }
    interview_type = type_aliases.get(req.interview_type, req.interview_type)

    result = _supabase.create_session(user_id, interview_type)
    if not result.get("success"):
        logger.error("Session creation failed: %s", result.get("error"))
        return jsonify(error("Failed to start session", "SESSION_CREATION_FAILED")), 500

    session_id = result["session_id"]
    state = _orchestrator.initialize_session(
        session_id=session_id,
        user_id=user_id,
        target_role=interview_type,
        interview_type=interview_type,
    )
    _supabase.log_event(session_id, "session_started", {"user_id": user_id, "interview_type": interview_type})

    logger.info("Session started: %s (user=%s, type=%s)", session_id, user_id, interview_type)
    return jsonify(success({
        "session_id": session_id,
        "user_id": user_id,
        "interview_type": interview_type,
        "start_time": result["data"]["start_time"],
        "phase": state.phase.value,
        "difficulty": state.difficulty.value,
        "skills": list(state.skills_distribution.keys()),
    }, "Session started successfully")), 201


@interview_bp.route("/get_question", methods=["GET"])
@require_auth
def get_question():
    """Return the next question from the orchestrator for an active session."""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify(error("session_id is required", "MISSING_SESSION_ID")), 400

    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    session = _supabase.get_session(session_id)
    if not session:
        return not_found_response("Session not found.")
    if session.get("status") != "active":
        return jsonify(error("Session is not active", "SESSION_INACTIVE")), 400

    result = _orchestrator.get_next_question(session_id)
    return jsonify(success(result, "Question retrieved successfully")), 200


@interview_bp.route("/submit_answer", methods=["POST"])
@require_auth
def submit_answer():
    """Submit a text answer for evaluation."""
    req, err = _parse_json(SubmitAnswerRequest)
    if err:
        return err

    if not verify_session_ownership(req.session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    session = _supabase.get_session(req.session_id)
    if not session or session.get("status") != "active":
        return jsonify(error("Session not found or not active", "SESSION_INVALID")), 404

    if len(req.answer_text) < 10:
        return jsonify(error("Answer must be at least 10 characters", "ANSWER_TOO_SHORT")), 400

    try:
        result = _orchestrator.record_and_evaluate_answer(
            session_id=req.session_id,
            question_id=req.question_id,
            answer_text=req.answer_text,
            is_code=False,
            transcription_metadata=req.transcription_metadata,
        )
    except Exception as e:
        logger.error(f"Error evaluating answer for session {req.session_id}: {e}", exc_info=True)
        return jsonify(error(f"Answer evaluation failed: {str(e)}", "EVALUATION_FAILED")), 500

    if req.input_modality == "voice" or req.transcription_metadata:
        meta = req.transcription_metadata or {}
        _supabase.log_event(req.session_id, "voice_answer_submitted", {
            "question_id": req.question_id,
            "input_modality": "voice",
            "transcript": req.answer_text,
            "transcription_model": meta.get("model_id"),
            "language": meta.get("language"),
            "duration": meta.get("duration_seconds") or meta.get("duration"),
            "timestamp": meta.get("timestamp"),
            "confidence": meta.get("confidence"),
            "latency_ms": meta.get("latency_ms"),
        })

    return jsonify(success({
        "question_id": req.question_id,
        "evaluation": result["evaluation"],
        "session_score": result["session_score"],
        "current_difficulty": result["current_difficulty"],
        "difficulty_adjusted": result["difficulty_adjusted"],
        "current_phase": result["current_phase"],
        "is_last_question": result["is_last_question"],
        "skills_overview": result["skills_overview"],
    }, "Answer submitted successfully")), 200



@interview_bp.route("/submit_code", methods=["POST"])
@require_auth
def submit_code():
    """Submit code for AI evaluation and persist the submission."""
    req, err = _parse_json(SubmitCodeRequest)
    if err:
        return err

    if not verify_session_ownership(req.session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    session = _supabase.get_session(req.session_id)
    if not session or session.get("status") != "active":
        return jsonify(error("Session not active", "SESSION_INACTIVE")), 400

    try:
        result = _orchestrator.record_and_evaluate_answer(
            session_id=req.session_id,
            question_id=req.question_id,
            answer_text=req.code,
            is_code=True,
            language=req.language,
        )
    except Exception as e:
        logger.error(f"Error evaluating code for session {req.session_id}: {e}", exc_info=True)
        return jsonify(error(f"Code evaluation failed: {str(e)}", "EVALUATION_FAILED")), 500

    return jsonify(success({
        "evaluation": result["evaluation"],
        "session_score": result["session_score"],
        "current_difficulty": result["current_difficulty"],
        "current_phase": result["current_phase"],
        "is_last_question": result["is_last_question"],
    }, "Code submitted successfully")), 200


@interview_bp.route("/run_code", methods=["POST"])
@require_auth
def run_code():
    """Execute code in the sandbox without creating a submission record."""
    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    language = data.get("language", "").strip()
    if not code or not language:
        return jsonify(error("code and language are required", "MISSING_FIELDS")), 400

    result = _sandbox.execute(code, language)
    return jsonify(success(result, "Code executed")), 200


# ─────────────────────────────────────────────────────────────────
# Voice / Transcription
# ─────────────────────────────────────────────────────────────────

@interview_bp.route("/transcribe", methods=["POST"])
@require_auth
def transcribe_audio():
    """
    Accept an audio file upload and return a Whisper transcription.

    The frontend sends a multipart/form-data request with:
      - file: the audio blob (WAV or WebM)
    Returns the transcript text plus all metadata so the user can
    edit the transcript before submitting it as an answer.
    """
    if "file" not in request.files:
        return jsonify(error("No audio file provided", "NO_AUDIO_FILE")), 400

    audio_file = request.files["file"]
    content_type = audio_file.content_type or "audio/wav"
    audio_bytes = audio_file.read()

    if not audio_bytes or len(audio_bytes) < 100:
        return jsonify(error("Audio file is empty or too small", "AUDIO_TOO_SMALL")), 400

    try:
        svc = _get_transcription_service()
        result = svc.transcribe(audio_bytes, content_type=content_type)

        if result.error:
            logger.error("Transcription error: %s", result.error)
            return jsonify(error(result.error, "TRANSCRIPTION_FAILED")), 500

        return jsonify(success({
            "transcript": result.transcript,
            "language": result.language,
            "duration_seconds": result.duration_seconds,
            "model_id": result.model_id,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "timestamp": result.timestamp,
            "segments": result.segments,
        }, "Transcription completed")), 200

    except Exception as exc:
        logger.exception("Transcription endpoint failed")
        return jsonify(error(str(exc), "TRANSCRIPTION_ERROR")), 500


@interview_bp.route("/submit_voice_answer", methods=["POST"])
@require_auth
def submit_voice_answer():
    """
    Submit a voice-transcribed answer for evaluation.

    This route accepts the (possibly user-edited) transcript and pushes
    it through the *same* evaluation pipeline as typed answers.
    Voice is treated as another input modality — not a separate evaluator.
    """
    req, err = _parse_json(SubmitVoiceAnswerRequest)
    if err:
        return err

    if not verify_session_ownership(req.session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    session = _supabase.get_session(req.session_id)
    if not session or session.get("status") != "active":
        return jsonify(error("Session not found or not active", "SESSION_INVALID")), 404

    if len(req.transcript) < 10:
        return jsonify(error("Transcript must be at least 10 characters", "TRANSCRIPT_TOO_SHORT")), 400

    # Feed transcript through the exact same pipeline as typed text answers
    result = _orchestrator.record_and_evaluate_answer(
        session_id=req.session_id,
        question_id=req.question_id,
        answer_text=req.transcript,
        is_code=False,
        transcription_metadata=req.transcription_metadata,
    )

    # Log the voice-specific metadata for audit / analytics
    meta = req.transcription_metadata or {}
    _supabase.log_event(req.session_id, "voice_answer_submitted", {
        "question_id": req.question_id,
        "input_modality": "voice",
        "transcript": req.transcript,
        "transcription_model": meta.get("model_id"),
        "language": meta.get("language"),
        "duration": meta.get("duration_seconds") or meta.get("duration"),
        "timestamp": meta.get("timestamp"),
        "confidence": meta.get("confidence"),
        "latency_ms": meta.get("latency_ms"),
    })


    return jsonify(success({
        "question_id": req.question_id,
        "input_modality": "voice",
        "evaluation": result["evaluation"],
        "session_score": result["session_score"],
        "current_difficulty": result["current_difficulty"],
        "difficulty_adjusted": result["difficulty_adjusted"],
        "current_phase": result["current_phase"],
        "is_last_question": result["is_last_question"],
        "skills_overview": result["skills_overview"],
        "transcription_metadata": req.transcription_metadata,
    }, "Voice answer submitted successfully")), 200


# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────

@interview_bp.route("/session/<session_id>", methods=["GET"])
@require_auth
def get_session_details(session_id: str):
    """Return aggregated session statistics, questions, events, and anomalies."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    stats = _supabase.get_session_statistics(session_id)
    return jsonify(success(stats, "Session details retrieved")), 200


@interview_bp.route("/session/<session_id>/state", methods=["GET"])
@require_auth
def get_session_orchestrator_state(session_id: str):
    """Return backend-authoritative orchestrator state for the session."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    try:
        state = _orchestrator.get_session_state(session_id)
        return jsonify(success(state, "Session state retrieved")), 200
    except Exception as e:
        logger.warning("Session state retrieval failed for %s (%s). Attempting initialization recovery.", session_id, e)
        session = _supabase.get_session(session_id)
        target_role = (session or {}).get("interview_type", "Software Engineer")
        _orchestrator.initialize_session(session_id, g.user_id, target_role, target_role)
        state = _orchestrator.get_session_state(session_id)
        return jsonify(success(state, "Session state retrieved")), 200


@interview_bp.route("/end_session/<session_id>", methods=["POST"])
@require_auth
def end_session(session_id: str):
    """Finalize a session, trigger synthesis, and mark it as completed."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    session = _supabase.get_session(session_id)
    if not session:
        return not_found_response("Session not found.")
    if session.get("status") == "completed":
        return jsonify(error("Session is already completed", "SESSION_ALREADY_COMPLETED")), 400

    final = _orchestrator.finalize_interview(session_id)
    return jsonify(success(final, "Session ended successfully")), 200


@interview_bp.route("/session/<session_id>/questions/<question_id>/selection_rationale", methods=["GET"])
@require_auth
def get_question_selection_rationale(session_id: str, question_id: str):
    """Return the evidence-based rationale answering: 'Why was this question selected?'."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    rationale_data = _orchestrator.get_question_selection_rationale(session_id, question_id)
    return jsonify(success(rationale_data, "Question selection rationale retrieved")), 200


@interview_bp.route("/user/<user_id>/sessions", methods=["GET"])
@require_auth
def get_user_sessions(user_id: str):
    """List recent interview sessions for the authenticated user."""
    if user_id != g.user_id:
        return unauthorized_response("You can only access your own sessions.")
    limit = min(max(request.args.get("limit", 10, type=int), 1), 50)
    sessions = _supabase.get_user_sessions(user_id, limit)
    return jsonify(success(
        {"user_id": user_id, "sessions": sessions, "total_count": len(sessions)},
        "Sessions retrieved",
    )), 200


# ─────────────────────────────────────────────────────────────────
# Follow-up and practice
# ─────────────────────────────────────────────────────────────────

@interview_bp.route("/follow_up_question", methods=["POST"])
@require_auth
def get_follow_up_question():
    """Generate a targeted follow-up question for a given Q&A pair."""
    req, err = _parse_json(FollowUpRequest)
    if err:
        return err
    follow_up_data = _ai_engine.generate_follow_up_question(req.question, req.answer, {})
    follow_up = follow_up_data.get("question_text", str(follow_up_data))
    
    question_id = str(uuid.uuid4())
    ml_skills = None
    try:
        from app.services.question_skill_service import QuestionSkillService
        skill_svc = QuestionSkillService(_supabase)
        if skill_svc.is_enabled():
            ml_skills = skill_svc.predict_and_persist(question_id, follow_up)
    except Exception as e:
        logger.error(f"Failed to derive ML skills for follow-up question: {e}")

    return jsonify(success({
        "question_id": question_id,
        "follow_up_question": follow_up,
        "context": {"original_question": req.question, "interview_type": req.interview_type},
        "ml_skills": ml_skills,
        "predicted_skills": ml_skills.get("predicted_skills", []) if ml_skills else []
    }, "Follow-up question generated")), 200


@interview_bp.route("/practice/sequence", methods=["POST"])
@require_auth
def practice_sequence():
    """Generate an adaptive practice sequence based on candidate state and chosen mode."""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "weakest_skills")
    target_role = data.get("target_role", "Software Engineer")
    num_questions = int(data.get("num_questions", 4))

    # Retrieve candidate model
    cand_dict = _supabase.get_candidate_model(g.user_id) if hasattr(g, "user_id") and g.user_id else None

    from ml.models.adaptive_practice import AdaptivePracticeEngine
    engine = AdaptivePracticeEngine()
    plan = engine.generate_practice_sequence(
        candidate_data=cand_dict,
        mode=mode,
        target_role=target_role,
        num_questions=num_questions
    )

    return jsonify(success(plan.model_dump(), "Adaptive practice sequence generated")), 200


@interview_bp.route("/practice/coding", methods=["POST"])
@require_auth
def practice_coding():
    """Generate an adaptive practice question outside of a formal session using AdaptivePracticeEngine."""
    data = request.get_json(silent=True) or {}
    interview_type = data.get("interview_type", "Software Engineer").strip()
    requested_difficulty = data.get("difficulty")
    topic = data.get("topic")
    mode = data.get("mode", "weakest_skills")
    
    cand_dict = _supabase.get_candidate_model(g.user_id) if hasattr(g, "user_id") and g.user_id else None

    from ml.models.adaptive_practice import AdaptivePracticeEngine
    engine = AdaptivePracticeEngine()
    selected_item = engine.select_next_practice_question(
        candidate_data=cand_dict,
        mode=mode,
        target_role=interview_type,
        preferred_topic=topic,
        preferred_difficulty=requested_difficulty
    )

    # ML Difficulty Assessment
    from app.services.question_difficulty_service import QuestionDifficultyService
    ml_diff_service = QuestionDifficultyService()
    if ml_diff_service.is_enabled():
        ml_result = ml_diff_service.assess_difficulty(selected_item.question_text)
        final_difficulty = ml_result.get("predicted_difficulty", selected_item.difficulty)
        ml_metadata = ml_result
    else:
        final_difficulty = selected_item.difficulty
        ml_metadata = None

    # ML Skill Classification & Persistence
    ml_skills = None
    try:
        from app.services.question_skill_service import QuestionSkillService
        skill_svc = QuestionSkillService(_supabase)
        if skill_svc.is_enabled():
            ml_skills = skill_svc.predict_and_persist(selected_item.question_id, selected_item.question_text)
    except Exception as e:
        logger.error(f"Failed to derive and persist ML skills for coding practice: {e}")
        
    return jsonify(success({
        "question_id": selected_item.question_id,
        "title": selected_item.title,
        "question": selected_item.question_text,
        "interview_type": selected_item.interview_type,
        "difficulty": final_difficulty,
        "topic": selected_item.skill_focus,
        "selection_reason": selected_item.selection_reason,
        "remediation_objective": selected_item.remediation_objective,
        "progression_stage": selected_item.progression_stage,
        "expected_time_minutes": selected_item.expected_time_minutes,
        "rubric": selected_item.rubric,
        "ml_metadata": ml_metadata,
        "ml_skills": ml_skills,
        "predicted_skills": ml_skills.get("predicted_skills", []) if ml_skills else selected_item.canonical_skills
    }, "Adaptive practice question generated")), 200


@interview_bp.route("/practice/evaluate", methods=["POST"])
@require_auth
def practice_evaluate():
    """Evaluate a candidate response to a practice drill using the AI engine."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    question_id = data.get("question_id")
    topic = data.get("topic", "General")
    difficulty = data.get("difficulty", "intermediate")

    if not question or not answer:
        return jsonify(error("question and answer are required", "MISSING_FIELDS")), 400

    evaluation = _ai_engine.evaluate_answer(question, answer, {})

    # Record practice attempt in candidate state if authenticated
    try:
        user_id = getattr(g, "user_id", None)
        if user_id:
            cand_dict = _supabase.get_candidate_model(user_id)
            if cand_dict:
                from ml.models.candidate_model import CandidateModel, ObservationEvent
                cand_model = CandidateModel(**cand_dict)
                if question_id and question_id not in cand_model.practice_history:
                    cand_model.practice_history.append(question_id)
                # Track difficulty exposure
                if topic not in cand_model.difficulty_exposure:
                    cand_model.difficulty_exposure[topic] = {}
                cand_model.difficulty_exposure[topic][difficulty] = cand_model.difficulty_exposure[topic].get(difficulty, 0) + 1
                
                score = evaluation.get("score") or evaluation.get("overall_score", 70)
                if score < 65:
                    weakness_text = f"Low score ({score:.0f}%) on {difficulty} drill for {topic}"
                    if weakness_text not in cand_model.weak_skill_signals:
                        cand_model.weak_skill_signals.append(weakness_text)
                _supabase.save_candidate_model(user_id, cand_model.model_dump())
    except Exception as e:
        logger.error(f"Failed to record practice attempt in candidate model: {e}")

    return jsonify(success(evaluation, "Practice answer evaluated")), 200


# ─────────────────────────────────────────────────────────────────
# Session integrity
# ─────────────────────────────────────────────────────────────────

@interview_bp.route("/security/check", methods=["POST"])
@require_auth
def security_check():
    """Process a batch of browser events and return a session integrity report."""
    req, err = _parse_json(SecurityCheckRequest)
    if err:
        return err

    if not verify_session_ownership(req.session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")

    report = _security.process_raw_events(req.session_id, req.events)
    if report.review_recommended:
        _supabase.log_anomaly(
            req.session_id,
            "session_integrity_flag",
            "HIGH" if report.aggregation_score > 0.8 else "MEDIUM",
            {
                "aggregation_score": report.aggregation_score,
                "signals": [sig.model_dump() for sig in report.signals],
                "confidence": report.confidence_in_assessment,
            },
        )
    return jsonify(success({
        "review_recommended": report.review_recommended,
        "aggregation_score": report.aggregation_score,
        "signals": [sig.model_dump() for sig in report.signals],
        "summary": report.summary,
        "confidence": report.confidence_in_assessment,
    }, "Integrity check completed")), 200


@interview_bp.route("/security/report/<session_id>", methods=["GET"])
@require_auth
def get_security_report(session_id: str):
    """Return persisted integrity events for a session."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    # Delegated to the logging blueprint via shared service
    return jsonify(error("Use /api/anomalies/<session_id>", "USE_ANOMALIES_ENDPOINT")), 301
