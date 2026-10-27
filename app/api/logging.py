"""
API blueprint — session event and anomaly logging, health, and metrics.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, request
from pydantic import ValidationError

from app.security import require_auth, verify_session_ownership, unauthorized_response
from app.schemas import LogEventRequest, LogAnomalyRequest, success, error
from app.services.supabase_service import SupabaseService

logger = logging.getLogger(__name__)

logging_bp = Blueprint("logging", __name__)

_supabase = SupabaseService()
# Keep for test backwards-compatibility
supabase_service = _supabase

VALID_EVENT_TYPES = {
    "session_started", "session_ended", "question_generated",
    "answer_submitted", "evaluation_completed", "user_action",
    "system_event", "error_occurred", "performance_metric",
}

VALID_ANOMALY_TYPES = {
    "low_score", "rapid_answers", "copy_paste", "multiple_sessions",
    "unusual_pattern", "system_abuse", "performance_issue",
    "session_integrity_flag",
}

VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


@logging_bp.route("/log_event", methods=["POST"])
@require_auth
def log_event():
    """Persist a session lifecycle event."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    event_type = data.get("event_type", "").strip()

    if not session_id:
        return jsonify(error("session_id is required", "MISSING_SESSION_ID")), 400
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    if not event_type:
        return jsonify(error("event_type is required", "MISSING_EVENT_TYPE")), 400

    details = data.get("details", {})
    ok = _supabase.log_event(session_id, event_type, details)
    if not ok:
        return jsonify(error("Failed to log event", "EVENT_LOGGING_FAILED")), 500

    return jsonify(success({
        "event_type": event_type,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, "Event logged")), 200


@logging_bp.route("/log_anomaly", methods=["POST"])
@require_auth
def log_anomaly():
    """Persist a session integrity anomaly."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    anomaly_type = data.get("anomaly_type", "").strip()
    severity = data.get("severity", "").upper().strip()

    if not session_id:
        return jsonify(error("session_id is required", "MISSING_SESSION_ID")), 400
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    if not anomaly_type:
        return jsonify(error("anomaly_type is required", "MISSING_ANOMALY_TYPE")), 400
    if severity not in VALID_SEVERITIES:
        return jsonify(error(f"severity must be one of {sorted(VALID_SEVERITIES)}", "INVALID_SEVERITY")), 400

    details = data.get("details", {})
    ok = _supabase.log_anomaly(session_id, anomaly_type, severity, details)
    if not ok:
        return jsonify(error("Failed to log anomaly", "ANOMALY_LOGGING_FAILED")), 500

    logger.warning("Anomaly logged: %s (%s) for session %s", anomaly_type, severity, session_id)
    return jsonify(success({
        "anomaly_type": anomaly_type,
        "severity": severity,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, "Anomaly logged")), 200


@logging_bp.route("/events/<session_id>", methods=["GET"])
@require_auth
def get_session_events(session_id: str):
    """Return all session lifecycle events for a session."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    stats = _supabase.get_session_statistics(session_id)
    events = stats.get("events", [])
    return jsonify(success({
        "session_id": session_id,
        "events": events,
        "total_count": len(events),
    }, "Events retrieved")), 200


@logging_bp.route("/anomalies/<session_id>", methods=["GET"])
@require_auth
def get_session_anomalies(session_id: str):
    """Return all integrity events for a session."""
    if not verify_session_ownership(session_id, g.user_id):
        return unauthorized_response("You do not have access to this session.")
    stats = _supabase.get_session_statistics(session_id)
    anomalies = stats.get("anomalies", [])
    return jsonify(success({
        "session_id": session_id,
        "anomalies": anomalies,
        "total_count": len(anomalies),
    }, "Anomalies retrieved")), 200


@logging_bp.route("/health", methods=["GET"])
def health_check():
    """Database and API liveness probe."""
    db = _supabase.health_check()
    status = "healthy" if db.get("status") == "healthy" else "unhealthy"
    http_status = 200 if status == "healthy" else 503
    payload = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {"database": db.get("status", "unknown"), "api": "healthy"},
    }
    if db.get("error"):
        payload["database_error"] = str(db.get("error"))
    return jsonify(payload), http_status


@logging_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """Basic runtime metrics (placeholder — extend with real counters)."""
    return jsonify(success({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Extend with Prometheus or a real metrics store",
    }, "Metrics retrieved")), 200
