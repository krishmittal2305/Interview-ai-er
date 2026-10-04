"""
Application factory.

Creates and configures the Flask application.
All blueprints are registered here with their URL prefixes.

Environment variables required:
  SUPABASE_URL      Supabase project URL
  SUPABASE_KEY      Supabase service role key
  GEMINI_API_KEY    Google Gemini API key
  FLASK_SECRET_KEY  Flask session secret (optional in dev)
  FRONTEND_URL      Allowed CORS origin (default: http://localhost:3000)
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)

    # ── Configuration ─────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")
    app.config["SUPABASE_URL"] = os.getenv("SUPABASE_URL")
    app.config["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
    app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    app.config["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    app.config["GROQ_MODEL"] = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    app.config["AI_PROVIDER"] = os.getenv("AI_PROVIDER", "groq")
    app.config["FRONTEND_URL"] = os.getenv("FRONTEND_URL", "http://localhost:3000")

    _validate_env(app)

    # ── CORS ──────────────────────────────────────────────────────
    CORS(app, origins=[app.config["FRONTEND_URL"]], supports_credentials=True)

    # ── Blueprints ────────────────────────────────────────────────
    # Primary location: app/api/ (new structured layout)
    from app.api.interview import interview_bp
    from app.api.logging import logging_bp
    from app.api.intelligence import intelligence_bp
    from app.api.analytics import analytics_bp
    from app.api.ml_health import ml_health_bp
    from app.api.observability import observability_bp

    app.register_blueprint(interview_bp, url_prefix="/api")
    app.register_blueprint(logging_bp, url_prefix="/api")
    app.register_blueprint(intelligence_bp, url_prefix="/api/intelligence")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(ml_health_bp, url_prefix="/api/ml")
    app.register_blueprint(observability_bp, url_prefix="/api/observability")

    # ── Error handlers ───────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(exc):
        return jsonify({"error": True, "message": "Bad request", "code": "BAD_REQUEST"}), 400

    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": True, "message": "Not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"error": True, "message": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("Unhandled exception")
        return jsonify({"error": True, "message": "Internal server error", "code": "INTERNAL_ERROR"}), 500

    logger.info("Application factory: app created with %d blueprints", len(app.blueprints))
    return app


def _validate_env(app: Flask) -> None:
    """Raise a clear error if required environment variables are missing."""
    required = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing = [k for k in required if not app.config.get(k)]
    if not (app.config.get("GROQ_API_KEY") or app.config.get("GEMINI_API_KEY")):
        missing.append("GROQ_API_KEY or GEMINI_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy env.example to .env and fill in the values."
        )
