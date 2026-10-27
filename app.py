#!/usr/bin/env python3
"""
Interview AI Platform — Backend Application Entry Point.
"""
from __future__ import annotations

import logging
import os
from app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Application factory instantiation
app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")

    logger.info("Starting Interview AI Platform Backend on %s:%d (debug=%s)", host, port, debug)
    app.run(host=host, port=port, debug=debug)
