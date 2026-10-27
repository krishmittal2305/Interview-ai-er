"""
WSGI Entry Point for Production Deployments (Render, Gunicorn, Heroku).
Avoids package name collision between the root app.py script and the app/ directory package.
"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
