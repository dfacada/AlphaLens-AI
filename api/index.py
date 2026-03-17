"""
Vercel serverless entry point.
Re-exports the FastAPI `app` from server.py so Vercel can discover it.
"""
from api.server import app  # noqa: F401
