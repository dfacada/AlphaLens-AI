"""
Vercel serverless entry point.
Re-exports the FastAPI `app` from server.py so Vercel can discover it.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for Vercel's runtime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import app  # noqa: F401
