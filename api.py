"""Compatibility entry point for local uvicorn runs.

The canonical FastAPI application lives in app.main after the upstream FastAPI
pipeline integration. Keeping this shim allows the older `uvicorn api:app`
command to keep working without duplicating routes.
"""

from app.main import app

__all__ = ["app"]
