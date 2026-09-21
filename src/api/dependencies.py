"""
ML engine singleton dependency for FastAPI.

Provides a single shared RailwayMLEngine instance via FastAPI's
dependency injection. The engine is initialized once on first request.
"""

from src.services.ml_engine import RailwayMLEngine

_engine: RailwayMLEngine | None = None


def get_ml_engine() -> RailwayMLEngine:
    """Return the shared RailwayMLEngine singleton."""
    global _engine
    if _engine is None:
        _engine = RailwayMLEngine()
    return _engine
