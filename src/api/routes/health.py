"""
Service health and model readiness check endpoints.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_ml_engine
from src.services.ml_engine import RailwayMLEngine

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(engine: RailwayMLEngine = Depends(get_ml_engine)):
    """Check service and ML engine availability."""
    try:
        health = engine.health()
        model_available = health.get("artifacts", {}).get(
            "calibrated_xgboost_available", False
        )
        return {
            "status": health.get("status", "ok"),
            "service": "railway-ai-ml",
            "model_available": model_available,
            "engine_version": health.get("version", "unknown"),
            "components": health.get("components", {}),
            "artifacts": health.get("artifacts", {}),
        }
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "error": "SERVICE_UNAVAILABLE",
                "detail": "ML engine failed to initialize",
                "status_code": 503,
            },
        )
