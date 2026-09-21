"""
Maintenance ML evaluation and block scheduling endpoints.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_ml_engine
from src.api.schemas.requests import EvaluateRequest, RecommendWindowsRequest
from src.services.ml_engine import RailwayMLEngine

router = APIRouter(tags=["ml"])


def _tasks_to_dicts(request) -> tuple[list[dict], list[dict] | None]:
    """Convert Pydantic task models to plain dicts for the engine."""
    tasks = [t.model_dump(exclude_none=True) for t in request.tasks]
    block_windows = None
    if request.block_windows is not None:
        block_windows = [bw.model_dump() for bw in request.block_windows]
    return tasks, block_windows


@router.post("/evaluate")
def evaluate(
    request: EvaluateRequest,
    engine: RailwayMLEngine = Depends(get_ml_engine),
):
    """Run full ML evaluation: risk + priority + explanation + scheduling.

    Delegates to RailwayMLEngine.evaluate_for_backend().
    """
    # Check model availability
    try:
        health = engine.health()
        model_available = health.get("artifacts", {}).get(
            "calibrated_xgboost_available", False
        )
    except Exception:
        model_available = False

    if not model_available:
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_UNAVAILABLE",
                "detail": "Production ML model is not loaded",
                "status_code": 503,
            },
        )

    tasks, block_windows = _tasks_to_dicts(request)

    try:
        result = engine.evaluate_for_backend(
            input_data=tasks,
            block_windows=block_windows,
            max_recommendations=request.max_recommendations,
        )
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "detail": f"ML evaluation failed: {type(exc).__name__}",
                "status_code": 500,
            },
        )


@router.post("/recommend-windows")
def recommend_windows(
    request: RecommendWindowsRequest,
    engine: RailwayMLEngine = Depends(get_ml_engine),
):
    """Generate CP-SAT scheduling recommendations for maintenance tasks.

    Delegates to RailwayMLEngine.recommend_windows().
    """
    # Check model availability
    try:
        health = engine.health()
        model_available = health.get("artifacts", {}).get(
            "calibrated_xgboost_available", False
        )
    except Exception:
        model_available = False

    if not model_available:
        return JSONResponse(
            status_code=503,
            content={
                "error": "MODEL_UNAVAILABLE",
                "detail": "Production ML model is not loaded",
                "status_code": 503,
            },
        )

    tasks, block_windows = _tasks_to_dicts(request)

    try:
        # Compute failure probabilities and priority scores prior to window optimization
        scored = engine.predict(input_data=tasks)

        result = engine.recommend_windows(
            input_data=scored,
            block_windows=block_windows,
            max_recommendations=request.max_recommendations,
        )

        # Convert any DataFrame values to serializable dicts
        serializable = {}
        for key, value in result.items():
            if hasattr(value, "to_dict"):
                serializable[key] = value.to_dict(orient="records")
            else:
                serializable[key] = value

        return serializable
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "detail": f"Scheduling failed: {type(exc).__name__}",
                "status_code": 500,
            },
        )
