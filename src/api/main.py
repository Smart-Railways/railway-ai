"""
Railway-AI ML Service — FastAPI application.

HTTP routing, request validation, CORS, and response serialization
for the RailwayMLEngine service.
"""

import os

from fastapi import FastAPI

from src.api.routes import health, ml

app = FastAPI(
    title="Railway-AI ML Service",
    description="Thin HTTP interface around RailwayMLEngine for railway maintenance ML evaluation, priority scoring, and CP-SAT scheduling.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — disabled by default. Set ML_CORS_ORIGINS to a comma-separated list
# of allowed origins to enable (e.g. "http://localhost:3000,https://app.example.com").
# ---------------------------------------------------------------------------
_cors_origins_raw = os.environ.get("ML_CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(ml.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ML_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("ML_SERVICE_PORT", "8000"))
    uvicorn.run("src.api.main:app", host=host, port=port)
