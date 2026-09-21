"""
API test suite for the Railway-AI FastAPI service.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.dependencies import get_ml_engine

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TASK = {
    "task_id": "TSK-API-001",
    "section_id": "NDL-MTJ-01",
    "department": "ENGINEERING",
    "estimated_duration": 60,
    "required_manpower": 2,
    "predicted_delay_minutes": 5.0,
    "asset_age_years": 15.0,
    "condition_score": 5.0,
    "criticality": 4.0,
    "usage_factor": 0.7,
    "historical_failure_count": 3,
    "historical_downtime_hours": 50.0,
    "days_since_last_failure": 90.0,
}

SAMPLE_TASKS_MULTI = [
    {
        "task_id": "TSK-API-001",
        "section_id": "NDL-MTJ-01",
        "department": "ENGINEERING",
        "estimated_duration": 60,
        "required_manpower": 2,
        "predicted_delay_minutes": 5.0,
        "asset_age_years": 15.0,
        "condition_score": 5.0,
        "criticality": 4.0,
        "usage_factor": 0.7,
        "historical_failure_count": 3,
        "historical_downtime_hours": 50.0,
        "days_since_last_failure": 90.0,
    },
    {
        "task_id": "TSK-API-002",
        "section_id": "NDL-MTJ-02",
        "department": "S&T",
        "estimated_duration": 45,
        "required_manpower": 3,
        "predicted_delay_minutes": 0.0,
        "asset_age_years": 5.0,
        "condition_score": 8.0,
        "criticality": 2.0,
        "usage_factor": 0.4,
        "historical_failure_count": 0,
        "historical_downtime_hours": 0.0,
        "days_since_last_failure": 500.0,
    },
    {
        "task_id": "TSK-API-003",
        "section_id": "NDL-MTJ-01",
        "department": "TRACTION",
        "estimated_duration": 90,
        "required_manpower": 4,
        "predicted_delay_minutes": 15.0,
        "asset_age_years": 25.0,
        "condition_score": 3.0,
        "criticality": 5.0,
        "usage_factor": 0.9,
        "historical_failure_count": 7,
        "historical_downtime_hours": 120.0,
        "days_since_last_failure": 20.0,
    },
]


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health returns 200 with expected structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "railway-ai-ml"
        assert isinstance(data["model_available"], bool)
        assert data["model_available"] is True
        assert "engine_version" in data
        assert "components" in data
        assert "artifacts" in data


# ---------------------------------------------------------------------------
# 2-7. Evaluate endpoint
# ---------------------------------------------------------------------------

class TestEvaluateEndpoint:
    def test_evaluate_valid_request(self):
        """POST /api/v1/evaluate with valid tasks returns 200."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": SAMPLE_TASKS_MULTI},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert "engine_version" in data
        assert data["tasks_evaluated"] == 3
        assert "results" in data
        assert "metrics" in data

    def test_evaluate_response_has_four_categories(self):
        """Each result contains all 4 frozen output categories."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": [SAMPLE_TASK]},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) >= 1
        result = results[0]
        assert "risk" in result
        assert "priority" in result
        assert "explanation" in result
        assert "scheduling" in result
        # Metadata fields
        assert "task_id" in result
        assert "section_id" in result
        assert "department" in result
        assert "decision_rank" in result

    def test_evaluate_risk_fields(self):
        """Risk output has correct fields and value ranges."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": [SAMPLE_TASK]},
        )
        risk = response.json()["results"][0]["risk"]
        assert "failure_probability" in risk
        assert "predicted_failure_30d" in risk
        assert 0.0 <= risk["failure_probability"] <= 1.0
        assert isinstance(risk["predicted_failure_30d"], bool)

    def test_evaluate_priority_fields(self):
        """Priority output has correct fields and valid category."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": [SAMPLE_TASK]},
        )
        priority = response.json()["results"][0]["priority"]
        assert "priority_score" in priority
        assert "priority_category" in priority
        assert "score_breakdown" in priority
        assert 0.0 <= priority["priority_score"] <= 1.0
        assert priority["priority_category"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        # Score breakdown has 5 factors + total
        breakdown = priority["score_breakdown"]
        for factor in [
            "failure_risk_factor",
            "criticality_factor",
            "urgency_factor",
            "overdue_factor",
            "operational_factor",
            "total_score",
        ]:
            assert factor in breakdown

    def test_evaluate_scheduling_top3_cap(self):
        """Scheduling total_recommendations never exceeds 3."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": SAMPLE_TASKS_MULTI},
        )
        for result in response.json()["results"]:
            sched = result["scheduling"]
            assert sched["total_recommendations"] <= 3
            assert len(sched["recommended_windows"]) <= 3

    def test_evaluate_hitl_invariants(self):
        """Human-in-the-loop invariants are preserved in all windows."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": [SAMPLE_TASK]},
        )
        for result in response.json()["results"]:
            for window in result["scheduling"]["recommended_windows"]:
                assert window["selection_status"] == "RECOMMENDED"
                assert window["human_confirmation_required"] is True
                assert window["is_booked"] is False
                assert window["booking_status"] == "UNBOOKED"


# ---------------------------------------------------------------------------
# 8-9. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_evaluate_invalid_request_422(self):
        """Missing required fields returns 422."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": [{"wrong_field": "value"}]},
        )
        assert response.status_code == 422

    def test_evaluate_empty_tasks_422(self):
        """Empty tasks list returns 422 (min_length=1 constraint)."""
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": []},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 10-11. Recommend-windows endpoint
# ---------------------------------------------------------------------------

class TestRecommendWindowsEndpoint:
    def test_recommend_windows_valid(self):
        """POST /api/v1/recommend-windows returns 200 with recommendations."""
        response = client.post(
            "/api/v1/recommend-windows",
            json={"tasks": SAMPLE_TASKS_MULTI},
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "recommendations_df" in data
        assert "infeasible_tasks" in data
        assert "metrics" in data

    def test_recommend_windows_top3_cap(self):
        """Each task gets at most 3 recommendations."""
        response = client.post(
            "/api/v1/recommend-windows",
            json={"tasks": SAMPLE_TASKS_MULTI, "max_recommendations": 3},
        )
        assert response.status_code == 200
        data = response.json()
        for task_id, recs in data.get("recommendations", {}).items():
            assert len(recs) <= 3

    def test_recommend_windows_infeasible_task(self):
        """Infeasible task (exceeding capacity) returns 0 recommendations."""
        infeasible_task = {
            **SAMPLE_TASK,
            "task_id": "TSK-INFEASIBLE",
            "estimated_duration": 600,
            "required_manpower": 99,
        }
        response = client.post(
            "/api/v1/recommend-windows",
            json={
                "tasks": [infeasible_task],
                "block_windows": [
                    {"block_id": "B001", "start_slot": 1, "end_slot": 3}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        recs = data["recommendations"].get("TSK-INFEASIBLE", [])
        assert len(recs) == 0
        assert len(data.get("infeasible_tasks", [])) > 0

    def test_recommend_windows_hitl_invariants(self):
        """All recommended windows strictly preserve HITL invariants."""
        response = client.post(
            "/api/v1/recommend-windows",
            json={"tasks": [SAMPLE_TASK]},
        )
        assert response.status_code == 200
        data = response.json()
        for recs in data.get("recommendations", {}).values():
            for window in recs:
                assert window["selection_status"] == "RECOMMENDED"
                assert window["human_confirmation_required"] is True
                assert window["is_booked"] is False
                assert window["booking_status"] == "UNBOOKED"


# ---------------------------------------------------------------------------
# 12. JSON Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_response_matches_json_schema(self):
        """Evaluate response validates against schemas/ml_response.schema.json."""
        import jsonschema

        # Load the canonical schema
        schema_path = Path("schemas/ml_response.schema.json")
        if not schema_path.exists():
            # Try from repo root
            schema_path = (
                Path(__file__).resolve().parents[2] / "schemas" / "ml_response.schema.json"
            )
        assert schema_path.exists(), f"Schema not found at {schema_path}"

        with open(schema_path) as f:
            schema = json.load(f)

        # Get a real evaluation response
        response = client.post(
            "/api/v1/evaluate",
            json={"tasks": SAMPLE_TASKS_MULTI},
        )
        assert response.status_code == 200
        data = response.json()

        # Validate against the canonical schema
        jsonschema.validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# 13. Controlled Errors & Information Sanitization
# ---------------------------------------------------------------------------

class TestControlledErrors:
    def test_model_unavailable_503(self):
        """Engine with no model returns 503 on evaluate."""
        from src.services.ml_engine import RailwayMLEngine

        broken_engine = RailwayMLEngine.__new__(RailwayMLEngine)
        broken_engine.version = "0.1.0"
        broken_engine.health = lambda: {
            "status": "ok",
            "artifacts": {"calibrated_xgboost_available": False},
        }

        app.dependency_overrides[get_ml_engine] = lambda: broken_engine
        try:
            response = client.post(
                "/api/v1/evaluate",
                json={"tasks": [SAMPLE_TASK]},
            )
            assert response.status_code == 503
            data = response.json()
            assert data["error"] == "MODEL_UNAVAILABLE"
        finally:
            app.dependency_overrides.clear()

    def test_internal_error_500_suppresses_traceback(self):
        """Engine internal error returns 500 without leaking stack traces or paths."""
        from src.services.ml_engine import RailwayMLEngine

        broken_engine = RailwayMLEngine.__new__(RailwayMLEngine)
        broken_engine.version = "0.1.0"
        broken_engine.health = lambda: {
            "status": "ok",
            "artifacts": {"calibrated_xgboost_available": True},
        }
        def raise_boom(*args, **kwargs):
            raise RuntimeError("Sensitive internal engine detail in /internal/model/private.py")

        broken_engine.evaluate_for_backend = raise_boom

        app.dependency_overrides[get_ml_engine] = lambda: broken_engine
        try:
            response = client.post(
                "/api/v1/evaluate",
                json={"tasks": [SAMPLE_TASK]},
            )
            assert response.status_code == 500
            data = response.json()
            assert data["error"] == "INTERNAL_ERROR"
            assert "RuntimeError" in data["detail"]
            assert "Traceback" not in response.text
            assert "/internal/model" not in response.text
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 14. Configuration Safety
# ---------------------------------------------------------------------------

class TestConfigurationSafety:
    def test_cors_disabled_by_default(self):
        """Default server configuration does not send Access-Control-Allow-Origin."""
        response = client.get("/health", headers={"Origin": "http://evil.com"})
        assert "access-control-allow-origin" not in response.headers
