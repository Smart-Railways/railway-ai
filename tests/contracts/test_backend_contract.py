"""
Backend Integration Contract Test Suite — Railway-AI

Verifies the standardized output contract for backend integration:
Item A: Top-level response structure (status, engine_version, tasks_evaluated, results, metrics)
Item B: Task-level schema (task_id, section_id, department, decision_rank, risk, priority, explanation, scheduling)
Item C: 4 Frozen Output Categories present on every task
Item D: Pure native Python types (no numpy, no pandas scalars)
Item E: Range constraints on probabilities, scores, and factors
Item F: Exact 5-factor breakdown in score_breakdown
Item G: Decision category threshold mapping (LOW, MEDIUM, HIGH, CRITICAL)
Item H: Explainable reason codes and fallback logic
Item I: Top-3 window recommendation cap (at most 3 recommendations)
Item J: Sequential rank values (1, 2, 3)
Item K: Human-in-the-loop semantics (RECOMMENDED, UNBOOKED, confirmation required)
Item L: Elimination of fake / heuristic confidence fields from contract payload
Item M: ID and metadata preservation across pipeline
Item N: Native JSON serializability (json.dumps / json.loads round-trip)
Item O: Robust edge-case handling (empty input, infeasible tasks, single-dict input)
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.services.ml_engine import RailwayMLEngine


@pytest.fixture
def ml_engine():
    return RailwayMLEngine()


@pytest.fixture
def sample_standard_tasks():
    return [
        {
            "task_id": "TASK-001",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "estimated_duration": 60,
            "required_manpower": 4,
            "criticality": 8.0,
            "urgency_score": 0.85,
            "overdue_days": 15.0,
            "failure_probability": 0.045,
            "predicted_delay_minutes": 1.5,
        },
        {
            "task_id": "TASK-002",
            "section_id": "NDL-MTJ-01",
            "department": "S&T",
            "estimated_duration": 60,
            "required_manpower": 3,
            "criticality": 4.0,
            "urgency_score": 0.25,
            "overdue_days": 0.0,
            "failure_probability": 0.002,
            "predicted_delay_minutes": 0.0,
        },
        {
            "task_id": "TASK-003",
            "section_id": "MTJ-AGC-01",
            "department": "TRACTION",
            "estimated_duration": 90,
            "required_manpower": 2,
            "criticality": 9.5,
            "urgency_score": 0.95,
            "overdue_days": 30.0,
            "failure_probability": 0.080,
            "predicted_delay_minutes": 5.0,
        },
    ]


@pytest.fixture
def sample_raw_asset_tasks():
    return [
        {
            "task_id": "ASSET-RAW-001",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "estimated_duration": 60,
            "required_manpower": 4,
            "criticality": 8.0,
            "urgency_score": 0.85,
            "overdue_days": 15.0,
            "asset_age_years": 14.0,
            "condition_score": 55.0,
            "usage_factor": 1.4,
            "historical_failure_count": 3,
            "historical_downtime_hours": 18.0,
            "days_since_last_failure": 40.0,
        }
    ]


def _assert_pure_python_types(data, path="root"):
    """Recursively asserts that no numpy or pandas types exist in the tree."""
    if isinstance(data, dict):
        for k, v in data.items():
            assert not isinstance(k, (np.generic, pd.Series)), f"Numpy key at {path}.{k}"
            _assert_pure_python_types(v, f"{path}.{k}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _assert_pure_python_types(item, f"{path}[{idx}]")
    else:
        assert not isinstance(data, np.generic), (
            f"Found numpy type {type(data)} at {path}: {data}"
        )
        assert not isinstance(data, (pd.Series, pd.DataFrame)), (
            f"Found pandas object at {path}"
        )
        assert isinstance(data, (int, float, str, bool, type(None))), (
            f"Unexpected non-primitive type {type(data)} at {path}"
        )


class TestBackendContractValidation:
    """Verifies Items A through O of the ML Backend Integration Contract."""

    def test_item_a_top_level_structure(self, ml_engine, sample_standard_tasks):
        """Item A: Verify top-level envelope structure and metrics."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        assert isinstance(res, dict)
        expected_keys = {"status", "engine_version", "tasks_evaluated", "results", "metrics"}
        assert expected_keys.issubset(res.keys())
        assert res["status"] == "SUCCESS"
        assert res["tasks_evaluated"] == len(sample_standard_tasks)
        assert res["engine_version"] == ml_engine.version

        metrics = res["metrics"]
        assert isinstance(metrics, dict)
        for key in [
            "tasks_count",
            "critical_tasks_count",
            "high_tasks_count",
            "medium_tasks_count",
            "low_tasks_count",
            "tasks_with_recommendations",
            "infeasible_tasks",
        ]:
            assert key in metrics, f"Metric '{key}' missing from metrics dict"
        assert metrics["tasks_count"] == len(sample_standard_tasks)

    def test_item_b_task_level_structure(self, ml_engine, sample_standard_tasks):
        """Item B: Verify task-level schema and mandatory identifier fields."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        for task in res["results"]:
            for key in [
                "task_id",
                "section_id",
                "department",
                "decision_rank",
                "risk",
                "priority",
                "explanation",
                "scheduling",
            ]:
                assert key in task, f"Task-level key '{key}' missing"
            assert isinstance(task["task_id"], str)
            assert isinstance(task["section_id"], str)
            assert isinstance(task["department"], str)
            assert isinstance(task["decision_rank"], int)

    def test_item_c_four_frozen_categories(self, ml_engine, sample_standard_tasks):
        """Item C: Verify all 4 frozen output categories are present with their contracted child keys."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        for task in res["results"]:
            # Category 1: RISK
            risk = task["risk"]
            assert "failure_probability" in risk
            assert "predicted_failure_30d" in risk

            # Category 2: PRIORITY
            priority = task["priority"]
            assert "priority_score" in priority
            assert "priority_category" in priority
            assert "score_breakdown" in priority

            # Category 3: EXPLANATION
            explanation = task["explanation"]
            assert "reason_tags" in explanation
            assert "score_breakdown" in explanation
            assert "explanation_summary" in explanation

            # Category 4: SCHEDULING
            scheduling = task["scheduling"]
            assert "recommended_windows" in scheduling
            assert "total_recommendations" in scheduling
            assert "has_feasible_window" in scheduling

    def test_item_d_pure_native_python_types(self, ml_engine, sample_standard_tasks):
        """Item D: Verify absolutely no numpy/pandas types leak into the response payload."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)
        _assert_pure_python_types(res)

    def test_item_e_range_constraints(self, ml_engine, sample_standard_tasks):
        """Item E: Verify range bounds on probabilities, scores, and factor values."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        for task in res["results"]:
            # Risk bounds
            fp = task["risk"]["failure_probability"]
            assert 0.0 <= fp <= 1.0, f"Failure probability {fp} out of [0, 1]"
            assert isinstance(task["risk"]["predicted_failure_30d"], bool)

            # Priority bounds
            ps = task["priority"]["priority_score"]
            assert 0.0 <= ps <= 1.0, f"Priority score {ps} out of [0, 1]"
            assert task["priority"]["priority_category"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

            # Breakdown factor bounds
            bd = task["priority"]["score_breakdown"]
            for factor_key in [
                "failure_risk_factor",
                "criticality_factor",
                "urgency_factor",
                "overdue_factor",
                "operational_factor",
            ]:
                val = bd[factor_key]
                assert 0.0 <= val <= 1.0, f"Factor {factor_key}={val} out of [0, 1]"

    def test_item_f_five_factor_breakdown(self, ml_engine, sample_standard_tasks):
        """Item F: Verify exactly 5 decision factors and mathematical consistency with priority_score."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        weights = ml_engine.decision_engine.WEIGHTS
        assert len(weights) == 5

        for task in res["results"]:
            bd = task["priority"]["score_breakdown"]
            expected_factors = {
                "failure_risk_factor",
                "criticality_factor",
                "urgency_factor",
                "overdue_factor",
                "operational_factor",
                "total_score",
            }
            assert set(bd.keys()) == expected_factors

            # Verify linear combination
            recomputed = (
                weights["failure_risk_factor"] * bd["failure_risk_factor"]
                + weights["criticality_factor"] * bd["criticality_factor"]
                + weights["urgency_factor"] * bd["urgency_factor"]
                + weights["overdue_factor"] * bd["overdue_factor"]
                + weights["operational_factor"] * bd["operational_factor"]
            )
            recomputed = min(1.0, max(0.0, recomputed))
            assert pytest.approx(bd["total_score"], abs=1e-3) == task["priority"]["priority_score"]
            assert pytest.approx(recomputed, abs=1e-3) == bd["total_score"]

    def test_item_g_decision_category_thresholds(self, ml_engine):
        """Item G: Verify decision categories strictly follow the defined intervals."""
        test_scores = [
            (0.10, "LOW"),
            (0.25, "LOW"),
            (0.26, "MEDIUM"),
            (0.50, "MEDIUM"),
            (0.51, "HIGH"),
            (0.75, "HIGH"),
            (0.76, "CRITICAL"),
            (0.95, "CRITICAL"),
        ]

        tasks = []
        for idx, (score, _) in enumerate(test_scores):
            tasks.append({
                "task_id": f"CAT-TASK-{idx}",
                "section_id": "NDL-MTJ-01",
                "department": "ENGINEERING",
                "estimated_duration": 60,
                "required_manpower": 2,
                "criticality": score * 10,
                "urgency_score": score,
                "overdue_days": score * 30,
                "failure_probability": score,
            })

        res = ml_engine.evaluate_for_backend(tasks, apply_real_pressure=False)
        for task in res["results"]:
            score = task["priority"]["priority_score"]
            cat = task["priority"]["priority_category"]
            if score <= 0.25:
                assert cat == "LOW"
            elif score <= 0.50:
                assert cat == "MEDIUM"
            elif score <= 0.75:
                assert cat == "HIGH"
            else:
                assert cat == "CRITICAL"

    def test_item_h_reason_tags_and_fallback(self, ml_engine):
        """Item H: Verify transparent reason codes and fallback to ROUTINE_INSPECTION."""
        # Routine task with all factors near 0
        routine_task = [{
            "task_id": "ROUTINE-001",
            "section_id": "UNKNOWN_SEC",
            "department": "ENGINEERING",
            "estimated_duration": 60,
            "required_manpower": 2,
            "criticality": 1.0,
            "urgency_score": 0.05,
            "overdue_days": 0.0,
            "failure_probability": 0.001,
        }]
        res = ml_engine.evaluate_for_backend(routine_task, apply_real_pressure=False)
        tags = res["results"][0]["explanation"]["reason_tags"]
        assert "ROUTINE_INSPECTION" in tags

        # Urgent high risk task
        urgent_task = [{
            "task_id": "URGENT-001",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "estimated_duration": 60,
            "required_manpower": 4,
            "criticality": 9.0,
            "urgency_score": 0.90,
            "overdue_days": 20.0,
            "failure_probability": 0.60,
        }]
        res_u = ml_engine.evaluate_for_backend(urgent_task, apply_real_pressure=False)
        tags_u = res_u["results"][0]["explanation"]["reason_tags"]
        assert "HIGH_FAILURE_RISK" in tags_u
        assert "HIGH_URGENCY" in tags_u
        assert "HIGH_CRITICALITY" in tags_u
        assert "OVERDUE_MAINTENANCE" in tags_u

    def test_item_i_top_3_recommendation_cap(self, ml_engine, sample_standard_tasks):
        """Item I: Verify at most 3 block window recommendations are produced per task."""
        # Provide 5 available windows within the 12-slot (6-hour) horizon
        five_windows = [
            {"block_id": "B01", "start_slot": 1, "end_slot": 3},
            {"block_id": "B02", "start_slot": 3, "end_slot": 5},
            {"block_id": "B03", "start_slot": 5, "end_slot": 7},
            {"block_id": "B04", "start_slot": 7, "end_slot": 9},
            {"block_id": "B05", "start_slot": 9, "end_slot": 11},
        ]
        res = ml_engine.evaluate_for_backend(sample_standard_tasks, block_windows=five_windows)

        for task in res["results"]:
            sched = task["scheduling"]
            assert sched["total_recommendations"] <= 3
            assert len(sched["recommended_windows"]) <= 3
            assert sched["total_recommendations"] == len(sched["recommended_windows"])

    def test_item_j_sequential_ranks(self, ml_engine, sample_standard_tasks):
        """Item J: Verify recommendation ranks are strictly sequential starting at 1."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        for task in res["results"]:
            windows = task["scheduling"]["recommended_windows"]
            ranks = [w["rank"] for w in windows]
            expected_ranks = list(range(1, len(windows) + 1))
            assert ranks == expected_ranks

    def test_item_k_human_in_the_loop_invariants(self, ml_engine, sample_standard_tasks):
        """Item K: Verify CP-SAT produces strictly unbooked recommendations requiring operator confirmation."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        for task in res["results"]:
            for win in task["scheduling"]["recommended_windows"]:
                assert win["selection_status"] == "RECOMMENDED"
                assert win["human_confirmation_required"] is True
                assert win["is_booked"] is False
                assert win["booking_status"] == "UNBOOKED"

    def test_item_l_zero_fake_confidence_metrics(self, ml_engine, sample_standard_tasks, sample_raw_asset_tasks):
        """Item L: Verify heuristic confidence metrics are excluded from the public backend contract payload."""
        for dataset in [sample_standard_tasks, sample_raw_asset_tasks]:
            res = ml_engine.evaluate_for_backend(dataset)
            dumped_str = json.dumps(res)

            # Assert forbidden heuristic keys are absent from JSON
            assert '"confidence"' not in dumped_str
            assert '"failure_model_confidence"' not in dumped_str

            for task in res["results"]:
                assert "confidence" not in task
                assert "failure_model_confidence" not in task
                assert "confidence" not in task["risk"]
                assert "failure_model_confidence" not in task["risk"]
                assert "confidence" not in task["priority"]
                assert "confidence" not in task["explanation"]
                assert "confidence" not in task["scheduling"]

    def test_item_m_id_and_metadata_preservation(self, ml_engine, sample_standard_tasks):
        """Item M: Verify task IDs, sections, and departments are accurately preserved."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        res_tasks = {t["task_id"]: t for t in res["results"]}
        for orig in sample_standard_tasks:
            tid = orig["task_id"]
            assert tid in res_tasks
            assert res_tasks[tid]["section_id"] == orig["section_id"]
            assert res_tasks[tid]["department"] == orig["department"]

    def test_item_n_json_serialization_roundtrip(self, ml_engine, sample_standard_tasks):
        """Item N: Verify entire response payload serializes and deserializes losslessly via json."""
        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        # Must not throw TypeError or ValueError
        json_str = json.dumps(res, indent=2)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        reloaded = json.loads(json_str)
        assert reloaded == res

    def test_item_o_edge_cases(self, ml_engine):
        """Item O: Verify edge cases (empty input, infeasible tasks, single dict input, raw asset features)."""
        # 1. Empty list
        empty_res = ml_engine.evaluate_for_backend([])
        assert empty_res["status"] == "EMPTY_INPUT"
        assert empty_res["tasks_evaluated"] == 0
        assert empty_res["results"] == []
        assert empty_res["metrics"]["tasks_count"] == 0

        # 2. Empty DataFrame
        empty_df_res = ml_engine.evaluate_for_backend(pd.DataFrame())
        assert empty_df_res["status"] == "EMPTY_INPUT"
        assert empty_df_res["tasks_evaluated"] == 0

        # 3. Single dict input
        single_task = {
            "task_id": "SINGLE-001",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "estimated_duration": 60,
            "required_manpower": 4,
            "criticality": 7.0,
            "urgency_score": 0.5,
            "overdue_days": 10.0,
            "failure_probability": 0.02,
        }
        res_single = ml_engine.evaluate_for_backend(single_task)
        assert res_single["status"] == "SUCCESS"
        assert res_single["tasks_evaluated"] == 1
        assert res_single["results"][0]["task_id"] == "SINGLE-001"

        # 4. Infeasible task (duration too long for any window)
        infeasible_task = {
            "task_id": "TASK-TOO-LONG",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "estimated_duration": 400,  # exceeds 120-min window
            "required_manpower": 4,
            "criticality": 8.0,
            "urgency_score": 0.8,
            "overdue_days": 10.0,
            "failure_probability": 0.03,
        }
        tight_windows = [{"block_id": "B_TIGHT", "start_slot": 1, "end_slot": 4}]
        res_inf = ml_engine.evaluate_for_backend([infeasible_task], block_windows=tight_windows)
        assert res_inf["status"] == "SUCCESS"
        inf_task_res = res_inf["results"][0]
        assert inf_task_res["scheduling"]["total_recommendations"] == 0
        assert inf_task_res["scheduling"]["has_feasible_window"] is False
        assert inf_task_res["scheduling"]["recommended_windows"] == []
        assert "TASK-TOO-LONG" in res_inf["metrics"]["infeasible_tasks"]

        # 5. Raw asset features triggering XGBoost pipeline
        raw_task = {
            "task_id": "ASSET-XGBOOST",
            "section_id": "NDL-MTJ-01",
            "department": "ENGINEERING",
            "asset_age_years": 12.0,
            "condition_score": 60.0,
            "criticality": 8.0,
            "usage_factor": 1.2,
            "historical_failure_count": 2,
            "historical_downtime_hours": 10.0,
            "days_since_last_failure": 60.0,
        }
        res_raw = ml_engine.evaluate_for_backend(raw_task)
        assert res_raw["status"] == "SUCCESS"
        assert res_raw["results"][0]["risk"]["failure_probability"] >= 0.0
        assert isinstance(res_raw["results"][0]["risk"]["predicted_failure_30d"], bool)
        _assert_pure_python_types(res_raw)

    def test_item_p_schema_file_validation(self, ml_engine, sample_standard_tasks):
        """Item P: Verify response strictly adheres to docs/schemas/ml_response.schema.json."""
        import os
        from pathlib import Path
        schema_path = Path(__file__).resolve().parents[2] / "schemas/ml_response.schema.json"
        if not schema_path.exists():
            schema_path = Path(__file__).resolve().parents[2] / "docs/schemas/ml_response.schema.json"
        assert schema_path.exists(), f"Schema file not found at {schema_path}"

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        res = ml_engine.evaluate_for_backend(sample_standard_tasks)

        try:
            import jsonschema
            jsonschema.validate(instance=res, schema=schema)
        except ImportError:
            # Fallback manual validation if jsonschema is not installed in runner
            assert set(res.keys()) == set(schema["required"])
            assert set(res["results"][0].keys()) == set(schema["properties"]["results"]["items"]["required"])
