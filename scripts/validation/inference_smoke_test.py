import pandas as pd
import numpy as np
import json
from src.models.failure_predictor import FailureRiskPredictor
from src.decision.maintenance_decision_engine import MaintenanceDecisionEngine
from src.services.ml_engine import RailwayMLEngine

print("=== STEP 6: PRODUCTION INFERENCE SMOKE TEST ===")

# Instantiate default production predictor (loads models/production/calibrated_xgboost.pkl)
predictor = FailureRiskPredictor()
assert predictor.is_available is True, "Default predictor failed to load models/production/calibrated_xgboost.pkl!"
assert predictor.DEFAULT_THRESHOLD == 0.0120, "Threshold must be 0.0120!"

decision_engine = MaintenanceDecisionEngine()
ml_engine = RailwayMLEngine()

# 5 Representative Asset Test Cases
test_assets = pd.DataFrame([
    {
        "asset_id": "AST-LOW-01",
        "task_id": "TSK-001",
        "section_id": "SEC-A",
        "department": "ENGINEERING",
        "asset_age_years": 1.5,
        "condition_score": 98.0,
        "criticality": 3,
        "usage_factor": 0.60,
        "historical_failure_count": 0,
        "historical_downtime_hours": 0.0,
        "days_since_last_failure": 0.0,
        "urgency_score": 0.10,
        "overdue_days": 0.0,
        "railkit_operational_pressure": 0.10
    },
    {
        "asset_id": "AST-MED-02",
        "task_id": "TSK-002",
        "section_id": "SEC-B",
        "department": "S&T",
        "asset_age_years": 8.0,
        "condition_score": 75.0,
        "criticality": 6,
        "usage_factor": 1.05,
        "historical_failure_count": 1,
        "historical_downtime_hours": 4.5,
        "days_since_last_failure": 210.0,
        "urgency_score": 0.40,
        "overdue_days": 5.0,
        "railkit_operational_pressure": 0.40
    },
    {
        "asset_id": "AST-HIGH-03",
        "task_id": "TSK-003",
        "section_id": "SEC-C",
        "department": "TRACTION",
        "asset_age_years": 18.0,
        "condition_score": 42.0,
        "criticality": 8,
        "usage_factor": 1.45,
        "historical_failure_count": 4,
        "historical_downtime_hours": 18.5,
        "days_since_last_failure": 35.0,
        "urgency_score": 0.85,
        "overdue_days": 20.0,
        "railkit_operational_pressure": 0.75
    },
    {
        "asset_id": "AST-CRIT-04",
        "task_id": "TSK-004",
        "section_id": "SEC-D",
        "department": "S&T",
        "asset_age_years": 10.0,
        "condition_score": 60.0,
        "criticality": 10,
        "usage_factor": 1.20,
        "historical_failure_count": 2,
        "historical_downtime_hours": 8.0,
        "days_since_last_failure": 120.0,
        "urgency_score": 0.70,
        "overdue_days": 15.0,
        "railkit_operational_pressure": 0.60
    },
    {
        "asset_id": "AST-HIST-05",
        "task_id": "TSK-005",
        "section_id": "SEC-E",
        "department": "ENGINEERING",
        "asset_age_years": 14.0,
        "condition_score": 52.0,
        "criticality": 7,
        "usage_factor": 1.30,
        "historical_failure_count": 5,
        "historical_downtime_hours": 24.0,
        "days_since_last_failure": 22.0,
        "urgency_score": 0.75,
        "overdue_days": 18.0,
        "railkit_operational_pressure": 0.80
    }
])

# Test 1: Direct Predictor
preds = predictor.predict_risk(test_assets)
print("\n--- Direct Risk Predictor Results ---")
for idx, row in preds.iterrows():
    p = row["failure_probability"]
    flag = row["predicted_failure_30d"]
    print(f"Asset: {row['asset_id']:12} | Prob: {p:.5f} | Flag: {flag} | Age: {row['asset_age_years']} | Cond: {row['condition_score']}")
    assert np.isfinite(p) and 0.0 <= p <= 1.0, f"Invalid prob: {p}"
    assert flag in [True, False, 0, 1]

# Gradient checks
p_low = preds.loc[0, "failure_probability"]
p_med = preds.loc[1, "failure_probability"]
p_high = preds.loc[2, "failure_probability"]
assert p_high > p_med > p_low, f"Risk gradient violation: {p_high} !> {p_med} !> {p_low}"
print("Risk gradient verified: p_high > p_med > p_low")

# Test 2: RailwayMLEngine evaluate_for_backend()
backend_response = ml_engine.evaluate_for_backend(test_assets)
assert backend_response["status"] == "SUCCESS"
assert backend_response["tasks_evaluated"] == 5

print("\n--- Backend Contract Evaluation Results ---")
for res in backend_response["results"]:
    t_id = res["task_id"]
    p_cat = res["priority"]["priority_category"]
    p_score = res["priority"]["priority_score"]
    p_rank = res["decision_rank"]
    f_prob = res["risk"]["failure_probability"]
    reasons = res["explanation"]["reason_tags"]
    summary = res["explanation"]["explanation_summary"]
    recs = res["scheduling"]["recommended_windows"]

    print(f"Task: {t_id:8} | Rank: {p_rank} | Cat: {p_cat:8} | Score: {p_score:.4f} | Prob: {f_prob:.5f} | Tags: {reasons}")
    print(f"  Summary: {summary}")
    print(f"  Recommendations: {len(recs)} window(s)")

    assert np.isfinite(f_prob) and 0.0 <= f_prob <= 1.0
    assert p_cat in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert 0.0 <= p_score <= 1.0
    assert len(reasons) > 0
    assert len(summary) > 0

# Test 3: Priority Ordering
high_task = [r for r in backend_response["results"] if r["task_id"] == "TSK-003"][0]
low_task = [r for r in backend_response["results"] if r["task_id"] == "TSK-001"][0]
assert high_task["decision_rank"] < low_task["decision_rank"], "Priority rank violation between high and low task!"

print("\nSUCCESS: All 5 smoke test cases passed with valid probabilities, priorities, and explanations.")
