"""
Tests for Candidate V3 Failure Model Artifact & Predictor Compatibility.
"""

import os
import json
import joblib
import pytest
import pandas as pd
import numpy as np

from src.models.failure_predictor import FailureRiskPredictor


class TestCandidateV3Model:
    """Validation test suite for Candidate V3 failure risk predictor."""

    @pytest.fixture
    def candidate_path(self):
        p = "models/candidate/candidate_v3/calibrated_xgboost_v3.pkl"
        if not os.path.exists(p) and os.path.exists("models/candidate_v3/calibrated_xgboost_v3.pkl"):
            return "models/candidate_v3/calibrated_xgboost_v3.pkl"
        return p

    @pytest.fixture
    def candidate_predictor(self, candidate_path):
        return FailureRiskPredictor(model_path=candidate_path)

    @pytest.fixture
    def production_predictor(self):
        return FailureRiskPredictor(model_path="models/production/calibrated_xgboost.pkl")

    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame([
            {
                "asset_id": "AST-TEST-01",
                "asset_age_years": 12.0,
                "condition_score": 45.0,
                "criticality": 8,
                "usage_factor": 1.40,
                "historical_failure_count": 4,
                "historical_downtime_hours": 18.5,
                "days_since_last_failure": 30.0
            },
            {
                "asset_id": "AST-TEST-02",
                "asset_age_years": 2.0,
                "condition_score": 95.0,
                "criticality": 4,
                "usage_factor": 0.60,
                "historical_failure_count": 0,
                "historical_downtime_hours": 0.0,
                "days_since_last_failure": 600.0
            }
        ])

    def test_candidate_artifact_exists_and_loads(self, candidate_path, candidate_predictor):
        """Candidate V3 artifact exists and loads as a valid CalibratedClassifierCV."""
        assert os.path.exists(candidate_path)
        assert candidate_predictor.is_available is True
        assert candidate_predictor.model is not None

    def test_production_model_unmodified(self, production_predictor):
        """Production model remains available and functional."""
        assert production_predictor.is_available is True
        assert production_predictor.model is not None

    def test_candidate_feature_contract(self, candidate_predictor):
        """Candidate model uses exactly the 7 production features."""
        expected = [
            "asset_age_years", "condition_score", "criticality", "usage_factor",
            "historical_failure_count", "historical_downtime_hours", "days_since_last_failure"
        ]
        assert candidate_predictor.FEATURE_COLUMNS == expected

    def test_candidate_predict_risk_bounds_and_risk_gradient(self, candidate_predictor, sample_features):
        """Probabilities are strictly bounded [0, 1] and degraded asset ranks higher."""
        res = candidate_predictor.predict_risk(sample_features, threshold=0.0200)

        assert "failure_probability" in res.columns
        assert "predicted_failure_30d" in res.columns
        assert (res["failure_probability"] >= 0.0).all()
        assert (res["failure_probability"] <= 1.0).all()

        # AST-TEST-01 (degraded, older, high usage) must have higher risk than AST-TEST-02
        assert res.loc[0, "failure_probability"] > res.loc[1, "failure_probability"]

    def test_candidate_fallback_handling(self, candidate_predictor):
        """Missing features activate fallback gracefully."""
        df_incomplete = pd.DataFrame([{"asset_id": "AST-TEST-03", "condition_score": 35.0}])
        res = candidate_predictor.predict_risk(df_incomplete)
        assert res["model_source"].iloc[0] == "heuristic_fallback"
        assert "failure_probability" in res.columns

    def test_candidate_manifest_consistency(self):
        """Candidate V3 manifest exists and records required metadata."""
        manifest_path = "models/candidate/candidate_v3/candidate_v3_manifest.json"
        if not os.path.exists(manifest_path) and os.path.exists("models/candidate_v3/candidate_v3_manifest.json"):
            manifest_path = "models/candidate_v3/candidate_v3_manifest.json"
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["model_name"] == "candidate_v3_calibrated_xgboost"
        assert manifest["promoted_to_production"] is False
        assert manifest["production_promotion_status"] == "READY FOR PROMOTION REVIEW"
        assert manifest["metrics"]["test_untouched"]["pr_auc"] > 0.03
        assert manifest["metrics"]["test_untouched"]["roc_auc"] > 0.69
