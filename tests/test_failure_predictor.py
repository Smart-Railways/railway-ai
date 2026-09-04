"""
Tests for FailureRiskPredictor — Models Layer (Phases 9 and 10)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.failure_predictor import FailureRiskPredictor
from src.services.ml_engine import RailwayMLEngine


class TestFailureRiskPredictor:
    """Test suite for calibrated XGBoost failure prediction."""

    @pytest.fixture
    def predictor(self):
        return FailureRiskPredictor(model_path="models/calibrated_xgboost.pkl")

    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame([
            {
                "asset_id": "AST-001",
                "asset_age_years": 8.5,
                "condition_score": 42.0,
                "criticality": 4,
                "usage_factor": 1.35,
                "weather_stress": 0.8,
                "historical_failure_count": 3,
                "historical_downtime_hours": 14.5,
                "days_since_last_failure": 28.0
            },
            {
                "asset_id": "AST-002",
                "asset_age_years": 2.1,
                "condition_score": 92.0,
                "criticality": 2,
                "usage_factor": 0.65,
                "weather_stress": 0.2,
                "historical_failure_count": 0,
                "historical_downtime_hours": 0.0,
                "days_since_last_failure": 450.0
            }
        ])

    def test_model_loading(self, predictor):
        """Model file exists and is loaded successfully."""
        assert predictor.is_available is True
        assert predictor.model is not None

    def test_predict_risk_with_features(self, predictor, sample_features):
        """Telemetry features produce valid calibrated probabilities and flags."""
        result = predictor.predict_risk(sample_features)

        assert "failure_probability" in result.columns
        assert "predicted_failure_30d" in result.columns
        assert "failure_model_confidence" in result.columns
        assert "model_source" in result.columns

        # Verify bounds
        assert (result["failure_probability"] >= 0.0).all()
        assert (result["failure_probability"] <= 1.0).all()
        assert (result["failure_model_confidence"] >= 0.5).all()
        assert (result["failure_model_confidence"] <= 1.0).all()

        # AST-001 has worse condition and higher stress than AST-002
        assert result.loc[0, "failure_probability"] > result.loc[1, "failure_probability"]
        assert result.loc[0, "model_source"] == "calibrated_xgboost"

    def test_threshold_tuning(self, predictor, sample_features):
        """Custom threshold affects classification boundary."""
        res_low = predictor.predict_risk(sample_features, threshold=0.10)
        res_high = predictor.predict_risk(sample_features, threshold=0.90)

        # Lower threshold should classify equal or more failures than higher threshold
        assert res_low["predicted_failure_30d"].sum() >= res_high["predicted_failure_30d"].sum()

    def test_fallback_when_features_missing(self, predictor):
        """Fallback activates gracefully when required telemetry columns are missing."""
        df_incomplete = pd.DataFrame([
            {"asset_id": "AST-003", "condition_score": 30.0},
            {"asset_id": "AST-004", "condition_score": 85.0}
        ])

        result = predictor.predict_risk(df_incomplete)

        assert "failure_probability" in result.columns
        assert "predicted_failure_30d" in result.columns
        assert result["model_source"].iloc[0] == "heuristic_fallback"
        # Poor condition score yields higher failure probability
        assert result.loc[0, "failure_probability"] > result.loc[1, "failure_probability"]

    def test_fallback_when_model_file_missing(self):
        """Predictor works safely with heuristic fallback if model file does not exist."""
        pred_no_model = FailureRiskPredictor(model_path="models/non_existent.pkl")
        assert pred_no_model.is_available is False

        df = pd.DataFrame([{"condition_score": 50.0}])
        result = pred_no_model.predict_risk(df)
        assert result["model_source"].iloc[0] == "heuristic_fallback"
        assert result["failure_probability"].iloc[0] == 0.25

    def test_engine_integration(self, sample_features):
        """RailwayMLEngine uses FailureRiskPredictor during predict()."""
        engine = RailwayMLEngine()
        preds = engine.predict(sample_features)

        assert "failure_probability" in preds.columns
        assert "maintenance_decision_score" in preds.columns
        assert "decision_category" in preds.columns
        assert "failure_model_confidence" in preds.columns
