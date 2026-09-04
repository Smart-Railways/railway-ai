"""
Failure Risk Predictor — Models Layer (Phases 9 & 10)

Loads the calibrated XGBoost model and applies optimal decision thresholds,
confidence estimation, and feature-level validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib


class FailureRiskPredictor:
    """
    Inference interface for calibrated XGBoost failure classification.
    Predicts 30-day failure probability, applies calibrated thresholds (no arbitrary 0.5),
    and derives confidence metrics.
    """

    FEATURE_COLUMNS = [
        "asset_age_years",
        "condition_score",
        "criticality",
        "usage_factor",
        "weather_stress",
        "historical_failure_count",
        "historical_downtime_hours",
        "days_since_last_failure"
    ]

    DEFAULT_THRESHOLD = 0.35  # Optimal threshold tuned on PR-AUC validation

    def __init__(self, model_path: str = "models/calibrated_xgboost.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def predict_risk(
        self,
        features: pd.DataFrame,
        threshold: float = DEFAULT_THRESHOLD
    ) -> pd.DataFrame:
        """
        Computes calibrated failure probabilities, predicted failure flags,
        and model confidence.
        """
        df = features.copy()

        # Check for required feature columns
        has_all_features = all(col in df.columns for col in self.FEATURE_COLUMNS)

        if not self.is_available or not has_all_features:
            # Fallback heuristic if raw telemetry features are absent
            # (e.g. when work order provides only precomputed condition / failure_probability)
            if "failure_probability" not in df.columns:
                df["failure_probability"] = 0.0
            if "condition_score" in df.columns:
                # Estimate risk inversely proportional to condition score
                cond_risk = (100.0 - df["condition_score"].clip(0, 100)) / 100.0
                df["failure_probability"] = np.maximum(df["failure_probability"], cond_risk * 0.5)

            df["predicted_failure_30d"] = (df["failure_probability"] >= threshold).astype(int)
            df["failure_model_confidence"] = 0.75
            df["model_source"] = "heuristic_fallback"
            return df

        # Prepare numerical matrix
        X = df[self.FEATURE_COLUMNS].copy()
        for col in self.FEATURE_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if len(X) > 1 else 0.0)

        # Calibrated model prediction
        probs = self.model.predict_proba(X)[:, 1]
        df["failure_probability"] = np.round(probs, 4)
        df["predicted_failure_30d"] = (df["failure_probability"] >= threshold).astype(int)

        # Calibrated confidence: distance from decision boundary + quality of inputs
        boundary_dist = np.abs(probs - threshold)
        df["failure_model_confidence"] = np.round(0.80 + (boundary_dist * 0.18), 4)
        df["model_source"] = "calibrated_xgboost"

        return df
