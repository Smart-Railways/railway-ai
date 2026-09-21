"""
Reproducible Training Pipeline for Candidate V3 Failure Model.

This script trains, calibrates, and evaluates the Candidate V3 XGBoost model
using the verified 7-feature schema on Candidate V3 processed data.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_candidate_v3.py
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix
)

# Configuration & Hyperparameters
FEATURE_COLUMNS = [
    "asset_age_years",
    "condition_score",
    "criticality",
    "usage_factor",
    "historical_failure_count",
    "historical_downtime_hours",
    "days_since_last_failure"
]

HYPERPARAMETERS = {
    "n_estimators": 242,
    "max_depth": 3,
    "learning_rate": 0.018961625351033223,
    "min_child_weight": 6,
    "subsample": 0.819683927204421,
    "colsample_bytree": 0.8384967934413827,
    "reg_alpha": 0.001142048890339075,
    "reg_lambda": 0.16471572462751574,
    "scale_pos_weight": 2.0198585331558183,
    "random_state": 42,
    "eval_metric": "aucpr"
}

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5
OPERATING_THRESHOLD = 0.0200


def load_and_preprocess(data_dir: str = "data/processed/candidate_v3"):
    train_df = pd.read_csv(os.path.join(data_dir, "failure_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "failure_validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "failure_test.csv"))

    X_train = train_df[FEATURE_COLUMNS].fillna(0.0)
    y_train = train_df["failure_within_30_days"].values

    X_val = val_df[FEATURE_COLUMNS].fillna(0.0)
    y_val = val_df["failure_within_30_days"].values

    X_test = test_df[FEATURE_COLUMNS].fillna(0.0)
    y_test = test_df["failure_within_30_days"].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def evaluate(model, X, y, threshold=OPERATING_THRESHOLD):
    probs = model.predict_proba(X)[:, 1]
    n = len(y)
    pos = int(y.sum())
    base_rate = pos / max(1, n)

    pr_auc = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))
    brier = float(brier_score_loss(y, probs))

    preds = (probs >= threshold).astype(int)
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))

    top_k = {}
    for pct in [0.01, 0.05, 0.10]:
        k = max(1, int(n * pct))
        top_idx = np.argsort(probs)[-k:]
        captured = int(y[top_idx].sum())
        p_k = captured / k
        r_k = captured / max(1, pos)
        lift = p_k / base_rate if base_rate > 0 else 0.0
        top_k[f"top_{int(pct*100)}%"] = {
            "k": k, "captured": captured, "precision": round(p_k, 5),
            "recall": round(r_k, 5), "lift": round(lift, 2)
        }

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "brier": round(brier, 5),
        "threshold": threshold,
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "top_k": top_k
    }


def train_and_save(out_dir: str = "models/candidate_v3"):
    os.makedirs(out_dir, exist_ok=True)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess()

    print(f"Training XGBoost on {len(X_train)} records (positives={y_train.sum()})...")
    base_model = xgb.XGBClassifier(**HYPERPARAMETERS)

    print(f"Fitting CalibratedClassifierCV(method='{CALIBRATION_METHOD}', cv={CALIBRATION_CV})...")
    calibrated_model = CalibratedClassifierCV(base_model, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)
    calibrated_model.fit(X_train, y_train)

    val_metrics = evaluate(calibrated_model, X_val, y_val)
    test_metrics = evaluate(calibrated_model, X_test, y_test)

    print("\n--- Validation Metrics ---")
    print(f"PR-AUC: {val_metrics['pr_auc']}, ROC-AUC: {val_metrics['roc_auc']}, Brier: {val_metrics['brier']}")
    print(f"Recall@5%: {val_metrics['top_k']['top_5%']['recall']*100:.1f}%, Lift@5%: {val_metrics['top_k']['top_5%']['lift']:.2f}x")

    print("\n--- Untouched Test Metrics ---")
    print(f"PR-AUC: {test_metrics['pr_auc']}, ROC-AUC: {test_metrics['roc_auc']}, Brier: {test_metrics['brier']}")
    print(f"Recall@5%: {test_metrics['top_k']['top_5%']['recall']*100:.1f}%, Lift@5%: {test_metrics['top_k']['top_5%']['lift']:.2f}x")
    print(f"Recall@10%: {test_metrics['top_k']['top_10%']['recall']*100:.1f}%, Lift@10%: {test_metrics['top_k']['top_10%']['lift']:.2f}x")

    model_path = os.path.join(out_dir, "calibrated_xgboost_v3.pkl")
    joblib.dump(calibrated_model, model_path)
    print(f"\nSaved calibrated model artifact to {model_path}")

    return calibrated_model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_save()
