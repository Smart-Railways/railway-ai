import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix
)

# 1. Load data
train_df = pd.read_csv("data/processed/candidate_v3/failure_train.csv", parse_dates=["snapshot_date"])
val_df = pd.read_csv("data/processed/candidate_v3/failure_validation.csv", parse_dates=["snapshot_date"])
test_df = pd.read_csv("data/processed/candidate_v3/failure_test.csv", parse_dates=["snapshot_date"])

feat_7 = [
    "asset_age_years", "condition_score", "criticality", "usage_factor",
    "historical_failure_count", "historical_downtime_hours", "days_since_last_failure"
]

X_tr = train_df[feat_7].fillna(0)
y_tr = train_df["failure_within_30_days"].values

X_val = val_df[feat_7].fillna(0)
y_val = val_df["failure_within_30_days"].values

X_te = test_df[feat_7].fillna(0)
y_te = test_df["failure_within_30_days"].values

# Best hyperparameters from Optuna validation search
best_params = {
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

# Create output directories
os.makedirs("models/candidate_v3/baseline", exist_ok=True)
os.makedirs("models/candidate_v3/tuned", exist_ok=True)
os.makedirs("models/candidate_v3/calibrated", exist_ok=True)
os.makedirs("models/candidate_v3/reports", exist_ok=True)

# Define models to compare
# A. Current Production Model
prod_model = joblib.load("models/production/calibrated_xgboost.pkl")

# B. V3 Baseline XGBoost (default params, uncalibrated)
base_xgb = xgb.XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.85,
    colsample_bytree=0.8, scale_pos_weight=1.0, random_state=42, eval_metric="aucpr"
)
base_xgb.fit(X_tr, y_tr)
joblib.dump(base_xgb, "models/candidate_v3/baseline/xgb_baseline_uncalibrated.pkl")

# B2. V3 Baseline with Sigmoid Calibration
cal_base_xgb = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=5)
cal_base_xgb.fit(X_tr, y_tr)
joblib.dump(cal_base_xgb, "models/candidate_v3/baseline/xgb_baseline_calibrated.pkl")

# C. V3 Tuned XGBoost (Optuna params, uncalibrated)
tuned_xgb = xgb.XGBClassifier(**best_params)
tuned_xgb.fit(X_tr, y_tr)
joblib.dump(tuned_xgb, "models/candidate_v3/tuned/xgb_tuned_uncalibrated.pkl")

# D. V3 Best Calibrated Candidate (Optuna params + Sigmoid Calibration cv=5)
cal_tuned_xgb = CalibratedClassifierCV(tuned_xgb, method="sigmoid", cv=5)
cal_tuned_xgb.fit(X_tr, y_tr)
joblib.dump(cal_tuned_xgb, "models/candidate_v3/calibrated/calibrated_xgboost_v3.pkl")
# Also save as main candidate artifact
joblib.dump(cal_tuned_xgb, "models/candidate/candidate_v3/calibrated_xgboost_v3.pkl")

# E. Logistic Regression Baseline
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_tr, y_tr)

# F. Random Forest Baseline
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
rf.fit(X_tr, y_tr)

model_suite = [
    ("Current Production Model (calibrated XGBoost v2)", prod_model, "Sigmoid", 0.0120),
    ("Logistic Regression Baseline", lr, "None", 0.5000),
    ("Random Forest Baseline", rf, "None", 0.5000),
    ("V3 XGBoost Baseline (uncalibrated)", base_xgb, "None", 0.0120),
    ("V3 XGBoost Baseline (sigmoid cv=5)", cal_base_xgb, "Sigmoid (cv=5)", 0.0120),
    ("V3 Tuned XGBoost (uncalibrated)", tuned_xgb, "None", 0.0200),
    ("V3 Best Candidate (tuned + sigmoid cv=5)", cal_tuned_xgb, "Sigmoid (cv=5)", 0.0200),
]

def eval_split(model, X, y, threshold):
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
        cap = int(y[top_idx].sum())
        p_k = cap / k
        r_k = cap / max(1, pos)
        lift = p_k / base_rate if base_rate > 0 else 0.0
        top_k[int(pct*100)] = {"k": k, "prec": p_k, "rec": r_k, "lift": lift}

    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier,
        "prec": prec, "rec": rec, "f1": f1,
        "top_1": top_k[1], "top_5": top_k[5], "top_10": top_k[10]
    }

comparison_table = []
eval_dump = {}

for m_name, m_obj, cal_method, thresh in model_suite:
    val_m = eval_split(m_obj, X_val, y_val, thresh)
    te_m = eval_split(m_obj, X_te, y_te, thresh)

    eval_dump[m_name] = {
        "calibration": cal_method,
        "threshold": thresh,
        "validation": val_m,
        "test": te_m
    }

    comparison_table.append({
        "Model": m_name,
        "Calibration": cal_method,
        "Threshold": thresh,
        "Val PR-AUC": round(val_m["pr_auc"], 4),
        "Test PR-AUC": round(te_m["pr_auc"], 4),
        "Val ROC": round(val_m["roc_auc"], 4),
        "Test ROC": round(te_m["roc_auc"], 4),
        "Val Rec@1%": f"{val_m['top_1']['rec']*100:.1f}%",
        "Test Rec@1%": f"{te_m['top_1']['rec']*100:.1f}%",
        "Val Rec@5%": f"{val_m['top_5']['rec']*100:.1f}%",
        "Test Rec@5%": f"{te_m['top_5']['rec']*100:.1f}%",
        "Val Rec@10%": f"{val_m['top_10']['rec']*100:.1f}%",
        "Test Rec@10%": f"{te_m['top_10']['rec']*100:.1f}%",
        "Val Lift@1%": f"{val_m['top_1']['lift']:.2f}x",
        "Test Lift@1%": f"{te_m['top_1']['lift']:.2f}x",
        "Val Lift@5%": f"{val_m['top_5']['lift']:.2f}x",
        "Test Lift@5%": f"{te_m['top_5']['lift']:.2f}x",
        "Val Brier": round(val_m["brier"], 5),
        "Test Brier": round(te_m["brier"], 5),
    })

comp_df = pd.DataFrame(comparison_table)
print("\n=== FINAL MODEL COMPARISON TABLE ===")
print(comp_df.to_string(index=False))

# Save evaluations to models/candidate_v3/
with open("models/candidate_v3/candidate_v3_evaluation.json", "w") as f:
    json.dump(eval_dump, f, indent=2)

# Save candidate_v3 manifest
manifest = {
    "model_name": "candidate_v3_calibrated_xgboost",
    "version": "3.0.0-candidate",
    "target_horizon": "30d_next_monthly_observation",
    "target_definition": "prediction_date < failure_date <= prediction_date + DateOffset(months=1)",
    "training_data": "data/processed/candidate_v3/failure_train.csv (72,000 rows, 782 failures)",
    "validation_data": "data/processed/candidate_v3/failure_validation.csv (12,000 rows, 148 failures)",
    "test_data": "data/processed/candidate_v3/failure_test.csv (7,000 rows, 108 failures)",
    "feature_columns": feat_7,
    "hyperparameters": best_params,
    "calibration_method": "CalibratedClassifierCV(method='sigmoid', cv=5)",
    "operating_threshold": 0.0200,
    "operational_mode": "risk_ranking",
    "metrics": {
        "validation": {
            "pr_auc": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["pr_auc"],
            "roc_auc": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["roc_auc"],
            "brier_score": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["brier"],
            "recall_at_1pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["top_1"]["rec"],
            "recall_at_5pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["top_5"]["rec"],
            "lift_at_1pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["top_1"]["lift"],
            "lift_at_5pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["validation"]["top_5"]["lift"]
        },
        "test_untouched": {
            "pr_auc": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["pr_auc"],
            "roc_auc": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["roc_auc"],
            "brier_score": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["brier"],
            "recall_at_1pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["top_1"]["rec"],
            "recall_at_5pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["top_5"]["rec"],
            "lift_at_1pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["top_1"]["lift"],
            "lift_at_5pct": eval_dump["V3 Best Candidate (tuned + sigmoid cv=5)"]["test"]["top_5"]["lift"]
        }
    },
    "production_promotion_status": "READY FOR PROMOTION REVIEW",
    "promoted_to_production": False
}

with open("models/candidate_v3/candidate_v3_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\nSaved candidate manifest to models/candidate_v3/candidate_v3_manifest.json")
print("Saved evaluation report to models/candidate_v3/candidate_v3_evaluation.json")
