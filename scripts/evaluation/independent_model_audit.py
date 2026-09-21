import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve

print("=== 1. ARTIFACT & MANIFEST VERIFICATION ===")

candidate_model_path = "models/candidate/candidate_v3/calibrated_xgboost_v3.pkl"
manifest_path = "models/candidate_v3/candidate_v3_manifest.json"
eval_path = "models/candidate_v3/candidate_v3_evaluation.json"

assert os.path.exists(candidate_model_path), f"Missing {candidate_model_path}"
assert os.path.exists(manifest_path), f"Missing {manifest_path}"
assert os.path.exists(eval_path), f"Missing {eval_path}"

cand_model = joblib.load(candidate_model_path)
print("Candidate model loaded successfully:", type(cand_model))
print("Number of calibrated estimators:", len(cand_model.calibrated_classifiers_))
print("Estimator type:", type(cand_model.estimator))

with open(manifest_path) as f:
    manifest = json.load(f)
with open(eval_path) as f:
    eval_dump = json.load(f)

print("Manifest model_name:", manifest["model_name"])
print("Manifest version:", manifest["version"])
print("Manifest features:", manifest["feature_columns"])
print("Manifest threshold:", manifest["operating_threshold"])

feat_7 = manifest["feature_columns"]

# Verify base estimator features and params
base_est = cand_model.estimator
print("\nBase XGBoost parameters from artifact:")
for k, v in manifest["hyperparameters"].items():
    est_val = getattr(base_est, k, None)
    match = (est_val == v) if not isinstance(v, float) else np.isclose(est_val, v)
    print(f"  {k:20s}: manifest={v}, artifact={est_val} -> Match={match}")

# 2. Dataset loading and verification
print("\n=== 2. DATASET & SPLIT INTEGRITY VERIFICATION ===")
orig_test = pd.read_csv("data/processed/failure_test.csv", parse_dates=["snapshot_date"])
v3_test = pd.read_csv("data/processed/candidate_v3/failure_test.csv", parse_dates=["snapshot_date"])

print("Original test shape:", orig_test.shape)
print("Candidate V3 test shape:", v3_test.shape)

# Check that rows, dates, assets match 100%
assert len(orig_test) == len(v3_test), "Row count mismatch between original and v3 test sets!"
assert (orig_test["asset_id"] == v3_test["asset_id"]).all(), "Asset IDs mismatch between test sets!"
assert (orig_test["snapshot_date"] == v3_test["snapshot_date"]).all(), "Snapshot dates mismatch between test sets!"

for f in feat_7:
    diff = np.abs(orig_test[f].fillna(0) - v3_test[f].fillna(0)).max()
    print(f"Max feature diff between original and v3 test for {f:25s}: {diff}")
    assert diff < 1e-6, f"Feature mismatch in {f}!"

print("Original test positive failures (buggy target):", orig_test["failure_within_30_days"].sum())
print("Candidate V3 test positive failures (corrected target):", v3_test["failure_within_30_days"].sum())

# 3. Model Inference & Recalculation on Untouched Test Set
print("\n=== 3. INDEPENDENT RE-EVALUATION OF UNTOUCHED TEST SET ===")
prod_model = joblib.load("models/production/calibrated_xgboost.pkl")

X_test = v3_test[feat_7].fillna(0.0)
y_test_orig = orig_test["failure_within_30_days"].values
y_test_corr = v3_test["failure_within_30_days"].values

# Model predictions
prod_probs = prod_model.predict_proba(X_test)[:, 1]
cand_probs = cand_model.predict_proba(X_test)[:, 1]

# Verification of probability output bounds and sanity
assert np.isfinite(cand_probs).all(), "Candidate probabilities contain non-finite values!"
assert (cand_probs >= 0.0).all() and (cand_probs <= 1.0).all(), "Candidate probabilities out of [0, 1]!"
assert np.isfinite(prod_probs).all(), "Production probabilities contain non-finite values!"

print(f"Prod probs min={prod_probs.min():.5f}, mean={prod_probs.mean():.5f}, max={prod_probs.max():.5f}")
print(f"Cand probs min={cand_probs.min():.5f}, mean={cand_probs.mean():.5f}, max={cand_probs.max():.5f}")

def calc_all_metrics(y_true, probs, threshold):
    n = len(y_true)
    pos = int(y_true.sum())
    base_rate = pos / max(1, n)

    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    preds = (probs >= threshold).astype(int)
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    # Top-K
    top_k = {}
    for pct in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(n * pct))
        top_idx = np.argsort(probs)[-k:]
        cap = int(y_true[top_idx].sum())
        p_k = cap / k
        r_k = cap / max(1, pos)
        lift = p_k / base_rate if base_rate > 0 else 0.0
        top_k[int(pct*100)] = {"k": k, "captured": cap, "precision": p_k, "recall": r_k, "lift": lift}

    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier,
        "prec": prec, "rec": rec, "f1": f1,
        "top_k": top_k
    }

# A. Production on Original Labels
m_prod_orig = calc_all_metrics(y_test_orig, prod_probs, threshold=0.0120)
# B. Production on Corrected Labels
m_prod_corr = calc_all_metrics(y_test_corr, prod_probs, threshold=0.0120)
# C. Candidate V3 on Corrected Labels
m_cand_corr = calc_all_metrics(y_test_corr, cand_probs, threshold=0.0200)

print("\n--- RECALCULATED UNTOUCHED TEST SET METRICS ---")
print(f"A. Production on Original Buggy Target (52 positives):")
print(f"   PR-AUC: {m_prod_orig['pr_auc']:.5f} | ROC-AUC: {m_prod_orig['roc_auc']:.5f} | Brier: {m_prod_orig['brier']:.5f}")
print(f"   Rec@1%: {m_prod_orig['top_k'][1]['recall']*100:.2f}% | Rec@5%: {m_prod_orig['top_k'][5]['recall']*100:.2f}% | Rec@10%: {m_prod_orig['top_k'][10]['recall']*100:.2f}%")
print(f"   Lift@1%: {m_prod_orig['top_k'][1]['lift']:.2f}x | Lift@5%: {m_prod_orig['top_k'][5]['lift']:.2f}x | Lift@10%: {m_prod_orig['top_k'][10]['lift']:.2f}x")

print(f"\nB. Production on Corrected Target (108 positives):")
print(f"   PR-AUC: {m_prod_corr['pr_auc']:.5f} | ROC-AUC: {m_prod_corr['roc_auc']:.5f} | Brier: {m_prod_corr['brier']:.5f}")
print(f"   Rec@1%: {m_prod_corr['top_k'][1]['recall']*100:.2f}% | Rec@5%: {m_prod_corr['top_k'][5]['recall']*100:.2f}% | Rec@10%: {m_prod_corr['top_k'][10]['recall']*100:.2f}%")
print(f"   Lift@1%: {m_prod_corr['top_k'][1]['lift']:.2f}x | Lift@5%: {m_prod_corr['top_k'][5]['lift']:.2f}x | Lift@10%: {m_prod_corr['top_k'][10]['lift']:.2f}x")

print(f"\nC. Candidate V3 on Corrected Target (108 positives):")
print(f"   PR-AUC: {m_cand_corr['pr_auc']:.5f} | ROC-AUC: {m_cand_corr['roc_auc']:.5f} | Brier: {m_cand_corr['brier']:.5f}")
print(f"   Rec@1%: {m_cand_corr['top_k'][1]['recall']*100:.2f}% | Rec@5%: {m_cand_corr['top_k'][5]['recall']*100:.2f}% | Rec@10%: {m_cand_corr['top_k'][10]['recall']*100:.2f}%")
print(f"   Lift@1%: {m_cand_corr['top_k'][1]['lift']:.2f}x | Lift@5%: {m_cand_corr['top_k'][5]['lift']:.2f}x | Lift@10%: {m_cand_corr['top_k'][10]['lift']:.2f}x")

# Check exact reproducibility with reported numbers
print("\n--- EXACT REPRODUCIBILITY CHECK ---")
print(f"Reported Candidate V3 Test PR-AUC: 0.03375 vs Recalculated: {m_cand_corr['pr_auc']:.5f} (Diff: {abs(0.03375 - m_cand_corr['pr_auc']):.6f})")
print(f"Reported Candidate V3 Test ROC-AUC: 0.69963 vs Recalculated: {m_cand_corr['roc_auc']:.5f} (Diff: {abs(0.69963 - m_cand_corr['roc_auc']):.6f})")
print(f"Reported Candidate V3 Test Brier: 0.01510 vs Recalculated: {m_cand_corr['brier']:.5f} (Diff: {abs(0.01510 - m_cand_corr['brier']):.6f})")
print(f"Reported Candidate V3 Test Rec@5%: 12.04% vs Recalculated: {m_cand_corr['top_k'][5]['recall']*100:.2f}%")
print(f"Reported Candidate V3 Test Rec@10%: 25.93% vs Recalculated: {m_cand_corr['top_k'][10]['recall']*100:.2f}%")
print(f"Reported Candidate V3 Test Lift@10%: 2.59x vs Recalculated: {m_cand_corr['top_k'][10]['lift']:.2f}x")

# 4. Calibration Curve and Binned Risk Analysis
print("\n=== 4. CALIBRATION & RISK BIN AUDIT ===")
prob_true_prod, prob_pred_prod = calibration_curve(y_test_corr, prod_probs, n_bins=5, strategy="quantile")
prob_true_cand, prob_pred_cand = calibration_curve(y_test_corr, cand_probs, n_bins=5, strategy="quantile")

print("Production Model Calibration (Quantile Bins on Corrected Test Set):")
for pt, pp in zip(prob_true_prod, prob_pred_prod):
    print(f"  Predicted mean: {pp:.5f} | Observed empirical rate: {pt:.5f} | Error: {abs(pp - pt):.5f}")

print("\nCandidate V3 Model Calibration (Quantile Bins on Corrected Test Set):")
for pt, pp in zip(prob_true_cand, prob_pred_cand):
    print(f"  Predicted mean: {pp:.5f} | Observed empirical rate: {pt:.5f} | Error: {abs(pp - pt):.5f}")

# 5. Bootstrap Confidence Intervals (1000 resamples)
print("\n=== 5. STATISTICAL BOOTSTRAP UNCERTAINTY (1,000 resamples) ===")
np.random.seed(42)
n_bootstraps = 1000
n_samples = len(y_test_corr)

boot_metrics = {
    "prod_prauc": [], "cand_prauc": [], "diff_prauc": [],
    "prod_rocauc": [], "cand_rocauc": [], "diff_rocauc": [],
    "prod_rec5": [], "cand_rec5": [], "diff_rec5": [],
    "prod_rec10": [], "cand_rec10": [], "diff_rec10": [],
    "prod_lift10": [], "cand_lift10": [], "diff_lift10": []
}

for i in range(n_bootstraps):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    y_b = y_test_corr[idx]
    if y_b.sum() == 0 or y_b.sum() == n_samples:
        continue

    pp_b = prod_probs[idx]
    cp_b = cand_probs[idx]

    pr_p = average_precision_score(y_b, pp_b)
    pr_c = average_precision_score(y_b, cp_b)
    roc_p = roc_auc_score(y_b, pp_b)
    roc_c = roc_auc_score(y_b, cp_b)

    k5 = max(1, int(n_samples * 0.05))
    rec5_p = y_b[np.argsort(pp_b)[-k5:]].sum() / max(1, y_b.sum())
    rec5_c = y_b[np.argsort(cp_b)[-k5:]].sum() / max(1, y_b.sum())

    k10 = max(1, int(n_samples * 0.10))
    rec10_p = y_b[np.argsort(pp_b)[-k10:]].sum() / max(1, y_b.sum())
    rec10_c = y_b[np.argsort(cp_b)[-k10:]].sum() / max(1, y_b.sum())

    base_rate_b = y_b.sum() / n_samples
    lift10_p = (y_b[np.argsort(pp_b)[-k10:]].sum() / k10) / base_rate_b if base_rate_b > 0 else 0
    lift10_c = (y_b[np.argsort(cp_b)[-k10:]].sum() / k10) / base_rate_b if base_rate_b > 0 else 0

    boot_metrics["prod_prauc"].append(pr_p)
    boot_metrics["cand_prauc"].append(pr_c)
    boot_metrics["diff_prauc"].append(pr_c - pr_p)

    boot_metrics["prod_rocauc"].append(roc_p)
    boot_metrics["cand_rocauc"].append(roc_c)
    boot_metrics["diff_rocauc"].append(roc_c - roc_p)

    boot_metrics["prod_rec5"].append(rec5_p)
    boot_metrics["cand_rec5"].append(rec5_c)
    boot_metrics["diff_rec5"].append(rec5_c - rec5_p)

    boot_metrics["prod_rec10"].append(rec10_p)
    boot_metrics["cand_rec10"].append(rec10_c)
    boot_metrics["diff_rec10"].append(rec10_c - rec10_p)

    boot_metrics["prod_lift10"].append(lift10_p)
    boot_metrics["cand_lift10"].append(lift10_c)
    boot_metrics["diff_lift10"].append(lift10_c - lift10_p)

def ci(arr):
    return np.percentile(arr, 2.5), np.median(arr), np.percentile(arr, 97.5)

print("\n--- 95% BOOTSTRAP CONFIDENCE INTERVALS (Corrected Test Set) ---")
for key in ["prauc", "rocauc", "rec5", "rec10", "lift10"]:
    p_low, p_med, p_high = ci(boot_metrics[f"prod_{key}"])
    c_low, c_med, c_high = ci(boot_metrics[f"cand_{key}"])
    d_low, d_med, d_high = ci(boot_metrics[f"diff_{key}"])
    p_val_zero = (np.array(boot_metrics[f"diff_{key}"]) <= 0).mean()
    print(f"\nMetric: {key.upper()}")
    print(f"  Production:    median={p_med:.4f} [95% CI: {p_low:.4f} to {p_high:.4f}]")
    print(f"  Candidate V3:  median={c_med:.4f} [95% CI: {c_low:.4f} to {c_high:.4f}]")
    print(f"  Delta (C - P): median={d_med:+.4f} [95% CI: {d_low:+.4f} to {d_high:+.4f}]")
    print(f"  P(Delta <= 0): {p_val_zero:.4f}")
