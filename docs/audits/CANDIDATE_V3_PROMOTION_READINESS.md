# Candidate V3 Promotion-Readiness Technical Review

**Date**: September 20, 2026
**Auditor**: Independent ML Engineering Review
**Subject**: Candidate V3 Failure Model (`models/candidate_v3/calibrated_xgboost_v3.pkl`)
**Production Status**: Production model (`models/calibrated_xgboost.pkl`) is **100% UNTOUCHED and UNCHANGED**.

---

## 1. Executive Summary

This document provides the final technical promotion-readiness evaluation of **Candidate V3** (`models/candidate_v3/calibrated_xgboost_v3.pkl`).

Following the discovery and resolution of the 31-day timedelta defect in the historical target generation pipeline (which excluded 614 true failure events, or 59.15% of all targetable failures), Candidate V3 was retrained, calibrated, and independently audited.

### Summary Assessment
- **Artifact & Contract Compatibility**: Candidate V3 is a drop-in replacement for the failure probability estimator within `FailureRiskPredictor`. It uses the exact 7 operational features in identical order, requires zero changes to `RailwayMLEngine`, and preserves all downstream decision, priority, and CP-SAT scheduling logic.
- **Measurable Superiority**: On the untouched 2026 test set under identical ground truth, Candidate V3 delivers statistically significant improvements in ROC-AUC (+10.4%, $p < 0.001$) and Top-10% Recall (+55.5%, $p = 0.032$), alongside a +31.0% gain in PR-AUC ($p = 0.063$) and 5.7x lower probability calibration error.
- **Readiness Classification**: **READY WITH CONDITIONS** (conditional on operational agreement on the operating threshold / ranking policy).

---

## 2. Artifact Verification

The saved candidate artifact was loaded and inspected directly from disk:
- **Path**: `models/candidate_v3/calibrated_xgboost_v3.pkl`
- **Class**: `sklearn.calibration.CalibratedClassifierCV`
- **Sub-Estimators**: **5** calibrated classifiers (fitted via 5-fold cross-validation on 2019–2024 training data).
- **Base Estimator**: `xgboost.sklearn.XGBClassifier`
- **Hyperparameter Concordance**: 100% exact match between `candidate_v3_manifest.json` and the fitted artifact (`n_estimators=242`, `max_depth=3`, `learning_rate=0.01896`, `min_child_weight=6`, `subsample=0.8197`, `colsample_bytree=0.8385`, `reg_alpha=0.00114`, `reg_lambda=0.1647`, `scale_pos_weight=2.02`, `eval_metric='aucpr'`).
- **Feature Contract**: Exactly 7 features in required order:
  1. `asset_age_years`
  2. `condition_score`
  3. `criticality`
  4. `usage_factor`
  5. `historical_failure_count`
  6. `historical_downtime_hours`
  7. `days_since_last_failure`
- `weather_stress` is **NOT required** by Candidate V3.

**Status: PASS.**

---

## 3. Production Compatibility

We verified how `FailureRiskPredictor` and `RailwayMLEngine` interact with the model:
1. **Public Engine Interface**: `RailwayMLEngine.evaluate_for_backend()` and `predict()` ingest identical input DataFrames and return identical dictionary structures across the 4 frozen categories:
   - `risk` (`failure_probability`, `predicted_failure_30d`)
   - `priority` (`maintenance_decision_score`, `decision_category`, `decision_rank`, `score_breakdown`)
   - `explanation` (`reason_tags`)
   - `scheduling` (`task_id`, `assigned_window`, `recommendations`)
2. **Downstream Multi-Objective Scoring**:
   `MaintenanceDecisionEngine` computes `failure_risk_factor = failure_probability.clip(0, 1)`. Candidate V3 outputs continuous probabilities bounded in $[0.005, 0.16]$, which seamlessly integrate into the 5-factor scoring formula.
3. **CP-SAT Optimizer & Top-3 Recommender**:
   Block window optimization and cross-department coordination depend on task priority and section constraints, not internal XGBoost tree structures. Zero changes required.

**Status: PASS.**

---

## 4. Threshold Compatibility

### Current Production Threshold (0.0120)
- **Definition Point**: Defined as `FailureRiskPredictor.DEFAULT_THRESHOLD = 0.0120` and referenced as a fallback in `RailwayMLEngine.evaluate_for_backend()`.
- **Scope of Use**: Used **exclusively** to assign the boolean classification flag `predicted_failure_30d = (failure_probability >= threshold)`. It is NOT used in priority score calculation, decision categories, or CP-SAT optimization.
- **Prevalence Shift**:
  - The production model was trained under an artificial ~0.47% prevalence (buggy target), yielding predictions that rarely exceeded 0.0120 (mean = 0.0039, max = 0.0126 on test).
  - Candidate V3 was trained on the corrected ~1.14% prevalence data, yielding calibrated probabilities reflecting genuine failure rates (mean = 0.0154, max = 0.1590 on test).
- **Threshold Analysis**:
  - At threshold **0.0120**, Candidate V3 flags ~43% to 45% of assets (recall 56.8% val / 62.0% test).
  - At threshold **0.0200**, Candidate V3 flags ~14.6% of assets (recall 30.4% val / 33.3% test, F1 = 0.0473).
  - Alternatively, operational maintenance can utilize **Top-K risk percentile ranking** (e.g., Top 5% or Top 10% risk fleet), which avoids fixed scalar cutoff drift entirely.

**Assessment**:
> Threshold requires post-promotion calibration/operational validation.

**Status: CONDITIONAL.**

---

## 5. Probability Semantics

We verified that Candidate V3 outputs strictly preserve correct operational semantics:
- **Definition**: "Estimated probability of operational failure within the next 30 days (specifically, prior to or at the next scheduled monthly inspection cycle)."
- **Forbidden Terminology Checked**:
  - No claims of "guaranteed failure" or "deterministic prediction."
  - No confusion between failure risk and maintenance urgency.
  - Heuristic confidence fields (`failure_model_confidence`, `heuristic_confidence`) remain completely excluded from the public backend contract payload.

**Status: PASS.**

---

## 6. Existing ML Pipeline Regression Results

The complete automated test suite was executed against the repository:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

- **Total Test Cases**: **160**
- **Passed**: **160**
- **Failed**: **0**
- **Skipped**: **0**
- **Execution Time**: 9.81s

### Regression Coverage Breakdown:
- `tests/test_failure_predictor.py`: 6 passed (baseline predictor loading, heuristic fallback, threshold overrides)
- `tests/test_candidate_v3_model.py`: 6 passed (candidate artifact loading, parameter check, bounds check, fallback)
- `tests/test_backend_contract.py`: 16 passed (JSON schema validation, native types, 4 frozen categories)
- `tests/test_maintenance_lifecycle.py`: 32 passed (clock abstraction, state machine, idempotency, human confirmation)
- `tests/test_top3_recommendations.py`: 42 passed (hard non-overlap, manpower, cross-department coordination bonus)
- `tests/test_planning_regression.py`: 32 passed (pressure calculation, corridor routing, decision rank stability)
- `tests/test_section_evidence.py`: 26 passed (station index, renamed codes, real capture evidence)

**Status: PASS.**

---

## 7. Inference Smoke Test

A dedicated inference smoke test was executed across 5 representative asset scenarios using `models/candidate_v3/calibrated_xgboost_v3.pkl`:

| Scenario | Asset ID | Age | Condition | Crit | Usage | Failures | Prob | Flag (0.020) | Decision Rank | Decision Score | Category |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low-Risk Asset** | `AST-LOW-01` | 1.5 yr | 98.0 | 3 | 0.60 | 0 | **0.0057** | 0 | 5 (Lowest) | 0.0867 | LOW |
| **Medium-Risk Asset**| `AST-MED-02` | 8.0 yr | 75.0 | 6 | 1.05 | 1 | **0.0114** | 0 | 4 | 0.2504 | MEDIUM |
| **High-Criticality** | `AST-CRIT-04` | 10.0 yr | 60.0 | 10 | 1.20 | 2 | **0.0340** | 1 | 2 | 0.4427 | MEDIUM |
| **High Historical** | `AST-HIST-05` | 14.0 yr | 52.0 | 7 | 1.30 | 5 | **0.0772** | 1 | 3 | 0.4407 | MEDIUM |
| **Severe High-Risk** | `AST-HIGH-03` | 18.0 yr | 42.0 | 8 | 1.45 | 4 | **0.1641** | 1 | 1 (Highest)| 0.5467 | HIGH |

### Smoke Test Invariants Confirmed:
1. **Determinism**: Identical repeated input produces identical floating point output ($\Delta = 0.000000$).
2. **Boundedness**: All probabilities are finite and strictly bounded $[0.0, 1.0]$.
3. **Monotonic Risk Gradient**: Severe high-risk ($0.1641$) > Historical ($0.0772$) > High-crit ($0.0340$) > Medium ($0.0114$) > Low ($0.0057$).
4. **Integration**: Output feeds seamlessly into `MaintenanceDecisionEngine` without runtime conversion errors.

**Status: PASS.**

---

## 8. Versioning & Rollback Readiness

The repository maintains strict artifact isolation:

### Current Production State
- **Artifact**: `models/calibrated_xgboost.pkl` (Hash/timestamp unchanged from baseline).
- **Manifest**: `config/pipeline_manifest.json` and `models/tuned_xgboost_params.json`.
- **Threshold**: 0.0120.

### Candidate V3 State
- **Artifact**: `models/candidate_v3/calibrated_xgboost_v3.pkl`.
- **Manifest**: `models/candidate_v3/candidate_v3_manifest.json`.
- **Evaluation**: `models/candidate_v3/candidate_v3_evaluation.json`.
- **Sub-Variants**: `models/candidate_v3/baseline/` and `models/candidate_v3/tuned/`.

### Rollback Procedure
If Candidate V3 is promoted and an unexpected issue emerges, rollback requires only restoring `models/calibrated_xgboost.pkl` (or redirecting `model_path` in `FailureRiskPredictor.__init__`), taking less than 1 minute with zero database migrations or code rollbacks.

**Status: PASS.**

---

## 9. Reproducibility

The training, calibration, and evaluation pipeline is fully encapsulated in:
```bash
scripts/train_candidate_v3.py
```
- **Inputs**: `data/processed/candidate_v3/failure_train.csv`, `failure_validation.csv`, `failure_test.csv`.
- **Target**: `failure_within_30_days` (1-month window).
- **Features**: Explicitly defined 7-column list.
- **Random Seed**: Fixed (`seed=42`).
- **Dependencies**: Uses standard pinned packages in `.venv` (`xgboost 3.4.1`, `scikit-learn 1.9.0`).
- **Output**: Generates identical artifact and reproduces exact validation and test metrics.

**Status: PASS.**

---

## 10. Documentation Review

All relevant documentation files were reviewed for accuracy, consistency, and scientific honesty:
- `docs/MODEL_IMPROVEMENT_REPORT.md`: Verified. Updated to explicitly state that PR-AUC gain (+31.0%, $p=0.063$) is observed but not statistically significant at $\alpha=0.05$, while ROC-AUC ($p<0.001$) and Top-10% Recall ($p=0.032$) are statistically significant. Replaced claims of "full signal capacity" with "substantially improves ranking performance on the corrected synthetic test set."
- `docs/CANDIDATE_V3_FINAL_AUDIT.md`: Verified. Contains exact 1,000-resample bootstrap distributions, quantile calibration tables, and fairness breakdowns.
- `docs/ML_BACKEND_INTEGRATION_CONTRACT.md`: Verified. Accurately describes the frozen 4-category contract with `max_manpower = 12` and no outdated 88.5% recall claims.

**Status: PASS.**

---

## 11. Remaining Technical Risks

| Risk Item | Category | Description | Mitigation Strategy |
|:---|:---:|:---|:---|
| **Threshold Mismatch** | **HIGH** | Candidate V3 probabilities reflect 1.14% prevalence. Old 0.0120 threshold flags 43% of fleet. | Deploy with threshold 0.0200 or utilize Top-K fleet percentile ranking. |
| **Synthetic Aleatoric Noise** | **MEDIUM** | Generator caps $p^* \le 0.0586$, bounding precision below 5.86%. | Frame model strictly as a risk ranker for inspection targeting, not a deterministic binary alarm. |
| **Autoregressive Dynamics Gap** | **LOW** | Synthetic condition scores fluctuate with monthly white noise rather than physical wear laws. | Document limitation; model successfully relies on static age, condition level, usage, and past failures. |
| **Rollback Risk** | **LOW** | Potential operational disruption during deployment. | Production model remains preserved; rollback is instantaneous. |

---

## 12. Promotion Conditions

Candidate V3 is technically sound and ready for production promotion under the following **3 explicit conditions**:

1. **Operational Threshold Alignment**:
   Before updating default inference settings, maintenance planning leadership must approve either:
   - Setting the binary cutoff threshold to **0.0200** (flagging ~14.6% of assets with 30.4% recall), OR
   - Adopting **Top-K risk ranking** (e.g. prioritizing the Top 5% or Top 10% highest-risk assets) as the primary operational surface.
2. **Preservation of Rollback Asset**:
   Prior to copying `models/candidate_v3/calibrated_xgboost_v3.pkl` to `models/calibrated_xgboost.pkl`, the existing production artifact must be archived as `models/calibrated_xgboost_v2_legacy.pkl`.
3. **Formal Operator Authorization**:
   In adherence to human-in-the-loop safety protocols, model promotion must be explicitly authorized by engineering leadership rather than executed automatically.

---

## 13. Final Readiness Classification

### **READY WITH CONDITIONS**

Candidate V3 represents a substantial, statistically verified advancement over the baseline model:
- Eliminates the 31-day target truncation bug that corrupted 59.15% of historical failure labels.
- Delivers statistically significant improvements in ROC-AUC (0.6339 $\to$ 0.6996, $p < 0.001$) and Top-10% Recall (16.67% $\to$ 25.93%, $p = 0.032$).
- Improves probability calibration error by 5.7x across risk quintiles.
- Retains 100% contract compatibility with the frozen 4-category backend specification.

---

PROMOTION READINESS
-------------------
Artifact: PASS
Production Compatibility: PASS
Threshold Compatibility: CONDITIONAL
Inference Smoke Test: PASS
Regression Tests: PASS
Versioning: PASS
Rollback Readiness: PASS
Reproducibility: PASS
Documentation: PASS

FINAL STATUS:
READY WITH CONDITIONS

Production Model:
UNCHANGED

Backend:
UNCHANGED

API:
NOT IMPLEMENTED

Candidate V3:
NOT PROMOTED
