# Candidate V3 Controlled Production Promotion Report

**Promotion Timestamp**: September 20, 2026, 23:44:00 IST (2026-09-20T23:44:00+05:30)
**Environment**: Local WSL2 Ubuntu Linux (`/home/chirag/projects/railway-ai`)
**Promotion Mode**: Controlled Local Artifact Promotion

---

## 1. Executive Summary

This document records the official, controlled promotion of the **Candidate V3 Failure Model** to active local production within the Railway-AI ML repository.

Following the discovery and resolution of the 31-day timedelta defect in the historical target generation pipeline (which had dropped 614 of 1,038 failure events, or 59.15% of all ground-truth targetable failures), Candidate V3 was trained on the corrected dataset (`data/processed/candidate_v3/`), calibrated via 5-fold cross-validation Platt scaling, and independently audited with 1,000-resample bootstrap verification.

Candidate V3 has been promoted to `models/calibrated_xgboost.pkl`. The previous production model has been safely backed up and verified as `models/calibrated_xgboost_v2_legacy.pkl`.

---

## 2. Artifact Verification & Hashes

### Pre-Promotion Artifact States
- **Original Production Model Path**: `models/calibrated_xgboost.pkl`
  - File Size: `2,805,132` bytes
  - SHA256: `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386`
  - Model Type: `sklearn.calibration.CalibratedClassifierCV` (5 sub-estimators, sigmoid method)
  - Base Estimator: `xgboost.sklearn.XGBClassifier`
  - Features: Exactly 7 operational features

- **Candidate V3 Artifact Path**: `models/candidate_v3/calibrated_xgboost_v3.pkl`
  - File Size: `1,458,332` bytes
  - SHA256: `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae`
  - Model Type: `sklearn.calibration.CalibratedClassifierCV` (5 sub-estimators, sigmoid method)
  - Base Estimator: `xgboost.sklearn.XGBClassifier`
  - Features: Exactly 7 operational features

---

## 3. Backup & Promotion Verification

### Step 2: Backup of Current Production
The original production artifact was copied to:
`models/calibrated_xgboost_v2_legacy.pkl`

- Archived File SHA256: `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386`
- Original Production SHA256: `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386`
- **Result**: Byte-for-byte identical (**PASS**).

### Step 3: Promotion of Candidate V3
`models/candidate_v3/calibrated_xgboost_v3.pkl` was copied to `models/calibrated_xgboost.pkl`.
The original source `models/candidate_v3/calibrated_xgboost_v3.pkl` was strictly preserved.

- New Production SHA256: `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae`
- Candidate Source SHA256: `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae`
- **Result**: Hash matches Candidate V3 exactly (**PASS**).

### Artifact Hash Summary

| Artifact Role | File Path | SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Legacy Production Backup** | `models/calibrated_xgboost_v2_legacy.pkl` | `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386` | **VERIFIED** |
| **Active Production Model** | `models/calibrated_xgboost.pkl` | `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` | **PROMOTED** |
| **Candidate V3 Copy** | `models/candidate_v3/calibrated_xgboost_v3.pkl` | `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` | **PRESERVED** |

---

## 4. Threshold Invariant Status

- **Operating Decision Threshold**: `0.0120` (**UNCHANGED**).
- In accordance with promotion safety constraints:
  - The threshold was **NOT** replaced with `0.0200`.
  - No new threshold was invented or tuned during promotion.
  - `DEFAULT_THRESHOLD = 0.0120` in `FailureRiskPredictor` remains unchanged.
  - Threshold adjustment policy is designated as a Post-Promotion Validation Item.

---

## 5. Production Loading & Inference Smoke Test

### Production Loading Verification
`models/calibrated_xgboost.pkl` was loaded through the official production path:
- `FailureRiskPredictor()`: Model loaded successfully via `joblib.load()`, `is_available == True`.
- `RailwayMLEngine()`: Initialized with status `"ok"`, `artifacts.calibrated_xgboost_available == True`.
- Output Schema: 4 frozen categories (`risk`, `priority`, `explanation`, `scheduling`) intact.
- Heuristic Confidence Field: **NOT PRESENT** (verified absent from both `risk` and root payload).

### Inference Smoke Test Results (5 Representative Assets)

| Asset ID | Category | Age | Cond | Failures | Prob | Flag ($\tau=0.0120$) | Decision Rank | Priority Score | Reason Tags |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **AST-LOW-01** | Low Risk | 1.5 yr | 98.0 | 0 | **0.00570** | 0 | 5 | 0.0817 (LOW) | `['ROUTINE_INSPECTION']` |
| **AST-MED-02** | Medium Risk | 8.0 yr | 75.0 | 1 | **0.01140** | 0 | 4 | 0.2284 (LOW) | `['OVERDUE_MAINTENANCE']` |
| **AST-HIGH-03** | High Risk | 18.0 yr | 42.0 | 4 | **0.16410** | 1 | 1 | 0.4792 (MED) | `['HIGH_URGENCY', 'HIGH_CRITICALITY', 'OVERDUE_MAINTENANCE']` |
| **AST-CRIT-04** | High Criticality | 10.0 yr | 60.0 | 2 | **0.03400** | 1 | 2 | 0.4252 (MED) | `['HIGH_URGENCY', 'HIGH_CRITICALITY', 'OVERDUE_MAINTENANCE']` |
| **AST-HIST-05** | High History | 14.0 yr | 52.0 | 5 | **0.07720** | 1 | 3 | 0.4032 (MED) | `['HIGH_URGENCY', 'OVERDUE_MAINTENANCE']` |

- **Risk Gradient Verification**: $p_{high} (0.16410) > p_{med} (0.01140) > p_{low} (0.00570)$ (**PASS**).
- **Ranking Verification**: Degraded/high-risk assets rank higher in priority than healthy assets (**PASS**).
- **Scheduling Verification**: Feasible candidate maintenance windows successfully generated for all assets (**PASS**).

---

## 6. Full Regression Test Suite Result

The full automated test suite was executed in the production virtual environment:
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```
**Result**:
```
........................................................................ [ 45%]
........................................................................ [ 90%]
................                                                         [100%]
160 passed in 9.53s
```
- **Passed**: 160
- **Failed**: 0
- **Skipped**: 0
- **Regressions**: **ZERO**

---

## 7. Non-ML Component Verification

All non-ML business rules, optimization constraints, and contracts were verified as **100% UNCHANGED**:
1. **Priority Formula**: Unified 5-factor additive formula:
   $$\text{Priority} = 0.30 \times \text{Risk} + 0.20 \times \text{Criticality} + 0.20 \times \text{Urgency} + 0.15 \times \text{Overdue} + 0.15 \times \text{Operational}$$
2. **Priority Weights**: Sum strictly to 1.0 (`[0.30, 0.20, 0.20, 0.15, 0.15]`).
3. **Priority Categories**: `LOW` ($< 0.25$), `MEDIUM` ($[0.25, 0.50)$), `HIGH` ($[0.50, 0.75)$), `CRITICAL` ($\ge 0.75$).
4. **CP-SAT Block Optimizer**:
   - Up to Top-3 ranked candidate windows per task (`max_recommendations=3`).
   - Manpower capacity constraints preserved.
   - Same-section non-overlapping maintenance block constraints preserved.
   - Cross-department coordination preserved.
5. **Human-in-the-Loop Invariants**:
   - `selection_status = "RECOMMENDED"`, `human_confirmation_required = true`, `is_booked = false`, `booking_status = "UNBOOKED"`.
   - No automatic booking.
6. **Maintenance Lifecycle Manager**:
   - State machine transitions (`GENERATED`, `UPCOMING`, `ACTIVE`, `COMPLETED`, `MISSED`, `RESCHEDULED`) and append-only event log verified intact.
7. **Integration Contract & Schema**:
   - `docs/ML_BACKEND_INTEGRATION_CONTRACT.md` and `docs/schemas/ml_response.schema.json` verified intact.

---

## 8. Rollback Procedure & Verification

### Mechanical Verification
The legacy rollback artifact `models/calibrated_xgboost_v2_legacy.pkl` was loaded directly using `joblib` and initialized via `FailureRiskPredictor(model_path="models/calibrated_xgboost_v2_legacy.pkl")`. Inference was executed successfully (`prob = 0.00610`). The artifact is fully functional.

### Rollback Runbook (Zero Code Changes Required)
If an unexpected operational issue arises requiring rollback to V2:
1. **Restore Artifact**:
   ```bash
   cp models/calibrated_xgboost_v2_legacy.pkl models/calibrated_xgboost.pkl
   ```
2. **Verify Restored Hash**:
   ```bash
   sha256sum models/calibrated_xgboost.pkl
   # Must equal: 3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386
   ```
3. **Run Regression Tests**:
   ```bash
   PYTHONPATH=. .venv/bin/pytest tests/ -q
   ```
4. **Verify Engine Health**:
   ```bash
   PYTHONPATH=. .venv/bin/python -c "from src.services.ml_engine import RailwayMLEngine; print(RailwayMLEngine().health())"
   ```
Execution time: < 30 seconds. Zero database migrations or API contract adjustments required.

---

## 9. Known Limitations

1. **Synthetic Data**: The model was trained and evaluated on synthetic monthly asset snapshots (1,000 simulated assets across 91 monthly snapshots). Real railway failure telemetry remains required for real-world field validation.
2. **Statistical Significance Bounds**:
   - ROC-AUC improvement ($\Delta = +0.0650$) is statistically significant ($p < 0.001$, 95% CI $[+0.0251, +0.1087]$).
   - Top-10% Recall improvement (16.67% $\to$ 25.93%, Lift 2.59x) is statistically significant ($p = 0.032$, 95% CI $[+0.0093, +0.1759]$).
   - PR-AUC improvement (0.0258 $\to$ 0.0338, $+31.0\%$) was observed with $p = 0.063$ (not statistically significant at the strict $\alpha = 0.05$ threshold).
3. **Stochastic Nature**: Track failure prediction remains probabilistic; Candidate V3 does not predict failure events deterministically.
4. **Prevalence Shift & Current Threshold**: Candidate V3 was calibrated on the true ~1.14% failure base rate. At the legacy threshold $\tau = 0.0120$, approximately 43%–45% of the fleet is flagged for 30-day inspection.

---

## 10. Post-Promotion Validation Items

1. **Operational Threshold Alignment**:
   - Investigate whether the operational threshold should be formally calibrated to $\tau = 0.0200$ (which flags ~14.6% of the fleet, capturing 33.3% of test failures), or if operations should transition to percentile ranking (Top-5% or Top-10% worklist dispatch).
2. **Field Data Ingestion**:
   - As real maintenance and failure logs are collected from physical divisions, establish an ingestion pipeline to benchmark candidate models on real ground truth.

---

PROMOTION RESULT
----------------
Previous Production Model:
models/calibrated_xgboost_v2_legacy.pkl

Current Production Model:
models/calibrated_xgboost.pkl

Candidate V3 Artifact Preserved:
YES

Legacy Backup Verified:
PASS

New Production Hash Matches Candidate:
PASS

Inference Smoke Test:
PASS

Regression Tests:
PASS

Threshold:
UNCHANGED

Priority Logic:
UNCHANGED

Scheduling Logic:
UNCHANGED

API:
NOT IMPLEMENTED

Backend:
UNCHANGED

Frontend:
UNCHANGED

Node Worker:
UNCHANGED

Rollback Artifact:
VALID

FINAL STATUS:
PROMOTED
