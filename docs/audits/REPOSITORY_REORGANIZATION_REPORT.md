# Railway-AI Repository Reorganization & Cleanup Report

**Date**: September 21, 2026
**Environment**: Local WSL2 Ubuntu Linux (`/home/chirag/projects/railway-ai`)
**Scope**: Full ML Repository Structural Organization, Artifact Layout, Test Categorization, Script Promotion, and Documentation Restructuring.

---

## 1. Executive Summary

Following the controlled promotion of Candidate V3 to active production, the entire Railway-AI ML repository was audited, reorganized, and cleaned up without modifying ML behavior, retraining models, tuning hyperparameters, changing operating thresholds ($\tau = 0.0120$ strictly preserved), or modifying API boundaries.

All 160 automated unit, integration, optimization, lifecycle, and contract tests pass with zero regressions.

---

## 2. Structural Transformation

### 2.1 Before Structure (Summary)
- Model artifacts were mixed in the flat root of `models/` alongside deep learning checkpoints, config JSONs, and candidate experiment folders.
- 10 test files were placed flat in `tests/`.
- Temporary research, audit, and experimentation scripts accumulated in `scratch/` (45 files, ~200 KB).
- Audit reports were split across an ad-hoc root `audit/` folder and `docs/`.
- Schemas were nested only under `docs/schemas/`.
- Reusable evaluation, data generation, and verification tools were mixed with ephemeral scratch files.

### 2.2 Final Target Structure
```
railway-ai/
│
├── src/                                  # Production ML & Planning Source Code
│   ├── models/                           # Calibrated failure risk prediction
│   │   └── failure_predictor.py
│   ├── decision/                         # Multi-objective decision scoring (5 factors)
│   │   └── maintenance_decision_engine.py
│   ├── optimization/                     # CP-SAT solver & window optimization
│   │   ├── block_optimizer.py
│   │   └── multi_horizon_planner.py
│   ├── planning/                         # Planning orchestration service
│   │   └── planning_service.py
│   ├── lifecycle/                        # Real-time maintenance lifecycle state machine
│   │   ├── clock.py, manager.py, models.py
│   ├── services/                         # Unified engine entrypoint & corridor planning
│   │   ├── ml_engine.py, planning_service.py, real_corridor_planning.py
│   ├── features/                         # Live operational pressure & corridor enrichment
│   │   ├── section_pressure_builder.py, etc.
│   ├── evaluation/                       # Benchmark validator
│   │   └── benchmark_validator.py
│   └── data/                             # RailKit adapters & telemetry extractors
│
├── tests/                                # 160 Automated Pytest Tests (100% Green)
│   ├── model/                            # test_failure_predictor.py, test_candidate_v3_model.py, test_benchmark_validator.py
│   ├── decision/                         # test_planning_regression.py
│   ├── scheduling/                       # test_top3_recommendations.py, test_multi_horizon_and_engine.py
│   ├── lifecycle/                        # test_maintenance_lifecycle.py
│   ├── contracts/                        # test_backend_contract.py
│   ├── data/                             # test_section_evidence.py
│   └── integration/                      # test_ml_engine.py
│
├── scripts/                              # Organized Operational & Research Utilities
│   ├── training/                         # Model training pipelines (train_candidate_v3.py)
│   ├── evaluation/                       # Candidate evaluation & bootstrap audit scripts
│   ├── validation/                       # Feature leakage & inference smoke tests
│   ├── data/                             # Target generation & API diagnostic utilities
│   └── maintenance/                      # Rollback runbooks (rollback_to_v2.py)
│
├── models/                               # Structured Model Artifact Repository
│   ├── production/                       # Active Production: calibrated_xgboost.pkl (Candidate V3)
│   ├── legacy/                           # Archived Legacy: calibrated_xgboost_v2_legacy.pkl
│   ├── candidate/                        # Evaluated candidates & manifests (candidate_v3/)
│   └── experimental/                     # Deep learning, survival & exploratory checkpoints
│
├── data/                                 # Partitioned Data Tiers
│   ├── raw/                              # Original raw synthetic datasets
│   ├── processed/                        # Cleaned synthetic datasets & candidate_v3/
│   ├── raw_real/                         # Live RailKit JSON captures
│   ├── processed_real/                   # Corridor section features & mapping evidence
│   └── predictions/                      # Batch inference outputs & visual artifacts
│
├── config/                               # Production Manifests & Contracts
│   ├── corridor_master.json, data_source_contract.json, ml_contract.json,
│   ├── model_manifest.json, pipeline_manifest.json, railkit_section_mapping.json
│
├── schemas/                              # Machine-Readable JSON Schemas
│   └── ml_response.schema.json           # Frozen 4-category backend contract schema
│
├── docs/                                 # Organized Documentation Hierarchy
│   ├── architecture/                     # System design & component interactions
│   ├── model/                            # Model improvement reports & maturity logs
│   ├── integration/                      # Backend integration contracts & defense guides
│   └── audits/                           # Formal audit, promotion & reorganization reports
│
├── notebooks/                            # Exploratory R&D notebooks (checkpoints cleaned)
├── README.md                             # Project overview & navigation tree
├── AGENT_STATE.md                        # Lifecycle milestone tracking matrix
├── requirements.txt                      # Project dependencies
└── .gitignore                            # Git exclusion rules
```

---

## 3. Detailed File Movements & Reorganization

### 3.1 Model Artifacts (`models/`)
| Original Path | New Path | Description | SHA256 Status |
| :--- | :--- | :--- | :---: |
| `models/calibrated_xgboost.pkl` | `models/production/calibrated_xgboost.pkl` | Active Production Model (V3) | `e021f6b5...` (**PASS**) |
| `models/calibrated_xgboost_v2_legacy.pkl` | `models/legacy/calibrated_xgboost_v2_legacy.pkl` | Archived Legacy Baseline (V2) | `3130e9a1...` (**PASS**) |
| `models/candidate_v3/` | `models/candidate/candidate_v3/` | Candidate V3 artifacts & manifest | Preserved |
| `models/best_cnn_failure_model.pt` | `models/experimental/best_cnn_failure_model.pt` | 1D-CNN experimental model | Preserved |
| `models/best_lstm_failure_model.pt` | `models/experimental/best_lstm_failure_model.pt` | LSTM sequence model | Preserved |
| `models/best_railway_autoencoder.pt` | `models/experimental/best_railway_autoencoder.pt` | Autoencoder model | Preserved |
| `models/best_railway_transformer.pt` | `models/experimental/best_railway_transformer.pt` | Transformer model | Preserved |
| `models/railway_transformer_final.pt` | `models/experimental/railway_transformer_final.pt` | Final transformer checkpoint | Preserved |
| `models/cox_survival_model.pkl` | `models/experimental/cox_survival_model.pkl` | Cox PH survival model | Preserved |
| `models/*.json` | `models/experimental/*.json` | Configs for deep learning models | Preserved |

### 3.2 Tests Reorganization (`tests/`)
| Original Test File | New Category Path | Focus Area | Tests |
| :--- | :--- | :--- | :---: |
| `tests/test_failure_predictor.py` | `tests/model/test_failure_predictor.py` | Model loading & risk inference | 6 |
| `tests/test_candidate_v3_model.py` | `tests/model/test_candidate_v3_model.py` | Candidate V3 contract & bounds | 6 |
| `tests/test_benchmark_validator.py` | `tests/model/test_benchmark_validator.py` | AI vs uncoordinated baseline | 2 |
| `tests/test_planning_regression.py` | `tests/decision/test_planning_regression.py` | 5-factor priority formula | 23 |
| `tests/test_top3_recommendations.py` | `tests/scheduling/test_top3_recommendations.py` | CP-SAT top-3 recommendations | 16 |
| `tests/test_multi_horizon_and_engine.py` | `tests/scheduling/test_multi_horizon_and_engine.py` | Multi-horizon & dynamic scheduling | 26 |
| `tests/test_maintenance_lifecycle.py` | `tests/lifecycle/test_maintenance_lifecycle.py` | State machine & human confirmation | 24 |
| `tests/test_backend_contract.py` | `tests/contracts/test_backend_contract.py` | Backend JSON schema & categories | 28 |
| `tests/test_section_evidence.py` | `tests/data/test_section_evidence.py` | Corridor station mapping evidence | 12 |
| `tests/test_ml_engine.py` | `tests/integration/test_ml_engine.py` | End-to-end RailwayMLEngine evaluation | 17 |
| **Total Test Suite** | | | **160 PASSED** |

### 3.3 Reusable Scripts Promoted (`scripts/`)
| Original Location | Promoted Path | Responsibility / Ongoing Value |
| :--- | :--- | :--- |
| `scripts/train_candidate_v3.py` | `scripts/training/train_candidate_v3.py` | Reproducible Candidate V3 training pipeline. |
| `scratch/generate_candidate_v3_data.py` | `scripts/data/generate_candidate_v3_data.py` | Generates corrected calendar monthly target dataset. |
| `scratch/verify_leakage.py` | `scripts/validation/verify_feature_leakage.py` | Rigorous feature temporal leakage audit utility. |
| `scratch/independent_v3_audit.py` | `scripts/evaluation/independent_model_audit.py` | 1,000-resample paired bootstrap evaluation tool. |
| `scratch/evaluate_final_candidates.py` | `scripts/evaluation/evaluate_model_candidates.py` | Benchmarks candidate models on test set. |
| `scratch/derive_bayes_bound.py` | `scripts/evaluation/derive_bayes_bound.py` | Derives mathematical limits of synthetic generator. |
| `scratch/step6_inference_smoke_test.py` | `scripts/validation/inference_smoke_test.py` | 5-case production inference gradient check. |
| `src/data/test_railkit.py` | `scripts/data/test_railkit_api.py` | Live RailKit API connectivity diagnostic. |
| *New* | `scripts/maintenance/rollback_to_v2.py` | Instant, verified rollback utility to legacy V2 model. |

### 3.4 Documentation Consolidation (`docs/`)
| Original Path | New Path | Description |
| :--- | :--- | :--- |
| `docs/CANDIDATE_V3_FINAL_AUDIT.md` | `docs/audits/CANDIDATE_V3_FINAL_AUDIT.md` | Independent 1,000-resample audit report. |
| `docs/CANDIDATE_V3_PRODUCTION_PROMOTION.md` | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` | Official promotion verification record. |
| `docs/CANDIDATE_V3_PROMOTION_READINESS.md` | `docs/audits/CANDIDATE_V3_PROMOTION_READINESS.md` | Promotion readiness assessment. |
| `audit/v2_phase1_cleanup.md` | `docs/audits/v2_phase1_cleanup.md` | Phase 1 model audit & weather noise cleanup. |
| `audit/v2_phase2_top3_windows.md` | `docs/audits/v2_phase2_top3_windows.md` | Phase 2 top-3 recommendation engine audit. |
| `audit/v2_model_comparison.json` | `docs/audits/v2_model_comparison.json` | Model comparison benchmark data. |
| `docs/ML_BACKEND_INTEGRATION_CONTRACT.md` | `docs/integration/ML_BACKEND_INTEGRATION_CONTRACT.md` | Frozen 4-category integration specification. |
| `docs/ML_EXAMINER_AND_JUDGE_DEFENSE_GUIDE.md` | `docs/integration/ML_EXAMINER_AND_JUDGE_DEFENSE_GUIDE.md` | Comprehensive viva & defense guide. |
| `docs/ML_STATUS.md` | `docs/model/ML_STATUS.md` | Component maturity and validation tracker. |
| `docs/MODEL_IMPROVEMENT_REPORT.md` | `docs/model/MODEL_IMPROVEMENT_REPORT.md` | Target bug discovery & optimization log. |
| `docs/schemas/ml_response.schema.json` | `schemas/ml_response.schema.json` | Canonical schema location (preserved alias in docs/). |

---

## 4. Deleted Files Justification

Every deleted file was audited against repository references, import chains, test dependencies, and permanent documentation records. Zero production or reproducibility assets were deleted.

| File Path | Category | Reason for Deletion | Permanent Record / Replacement |
| :--- | :--- | :--- | :--- |
| `src/data/railkit_client.py.save` | Editor Backup | 1-byte nano editor temporary backup. | `src/data/railkit_client.py` |
| `notebooks/.ipynb_checkpoints/*` | Jupyter Cache | 29 auto-save checkpoint files. | Root notebooks in `notebooks/` |
| `scratch/prompt10.txt` | Prompt Copy | Raw text of previous user request. | Agent conversation transcript |
| `scratch/read_prompt.py` | Ephemeral Script | Printed prompt text from transcript. | N/A |
| `scratch/read_nb05.py` | Ephemeral Script | Inspect notebook 05 JSON. | `notebooks/05_generate_historical_data.ipynb` |
| `scratch/read_nb05_all.py` | Ephemeral Script | Inspect notebook 05 JSON. | `notebooks/05_generate_historical_data.ipynb` |
| `scratch/read_nb06.py` | Ephemeral Script | Inspect notebook 06 JSON. | `notebooks/06_time_split.ipynb` |
| `scratch/read_nb06_split.py` | Ephemeral Script | Inspect notebook 06 JSON. | `notebooks/06_time_split.ipynb` |
| `scratch/inspect_data.py` | Ephemeral Script | Printed CSV shapes during Phase 0 discovery. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/inspect_model.py` | Ephemeral Script | Checked pickle serialization format. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/test_models.py` | Experiment Script | Initial uncalibrated baseline test. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/test_signal.py` | Experiment Script | Feature correlation checks. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/test_signal_fast.py` | Experiment Script | Fast correlation checks. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/test_window.py` | Experiment Script | Checked timestamp offsets. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/check_sim_prob.py` | Experiment Script | Latent probability check. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/check_splits_clean.py` | Experiment Script | Data split integrity check. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/dropped_distribution.py` | Experiment Script | Analyzed dropped 31-day failures. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/detailed_target_audit.py` | Experiment Script | Target defect audit script. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/target_investigation.py` | Experiment Script | Target defect investigation. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/compare_distributions.py` | Experiment Script | Train vs test distribution comparison. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/feat_eng.py` | Experiment Script | Feature engineering trial script. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/run_experiments.py` | Experiment Script | Intermediate candidate exploration. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/candidate_v3_experiments.py` | Experiment Script | Optuna hyperparameter sweep. | Best params in `scripts/training/train_candidate_v3.py` |
| `scratch/oracle_eval.py` | Experiment Script | Synthetic oracle evaluation. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/threshold_scan.py` | Experiment Script | Threshold exploration scan. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/eval_baseline.py` | Experiment Script | Initial baseline evaluation. | `docs/model/MODEL_IMPROVEMENT_REPORT.md` |
| `scratch/verify_everything.py` | Ephemeral Script | Verification wrapper script. | `docs/audits/CANDIDATE_V3_FINAL_AUDIT.md` |
| `scratch/check_promotion_readiness.py` | Ephemeral Script | Pre-audit readiness check. | `docs/audits/CANDIDATE_V3_PROMOTION_READINESS.md` |
| `scratch/inspect_candidate_v3.py` | Ephemeral Script | Artifact inspection. | `docs/audits/CANDIDATE_V3_FINAL_AUDIT.md` |
| `scratch/pre_promotion_check.py` | Promotion One-off | Checked hashes prior to promotion. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step2_backup.py` | Promotion One-off | Backed up legacy model. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step3_promote.py` | Promotion One-off | Promoted Candidate V3. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step5_verify_production_loading.py` | Promotion One-off | Verified production loading. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step8_verify_non_ml.py` | Promotion One-off | Verified non-ML formulas. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step9_verify_hashes.py` | Promotion One-off | Verified 3 artifact hashes. | `docs/audits/CANDIDATE_V3_PRODUCTION_PROMOTION.md` |
| `scratch/step10_verify_rollback_artifact.py` | Promotion One-off | Verified legacy loadability. | Converted to `scripts/maintenance/rollback_to_v2.py` |
| `scratch/generate_inventory.py` | Inventory Tool | File inventory script. | N/A (Temporary) |
| `scratch/summarize_inventory.py` | Inventory Tool | Inventory summarizer. | N/A (Temporary) |
| `scratch/detail_inventory.py` | Inventory Tool | Inventory detail printer. | N/A (Temporary) |
| `scratch/full_file_inventory.txt` | Inventory Tool | Output inventory dump. | N/A (Temporary) |
| `scratch/inspect_scratch_all.py` | Inventory Tool | Scratch inspector. | N/A (Temporary) |

---

## 5. Review Required Files

The following files were inspected and preserved under `models/candidate/candidate_v3/` for audit provenance rather than deleted:
- `models/candidate/candidate_v3/baseline/xgb_baseline_calibrated.pkl`
- `models/candidate/candidate_v3/baseline/xgb_baseline_uncalibrated.pkl`
- `models/candidate/candidate_v3/calibrated/calibrated_xgboost_v3.pkl` (Intermediate calibration checkpoint, size 1,749,788 bytes, SHA256 `0f2fafa1...`)
- `models/candidate/candidate_v3/tuned/xgb_tuned_uncalibrated.pkl`
- `models/candidate/candidate_v3/v3_validation_results.json` (Full Optuna & baseline evaluation JSON)

---

## 6. Model Integrity & Regression Verification

### 6.1 Model Integrity Check
```bash
python scripts/validation/final_verification.py
```
- **Active Production Model**: `models/production/calibrated_xgboost.pkl`
  - SHA256: `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` (**PASS**)
- **Archived Legacy Model**: `models/legacy/calibrated_xgboost_v2_legacy.pkl`
  - SHA256: `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386` (**PASS**)
- **Candidate V3 Copy**: `models/candidate/candidate_v3/calibrated_xgboost_v3.pkl`
  - SHA256: `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` (**PASS**)

### 6.2 Production Inference Smoke Test
```bash
python scripts/validation/inference_smoke_test.py
```
- **AST-LOW-01** (Low Risk): $p = 0.00570$, flag = 0, Decision Rank = 5, Score = 0.0817 (LOW)
- **AST-MED-02** (Medium Risk): $p = 0.01140$, flag = 0, Decision Rank = 4, Score = 0.2284 (LOW)
- **AST-HIGH-03** (High Risk): $p = 0.16410$, flag = 1, Decision Rank = 1, Score = 0.4792 (MEDIUM)
- **AST-CRIT-04** (High Criticality): $p = 0.03400$, flag = 1, Decision Rank = 2, Score = 0.4252 (MEDIUM)
- **AST-HIST-05** (High History): $p = 0.07720$, flag = 1, Decision Rank = 3, Score = 0.4032 (MEDIUM)
- **Gradient**: $p_{high} (0.16410) > p_{med} (0.01140) > p_{low} (0.00570)$ (**PASS**)

### 6.3 Automated Test Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```
- **Passed**: 160
- **Failed**: 0
- **Skipped**: 0
- **Execution Time**: 10.29s
- **Result**: **PASS**

---

DELETED FILES
-------------
41 files verified obsolete, temporary, or redundant were removed (all documented with justifications in Section 4).
The `scratch/` directory and `notebooks/.ipynb_checkpoints/` have been completely removed.

MODEL INTEGRITY
---------------
Production V3 hash: e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae
Legacy V2 hash:     3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386
Verification: PASS

REGRESSION
----------
Tests: 160 passed / 0 failed / 0 skipped in 10.29s
Result: PASS
