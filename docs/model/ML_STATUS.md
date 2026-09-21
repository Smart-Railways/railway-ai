# Railway-AI Component Status & Dependency Audit (Phase 1)

**Source of Truth Document**
**Date of Audit**: 2026-09-05
**Auditor**: Railway-AI Engineering Audit Team
**Guiding Principle**: *Never confuse "implemented" with "validated".*

---

## 1. Component Maturity Matrix

We categorize every component across the five strict lifecycle stages:
1. **IMPLEMENTED**: Code/logic exists in a notebook or script.
2. **TESTED**: Formal unit or integration tests pass in `tests/`.
3. **REAL_DATA_TESTED**: Verified against genuine RailKit or field operational data.
4. **VALIDATED**: Statistically evaluated with measured metrics on production-scale real data.
5. **PRODUCTION_READY**: Ready for deployment and backend/frontend consumption.

| Component | Architecture / Model | Tests Passing | Real Data Connected | Current Stage | Production Candidate? |
|---|---|:---:|:---:|---|:---:|
| **Asset Feature Builder** | Handcrafted aggregations | ✅ | ⏳ (Awaiting asset logs) | **TESTED** | ⏳ |
| **Failure Classification** | Calibrated XGBoost | ✅ | ❌ (7 real failure logs only) | **TESTED** | ⏳ |
| **RUL Estimation** | Gradient Boosting Regressor | ✅ | ❌ (Need longitudinal curves) | **TESTED** | ⏳ |
| **Survival Analysis** | Cox Proportional Hazards | ✅ | ❌ (Need asset lifespans) | **TESTED** | ⏳ |
| **Maintenance Duration** | Heuristic + Regressor | ✅ | ❌ (Need actual durations) | **TESTED** | ⏳ |
| **RailKit Ingestion Engine** | REST Client + Storage + Adapter | ✅ | ✅ (8 captures, 5 trains) | **REAL_DATA_TESTED** | ✅ |
| **Section Evidence Engine** | `SectionEvidenceBuilder` | ✅ | ✅ (37 records, 11 sections) | **REAL_DATA_TESTED** | ✅ |
| **Operational Pressure Engine**| `RailKitSectionPressureBuilder`| ✅ | ✅ (11 corridor sections) | **REAL_DATA_TESTED** | ✅ |
| **Decision & Priority Engine** | `MaintenanceDecisionEngine` | ✅ | ✅ (Enriched by real pressure)| **REAL_DATA_TESTED** | ✅ |
| **6-Hour Block Optimizer** | CP-SAT (`BlockOptimizer`) | ✅ | ✅ (Scheduled real worklists) | **REAL_DATA_TESTED** | ✅ |
| **Multi-Horizon Optimization** | CP-SAT (`MultiHorizonPlanner`)| ✅ | ✅ (Weekly, Monthly, Dynamic) | **TESTED** | ✅ |
| **Anomaly Detection** | Isolation Forest + PyTorch Autoencoder | ✅ | ⏳ (Trained on simulated history) | **TESTED** | ⏳ |
| **Deep Learning Candidates** | LSTM, 1D-CNN, Transformer | ✅ | ❌ (Research layer) | **TESTED** | ❌ (Research) |
| **Evidence Fusion Layer** | Multi-model confidence join | ✅ | ⏳ (Includes real pressure) | **TESTED** | ✅ |
| **Empirical Benchmark Suite** | `BenchmarkValidator` | ✅ | ✅ (Proved > FIFO baseline) | **TESTED** | ✅ |
| **Unified Service Facade** | `RailwayMLEngine` | ✅ | ✅ (`health`, `predict`, `plan`) | **REAL_DATA_TESTED** | ✅ |

---

## 2. Synthetic Data Dependency Map

Synthetic data in `data/processed/` serves as our **permanent regression and stress-testing layer**:
- `failure_ml_dataset.csv`, `failure_train.csv`: Preserves failure classification training checkpoints.
- `asset_history_simulated.csv`: Preserves anomaly detection and autoencoder checkpoints.
- `survival_dataset.csv`: Preserves Cox PH survival curves.
- `multi_objective_decision_scores.csv`: Preserves offline priority engine benchmarks.

**Separation Rule**:
Synthetic data is never mixed with or claimed as real data. Real train tracking observations reside exclusively in `data/raw_real/railkit/` and `data/processed_real/`.

---

## 3. Real Data Coverage & Gaps

### What Is Verified Real:
- **Corridor Topology**: New Delhi → Mumbai (11 sections: `NDL-MTJ-01` to `SRT-MUM-01`).
- **Station Mappings Resolved**:
  - `JHS` → `VGLJ` (Virangana Lakshmibai Jhansi Jn)
  - `VAD` → `BRC` (Vadodara / Baroda)
  - `SRT` → `ST` (Surat)
  - `MUM` → `MMCT` (Mumbai Central)
- **Live Train Tracking**: Validated captures for pilot trains `12002`, `20164`, `22946`.
- **Operational Pressure**: Real delay intensity and route load calculated across all 11 sections.

### Major Real Data Blockers:
- Real asset metadata, longitudinal inspection records, and failure defect logs have not yet been provided.
- Consequently, supervised failure/RUL models remain trained on verified synthetic baselines until real asset databases are imported.

---

## 4. Test Verification Benchmark
Total active test cases in repository: **69 tests**
Execution status: **69/69 PASSED (0 failures, 0 regressions)**
