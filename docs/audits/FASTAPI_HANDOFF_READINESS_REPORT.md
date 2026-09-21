# FastAPI Service Handoff-Readiness Audit Report

**Date**: 2026-09-21
**Target Repository**: `Smart-Railways/railway-ai`
**Integration Boundary**: `Smart-Railways/node-worker` and backend services
**Phase**: Final Handoff Preparation & Verification
**Status**: COMPLETE

---

## 1. Executive Summary

This report documents the finalization of the Railway-AI ML FastAPI HTTP service for handoff to the Node Worker and backend integration teams. All known configuration and documentation warnings from the previous readiness audit have been resolved:
1. An explicit entrypoint runner (`python -m src.api.main`) has been added to honor `ML_SERVICE_HOST` and `ML_SERVICE_PORT` cleanly, alongside standard `uvicorn` invocation.
2. `docs/integration/ML_API.md` was updated to accurately document `recommendations_df` as returned by the live `POST /api/v1/recommend-windows` endpoint.
3. The formal handoff specification [`docs/integration/ML_API_HANDOFF.md`](file:///wsl$/Ubuntu/home/chirag/projects/railway-ai/docs/integration/ML_API_HANDOFF.md) was created to provide a complete, frozen integration reference.
4. Additional regression tests were added to `tests/api/test_api.py`, expanding the API test suite to 17 tests (total test suite: 177 passed, 0 failed, 0 skipped).
5. Live HTTP contract validation confirmed complete adherence to `schemas/ml_response.schema.json`.

---

## 2. What Was Changed

The following files were updated or created in this finalization phase:

| File | Change Category | Description |
|------|-----------------|-------------|
| `src/api/main.py` | Configuration | Added `if __name__ == "__main__":` entrypoint runner reading `ML_SERVICE_HOST` (default `0.0.0.0`) and `ML_SERVICE_PORT` (default `8000`). |
| `.env.example` | Configuration | Clarified comments for host, port, and CORS settings with zero secrets. |
| `docs/integration/ML_API.md` | Documentation | Updated startup instructions and added `recommendations_df` to the `/recommend-windows` example response. |
| `docs/integration/ML_API_HANDOFF.md` | Documentation | Created comprehensive integration handoff specification for other engineering teams. |
| `tests/api/test_api.py` | Regression Tests | Added tests for `recommendations_df`, capacity-exceeded infeasible tasks, recommend-windows HITL invariants, 500 error sanitization, and default CORS protection (17 tests total). |
| `docs/audits/FASTAPI_HANDOFF_READINESS_REPORT.md` | Audit Documentation | Created this final handoff audit report. |

---

## 3. What Was Intentionally NOT Changed

To protect the validated ML boundary and system stability, the following were strictly preserved:
- **Production Model Artifact**: Unchanged (`models/production/calibrated_xgboost.pkl`).
- **Legacy Model Artifact**: Unchanged (`models/legacy/calibrated_xgboost_v2_legacy.pkl`).
- **Production Threshold**: Maintained strictly at `0.0120`.
- **ML & Decision Logic**: `RailwayMLEngine`, `FailureRiskPredictor`, `MaintenanceDecisionEngine`, and `BlockOptimizer` logic remains 100% frozen.
- **Contract Semantics**: 4 frozen output categories (`risk`, `priority`, `explanation`, `scheduling`) and JSON schema constraints (`schemas/ml_response.schema.json`) remain strictly untouched.
- **External Repositories**: `Smart-Railways/node-worker` and `backend/` were NOT modified.
- **Git State**: No commits or pushes performed.

---

## 4. Endpoints List & Status

| Endpoint | Method | Status | Contract Reference | Description |
|----------|:------:|:------:|--------------------|-------------|
| `/health` | `GET` | ✅ READY | `HealthResponse` | Reports process health, component readiness, and dynamic model availability (`model_available: true`). |
| `/api/v1/evaluate` | `POST` | ✅ READY | `schemas/ml_response.schema.json` | Full batch evaluation: 30-day failure risk, 5-factor priority score, reason tags, and Top-3 CP-SAT block windows. |
| `/api/v1/recommend-windows` | `POST` | ✅ READY | `RecommendWindowsRequest` / `dict` | Constrained CP-SAT scheduling returning Top-3 recommended windows, `recommendations_df`, and infeasible task list. |

---

## 5. Configuration & Startup Status

### Startup Methods
1. **Via Python Module (Honors `.env` / Environment Variables)**:
   ```bash
   PYTHONPATH=. .venv/bin/python -m src.api.main
   ```
   Binds to `ML_SERVICE_HOST` (default `0.0.0.0`) and `ML_SERVICE_PORT` (default `8000`).

2. **Via Uvicorn CLI**:
   ```bash
   PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

### Security Settings
- **CORS**: Disabled by default. Enabled only when `ML_CORS_ORIGINS` is configured with non-empty, comma-separated origins. Wildcard `*` is never enabled.
- **Error Sanitization**: Handlers for 422, 500, and 503 suppress Python tracebacks and internal filesystem paths.

---

## 6. Test Suite & Validation Results

### Test Execution Summary
- **Existing ML / Domain Tests**: 160 passed
- **API Test Suite (`tests/api/test_api.py`)**: 17 passed
- **Combined Test Total**: **177 passed / 0 failed / 0 skipped** (Execution time: 10.71s)

### API Test Coverage Breakdown
1. `TestHealthEndpoint`:
   - `test_health_returns_200`: Status 200, `model_available: true`, components map.
2. `TestEvaluateEndpoint`:
   - `test_evaluate_valid_request`: Valid multi-task evaluation payload.
   - `test_evaluate_response_has_four_categories`: `risk`, `priority`, `explanation`, `scheduling`.
   - `test_evaluate_risk_fields`: `failure_probability` in `[0, 1]`, `predicted_failure_30d` bool.
   - `test_evaluate_priority_fields`: `priority_score` in `[0, 1]`, valid enum category, 5-factor breakdown.
   - `test_evaluate_scheduling_top3_cap`: Max 3 recommendations per task.
   - `test_evaluate_hitl_invariants`: `selection_status="RECOMMENDED"`, `human_confirmation_required=true`, `is_booked=false`, `booking_status="UNBOOKED"`.
3. `TestInputValidation`:
   - `test_evaluate_invalid_request_422`: Rejection of missing required fields.
   - `test_evaluate_empty_tasks_422`: Rejection of empty tasks array (`min_length=1`).
4. `TestRecommendWindowsEndpoint`:
   - `test_recommend_windows_valid`: 200 response with `recommendations`, `recommendations_df`, `infeasible_tasks`, `metrics`.
   - `test_recommend_windows_top3_cap`: Capped at Top-3 windows.
   - `test_recommend_windows_infeasible_task`: 0 windows returned when task exceeds capacity.
   - `test_recommend_windows_hitl_invariants`: Preserves advisory HITL flags.
5. `TestSchemaValidation`:
   - `test_response_matches_json_schema`: Validated against `schemas/ml_response.schema.json`.
6. `TestControlledErrors`:
   - `test_model_unavailable_503`: 503 returned when model is unloaded.
   - `test_internal_error_500_suppresses_traceback`: 500 error sanitizes error messages.
7. `TestConfigurationSafety`:
   - `test_cors_disabled_by_default`: Preflight does not emit ACAO headers.

---

## 7. Model Hash Integrity

| Model | File Path | Verified SHA256 Hash | Status |
|-------|-----------|----------------------|--------|
| **Production Model** | `models/production/calibrated_xgboost.pkl` | `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` | ✅ EXACT MATCH |
| **Legacy Model** | `models/legacy/calibrated_xgboost_v2_legacy.pkl` | `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386` | ✅ EXACT MATCH |

---

## 8. Known Limitations & Notes for Consumer Teams

1. **Synthetic Telemetry Baseline**: The model was trained and calibrated on synthetic corridor failure distributions representing Northern Railway operations. Feature distributions and calibration curves reflect this dataset.
2. **Advisory Semantics**: The scheduling optimizer computes mathematically feasible options based on static corridor block configurations; it does not interface with live signalling interlocking. Human confirmation is mandatory before track possession booking.
3. **Synchronous Threadpool Execution**: FastAPI executes endpoint functions in a worker threadpool. This is optimal for low-to-medium throughput local service architectures. If scaling beyond 100 concurrent requests/sec in the future, multiple uvicorn worker processes (`--workers N`) should be deployed.

---

## 9. External Repository Verification

- `Smart-Railways/node-worker`: **UNTOUCHED**. No files staged or modified.
- `backend/`: **UNTOUCHED**.
- Zero commits or pushes performed.

---

## 10. Conclusion & Final Verdict

The Railway-AI FastAPI service is fully validated, cleanly configured, and frozen for integration handoff.

# RAILWAY-AI API HANDOFF STATUS: PASS
