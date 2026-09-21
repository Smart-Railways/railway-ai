# FastAPI Service Integration-Readiness Audit Report

**Date**: 2026-09-21
**Target Repository**: `Smart-Railways/railway-ai`
**Consumer Target**: `Smart-Railways/node-worker` (integration boundary)
**Phase**: FastAPI Service Integration-Readiness Validation
**Status**: COMPLETE

---

## Executive Summary

This validation rigorously audits the HTTP boundary of the Railway-AI FastAPI service prior to connecting `Smart-Railways/node-worker`. Across 16 evaluation categories covering contract adherence, schema conformance, scheduling invariants, error handling, security, concurrency, documentation, and consumer simulation, the service meets all production integration criteria.

- **Overall Result**: **PASS** (14 PASS, 2 WARNING, 0 FAIL)
- **Zero ML / Scheduling Modification**: Production model, weights, thresholds, priority formulas, and CP-SAT logic remain 100% frozen and unmodified.
- **Contract Adherence**: HTTP responses strictly validate against `schemas/ml_response.schema.json`.
- **Node Consumer Interoperability**: Live Node.js v22 simulation passed all 34 end-to-end checks with zero errors.
- **Node Worker Repository**: Untouched. Zero changes committed or staged in `Smart-Railways/node-worker`.

---

## Detailed Evaluation by Category

### A. API Implementation Review
- **Status**: **PASS**
- **Analysis**:
  - All 12 files in `src/api/*`, `tests/api/*`, and root configuration were audited.
  - The API layer acts strictly as an HTTP transport adapter. Zero ML prediction, model loading, priority calculation, or CP-SAT scheduling logic is duplicated in `src/api/`.
  - The singleton pattern in `src/api/dependencies.py` wraps `RailwayMLEngine` cleanly without extraneous layers.
  - Route handlers delegate directly to `RailwayMLEngine.evaluate_for_backend(...)` and `recommend_windows(...)`.
  - No heuristic confidence scores, invented fields, or unauthorized parameters exist.

---

### B. Request Validation
- **Status**: **PASS**
- **Analysis**:
  - Malformed HTTP requests were tested against `POST /api/v1/evaluate`:
    1. **Empty body**: Correctly rejected with HTTP 422 (`Field required: tasks`).
    2. **Missing required fields** (e.g. omitted `task_id`): Correctly rejected with HTTP 422.
    3. **Wrong field types** (e.g. string for integer duration): Correctly rejected with HTTP 422.
    4. **Invalid numeric values** (e.g. negative duration, negative manpower): Enforced by Pydantic bounds (`ge=1`), correctly rejected with HTTP 422.
    5. **Empty task list** (`"tasks": []`): Enforced by `min_length=1`, correctly rejected with HTTP 422.
    6. **Invalid task structure** (non-dictionary elements): Correctly rejected with HTTP 422.
    7. **Extra unexpected fields**: Accepted cleanly and ignored by Pydantic default parsing without side effects.
    8. **Null values where prohibited**: Correctly rejected with HTTP 422.
  - All client validation errors return structured JSON with field locations and descriptions, leaking no internal file paths or stack traces.

---

### C. Response Contract
- **Status**: **PASS**
- **Analysis**:
  - Successful responses from `POST /api/v1/evaluate` were validated directly against the canonical `schemas/ml_response.schema.json` using `jsonschema.validate()`.
  - Multi-task, single-task, and minimal-input responses all validated with 0 schema violations.
  - Verified structure:
    - **Top-level**: `status` ("SUCCESS"), `engine_version` ("0.1.0"), `tasks_evaluated` (integer >= 0), `results` (array), `metrics` (object).
    - **Result item**: `task_id`, `section_id`, `department`, `decision_rank`, `risk`, `priority`, `explanation`, `scheduling`.
    - **Risk**: `failure_probability` (float in `[0, 1]`), `predicted_failure_30d` (boolean). No invented fields (`risk_category` or confidence scores are absent).
    - **Priority**: `priority_score` (float in `[0, 1]`), `priority_category` (enum: LOW, MEDIUM, HIGH, CRITICAL), `score_breakdown` (5 factors + total).
    - **Explanation**: `reason_tags` (list of strings), `explanation_summary` (non-empty string).
    - **Scheduling**: `recommended_windows` (list of up to 3 objects), `total_recommendations` (integer 0-3), `has_feasible_window` (boolean).

---

### D. Scheduling Contract
- **Status**: **PASS**
- **Analysis**:
  - Tested `POST /api/v1/recommend-windows` across diverse constraint scenarios:
    - **0 feasible windows**: When task duration or manpower exceeds block capacities, returns exactly 0 recommendations (`[]`) and flags the task in `infeasible_tasks`.
    - **1 feasible window**: Returns exactly 1 window when only 1 is feasible.
    - **2 feasible windows**: Returns exactly 2 windows when 2 are feasible.
    - **3+ feasible windows**: When 5 windows are available within the planning horizon, returns strictly capped Top-3 recommendations.
  - **No fabricated windows**: Every recommended window corresponds to a genuine input block window identifier.
  - **HITL Invariants Preserved**: Every recommended window strictly maintains:
    - `selection_status = "RECOMMENDED"`
    - `human_confirmation_required = true`
    - `is_booked = false`
    - `booking_status = "UNBOOKED"`
  - CP-SAT solver respects all hard constraints (non-overlapping slots on identical sections, manpower capacity limits).

---

### E. Error Handling
- **Status**: **PASS**
- **Analysis**:
  - Controlled failure testing confirmed proper HTTP status mapping:
    - **Model Unavailable**: Returns HTTP 503 with `{"error": "MODEL_UNAVAILABLE", "detail": "Production ML model is not loaded", "status_code": 503}`.
    - **Malformed Input**: Returns HTTP 422 with Pydantic structured errors.
    - **Engine / Runtime Exception**: Returns HTTP 500 with `{"error": "INTERNAL_ERROR", "detail": "ML evaluation failed: <ExceptionType>", "status_code": 500}`.
    - **Scheduler Exception**: Returns HTTP 500 with `{"error": "INTERNAL_ERROR", "detail": "Scheduling failed: <ExceptionType>", "status_code": 500}`.
    - **Health Initialization Exception**: Returns HTTP 503 with `SERVICE_UNAVAILABLE`.
  - **Zero Information Leakage**: Verified that no Python tracebacks, internal filesystem paths (`/home/...`), or model weights are exposed in error bodies.
  - Responses are strictly `application/json`. No synthetic or fabricated successful payloads are ever returned on failure.

---

### F. Health / Readiness
- **Status**: **PASS**
- **Analysis**:
  - `GET /health` returns HTTP 200 with JSON payload.
  - **Readiness Semantics**: The endpoint verifies both process health and model availability:
    - `status`: "ok"
    - `service`: "railway-ai-ml"
    - `model_available`: `true` (dynamically checked from `calibrated_xgboost_available`)
    - `engine_version`: "0.1.0"
    - `components`: Readiness status for all 7 internal engines (`ready`)
    - `artifacts`: Availability status for model and evidence artifacts
  - Returns HTTP 503 if the ML engine fails initialization.

---

### G. Concurrency Safety
- **Status**: **PASS**
- **Analysis**:
  - Tested 10 concurrent requests using `concurrent.futures.ThreadPoolExecutor(max_workers=10)` against the running local service.
  - Results: 10/10 requests succeeded with HTTP 200.
  - Zero crashes, zero segmentation faults, zero thread deadlocks.
  - **Consistency**: Repeated requests with identical inputs returned identical failure probabilities (`0.0297`), proving that shared model instances in memory undergo no in-place mutation or race conditions during inference.

---

### H. CORS
- **Status**: **PASS**
- **Analysis**:
  - **Default Configuration**: `ML_CORS_ORIGINS` is unset/empty. Verified that no `Access-Control-Allow-Origin` header is returned on preflight `OPTIONS` or `GET` requests.
  - **Configured Origins**: When `ML_CORS_ORIGINS="http://localhost:3000,https://railway-app.internal"` is configured, requests with `Origin: http://localhost:3000` receive matching ACAO headers.
  - **Unconfigured Origins**: Requests with untrusted origins receive no ACAO header.
  - **No Wildcard**: Wildcard `*` is never enabled by default, preventing unintended cross-origin access.

---

### I. Environment Configuration
- **Status**: **WARNING**
- **Analysis**:
  - `.env.example` contains:
    - `ML_SERVICE_HOST=0.0.0.0`
    - `ML_SERVICE_PORT=8000`
    - `ML_SERVICE_ENV=development`
    - `ML_CORS_ORIGINS=`
  - **PASS**: Zero secrets, credentials, or API keys are present.
  - **WARNING**: `main.py` only reads `ML_CORS_ORIGINS`. When starting via standard CLI `uvicorn src.api.main:app`, uvicorn uses its own defaults unless `--host` and `--port` CLI arguments are passed. This is normal for uvicorn deployments but worth documenting.

---

### J. Documentation Accuracy
- **Status**: **WARNING**
- **Analysis**:
  - Compared `docs/integration/ML_API.md` against OpenAPI specification (`/openapi.json`) and live endpoint behavior:
    - **Endpoint paths**: `/health`, `/api/v1/evaluate`, `/api/v1/recommend-windows` (100% match).
    - **HTTP methods & Status codes**: 200, 422, 500, 503 (100% match).
    - **Request payload fields & defaults**: 100% match.
    - **Evaluation response**: 100% match with `schemas/ml_response.schema.json`.
  - **WARNING**: In `POST /api/v1/recommend-windows`, the live endpoint response returns `recommendations`, `recommendations_df`, `infeasible_tasks`, and `metrics`. The sample JSON in `ML_API.md` line 244 omitted `recommendations_df` from the illustrative response block.

---

### K. Node Consumer Simulation
- **Status**: **PASS**
- **Analysis**:
  - An independent Node.js v22 test client was executed against the running FastAPI service simulating the future `Smart-Railways/node-worker` interaction.
  - Script tested:
    1. Service health check (`GET /health`)
    2. Batch task evaluation (`POST /api/v1/evaluate`)
    3. Output validation of all 4 categories and HITL invariants
    4. Scheduling query (`POST /api/v1/recommend-windows`)
    5. Client error handling (HTTP 422 on empty tasks payload)
    6. Network connection failure handling
  - Result: **34/34 checks PASSED** in Node runtime with zero unhandled promise rejections.

---

### L. Connection Failure Behavior
- **Status**: **PASS**
- **Analysis**:
  - Connection failure was simulated by attempting connections to an unallocated port (`18999`).
  - Node client threw standard `fetch failed` / `ECONNREFUSED` without hanging.
  - **Recommended Node Worker Policy for Future Integration**:
    - Set explicit request timeout: `AbortSignal.timeout(5000)`.
    - Retry policy: Maximum 2 retries with exponential backoff on HTTP 503 or network disconnect; zero retries on HTTP 422.
    - Graceful degradation: Mark maintenance task as `PENDING_ML_SERVICE` if ML service is unreachable rather than crashing the Node Worker process.

---

### M. Security Review
- **Status**: **PASS**
- **Analysis**:
  - **Hardcoded secrets**: None.
  - **Unrestricted CORS**: Disabled by default; no wildcard `*`.
  - **Information leakage**: Error handlers suppress stack traces and file paths.
  - **Unsafe deserialization**: Incoming data is parsed strictly as JSON into Pydantic models; no `pickle.loads` or `joblib.load` is exposed to user input.
  - **Arbitrary code execution**: None.
  - **Model path tampering**: Fixed internal paths; client cannot specify custom model paths.
  - **Debug mode**: `debug=False` by default in `FastAPI()`.

---

### N. Performance Baseline
- **Status**: **PASS**
- **Analysis**:
  - **Sequential Baseline** (20 requests with 3 tasks):
    - Minimum latency: `83.7 ms`
    - Maximum latency: `112.4 ms`
    - Average latency: `92.2 ms`
    - Median latency: `89.9 ms`
  - **Concurrent Baseline** (10 requests, 5 concurrent workers):
    - Minimum latency: `379.2 ms`
    - Maximum latency: `528.8 ms`
    - Average latency: `454.6 ms`
    - Median latency: `465.8 ms`
  - All responses completed within acceptable bounds for local HTTP microservices.

---

### O. Model Integrity
- **Status**: **PASS**
- **Analysis**:
  - SHA256 hashes of all serialized models were computed and verified:
    - **Production Model** (`models/production/calibrated_xgboost.pkl`):
      `e021f6b5ebbfe48e658a59a24a9ea2df171eb9d4d5667a802f70c8bc9b0017ae` (Exact match)
    - **Legacy Model** (`models/legacy/calibrated_xgboost_v2_legacy.pkl`):
      `3130e9a149e597f1fa24a638543de904d777814c915095d3fef45801bdebe386` (Exact match)
  - Zero model weights or configurations were modified during this validation phase.

---

### P. Regression Tests
- **Status**: **PASS**
- **Analysis**:
  - Full pytest test suite was executed:
    `PYTHONPATH=. .venv/bin/pytest tests/ -q`
  - **Test Suite Results**:
    - Existing ML/Domain tests: 160 passed
    - API tests: 13 passed
    - Total tests: **173 passed / 0 failed / 0 skipped** (10.78s)
  - No existing tests were modified or weakened.

---

## Repository Boundary Confirmation

- `Smart-Railways/node-worker`: Verified untouched. No files staged or modified during this phase.
- `backend/`: Verified untouched.
- `railway-ai`: Contains only the validated FastAPI service implementation, test suite, and audit documentation. Zero commits or pushes performed.

---

## Verdict

# FASTAPI INTEGRATION READINESS: PASS
