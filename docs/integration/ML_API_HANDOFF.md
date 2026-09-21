# Railway-AI ML API Integration Handoff Specification

**Document Version**: 1.0.0
**Target Consumer**: `Smart-Railways/node-worker` (Node.js + TypeScript) and Railway Operations Backend
**API Version**: `/api/v1/`
**Status**: FROZEN & VALIDATED

---

## A. Base Service

### 1. Service Overview
The Railway-AI ML Service is a standalone HTTP microservice that encapsulates:
- XGBoost failure risk prediction calibrated for 30-day maintenance horizons.
- 5-factor weighted priority decision scoring.
- Rule-based explainability tags and decision summaries.
- Google OR-Tools CP-SAT constrained optimization for maintenance block scheduling.

All ML and scheduling intelligence executes within this Python service. External consumers (Node Worker, backend) interact exclusively via standardized HTTP/JSON interfaces.

### 2. Local Startup
From the root of the `railway-ai` repository:

```bash
cd /home/chirag/projects/railway-ai

# Method 1: Python entrypoint (honors ML_SERVICE_HOST and ML_SERVICE_PORT env vars)
PYTHONPATH=. .venv/bin/python -m src.api.main

# Method 2: Direct Uvicorn CLI
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Service Configuration & Host/Port
Configuration is loaded via environment variables (`.env` or process environment):

| Variable | Default | Purpose |
|----------|---------|---------|
| `ML_SERVICE_HOST` | `0.0.0.0` | Bind IP address |
| `ML_SERVICE_PORT` | `8000` | Port number |
| `ML_SERVICE_ENV` | `development` | Deployment environment identifier |
| `ML_CORS_ORIGINS` | *(empty)* | Comma-separated allowed CORS origins (disabled by default) |

- **Default Base URL**: `http://127.0.0.1:8000` (or `http://localhost:8000`)
- **Interactive Documentation**:
  - Swagger UI: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`
  - OpenAPI Spec: `http://localhost:8000/openapi.json`

### 4. Production Deployment (Render / Cloud)
- **Environment Variables & Port Behavior**:
  - Cloud platforms (such as Render) automatically inject the `PORT` environment variable.
  - Set `ML_SERVICE_HOST=0.0.0.0` to bind to all available interfaces.
  - `ML_SERVICE_PORT=8000` serves as the fallback local port when `PORT` is not defined.
  - `ML_CORS_ORIGINS`: Optional comma-separated list of allowed origins. Leave empty unless browser-based requests require CORS.
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Startup Command**:
  ```bash
  uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
  ```
- **Service Endpoints for Operations**:
  - `GET /health`: Health and readiness probe returning HTTP 200 with model/component status, or 503 if unready.
  - `POST /api/v1/evaluate`: Full ML evaluation (failure risk, priority score, decision tags, CP-SAT block windows).
  - `POST /api/v1/recommend-windows`: Maintenance block window scheduling recommendations via CP-SAT solver.

---

## B. Endpoints Specification

### Endpoint 1: Health & Readiness Check
- **HTTP Method**: `GET`
- **Path**: `/health`
- **Purpose**: Verifies that the FastAPI process is running, internal pipeline components are initialized, and the calibrated XGBoost production model is loaded in memory.
- **Request**: No request body.
- **HTTP Status Codes**:
  - `200 OK`: Service healthy and model ready for inference.
  - `503 Service Unavailable`: Process started but ML engine/model failed initialization.

#### Example Request:
```http
GET /health HTTP/1.1
Host: localhost:8000
```

#### Example Response (200 OK):
```json
{
  "status": "ok",
  "service": "railway-ai-ml",
  "model_available": true,
  "engine_version": "0.1.0",
  "components": {
    "decision_engine": "ready",
    "failure_predictor": "ready",
    "block_optimizer": "ready",
    "multi_horizon_planner": "ready",
    "planning_service": "ready",
    "real_corridor_planner": "ready",
    "lifecycle_manager": "ready"
  },
  "artifacts": {
    "calibrated_xgboost_available": true,
    "real_section_evidence_available": true
  }
}
```

---

### Endpoint 2: Full ML Evaluation
- **HTTP Method**: `POST`
- **Path**: `/api/v1/evaluate`
- **Purpose**: Evaluates a batch of maintenance tasks and returns failure risk, priority scoring, explanation tags, and ranked CP-SAT block window recommendations in a single unified payload.
- **HTTP Status Codes**:
  - `200 OK`: Successful evaluation.
  - `422 Unprocessable Entity`: Request validation failure (malformed JSON, empty task list, missing fields).
  - `500 Internal Server Error`: Controlled runtime failure within the ML engine (returns structured JSON, zero stack traces).
  - `503 Service Unavailable`: Production model is missing or unavailable.

#### Request Schema & Fields:
- `tasks` *(array of objects, **REQUIRED**, min 1 task)*:
  - `task_id` *(string, **REQUIRED**)*: Unique identifier for the maintenance task.
  - `section_id` *(string, optional, default: `"NDL-MTJ-01"`)*: Corridor section code.
  - `department` *(string, optional, default: `"ENGINEERING"`)*: One of `ENGINEERING`, `S&T`, `TRACTION`.
  - `estimated_duration` *(integer minutes, optional, default: `60`, min: `1`)*: Required block duration.
  - `required_manpower` *(integer, optional, default: `2`, min: `1`)*: Personnel count needed on track.
  - `predicted_delay_minutes` *(float, optional, default: `0.0`, min: `0.0`)*: Upstream estimated train delay.
  - **Asset Telemetry Features** *(7 production features, optional - defaults applied if omitted)*:
    - `asset_age_years` *(float, optional)*: Age of the asset in years.
    - `condition_score` *(float, optional, 1-10)*: Asset inspection condition score.
    - `criticality` *(float, optional, 1-10)*: Asset operational criticality rating.
    - `usage_factor` *(float, optional, 0.0-1.0)*: Intensity of track/signaling usage.
    - `historical_failure_count` *(integer, optional)*: Previous recorded failure events.
    - `historical_downtime_hours` *(float, optional)*: Cumulative downtime hours.
    - `days_since_last_failure` *(float, optional)*: Days elapsed since most recent breakdown.
  - **Precomputed Alternatives** *(optional)*:
    - `failure_probability` *(float, optional, 0.0-1.0)*
    - `urgency_score` *(float, optional, 0.0-1.0)*
    - `overdue_days` *(float, optional)*
    - `railkit_operational_pressure` *(float, optional, 0.0-1.0)*
- `block_windows` *(array of objects, optional, default: internal standard blocks)*:
  - `block_id` *(string, **REQUIRED**)*: Block window identifier (e.g. `"B001"`).
  - `start_slot` *(integer, **REQUIRED**, min: `0`)*: 30-minute slot start index (inclusive).
  - `end_slot` *(integer, **REQUIRED**, min: `1`)*: 30-minute slot end index (exclusive).
- `max_recommendations` *(integer, optional, default: `3`, bounds: `1` to `3`)*: Cap on recommended windows per task.

#### Example Request:
```json
{
  "tasks": [
    {
      "task_id": "TSK-2026-001",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "estimated_duration": 60,
      "required_manpower": 2,
      "predicted_delay_minutes": 5.0,
      "asset_age_years": 14.5,
      "condition_score": 5.2,
      "criticality": 4.0,
      "usage_factor": 0.72,
      "historical_failure_count": 3,
      "historical_downtime_hours": 48.0,
      "days_since_last_failure": 85.0
    }
  ],
  "max_recommendations": 3
}
```

#### Example Response (200 OK):
```json
{
  "status": "SUCCESS",
  "engine_version": "0.1.0",
  "tasks_evaluated": 1,
  "results": [
    {
      "task_id": "TSK-2026-001",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "decision_rank": 1,
      "risk": {
        "failure_probability": 0.0297,
        "predicted_failure_30d": true
      },
      "priority": {
        "priority_score": 0.1247,
        "priority_category": "LOW",
        "score_breakdown": {
          "failure_risk_factor": 0.0297,
          "criticality_factor": 0.4,
          "urgency_factor": 0.0,
          "overdue_factor": 0.0,
          "operational_factor": 0.2388,
          "total_score": 0.1247
        }
      },
      "explanation": {
        "reason_tags": [
          "ROUTINE_INSPECTION"
        ],
        "score_breakdown": {
          "failure_risk_factor": 0.0297,
          "criticality_factor": 0.4,
          "urgency_factor": 0.0,
          "overdue_factor": 0.0,
          "operational_factor": 0.2388,
          "total_score": 0.1247
        },
        "explanation_summary": "Task TSK-2026-001 prioritized as LOW (score: 0.1247) driven by: ROUTINE_INSPECTION."
      },
      "scheduling": {
        "recommended_windows": [
          {
            "rank": 1,
            "block_id": "B001",
            "start_slot": 1,
            "end_slot": 3,
            "duration_minutes": 60,
            "required_manpower": 2,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": false,
              "coordination_departments": [],
              "coordination_reason": "None"
            },
            "selection_status": "RECOMMENDED",
            "human_confirmation_required": true,
            "is_booked": false,
            "booking_status": "UNBOOKED"
          },
          {
            "rank": 2,
            "block_id": "B002",
            "start_slot": 6,
            "end_slot": 8,
            "duration_minutes": 60,
            "required_manpower": 2,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": false,
              "coordination_departments": [],
              "coordination_reason": "None"
            },
            "selection_status": "RECOMMENDED",
            "human_confirmation_required": true,
            "is_booked": false,
            "booking_status": "UNBOOKED"
          }
        ],
        "total_recommendations": 2,
        "has_feasible_window": true
      }
    }
  ],
  "metrics": {
    "tasks_count": 1,
    "critical_tasks_count": 0,
    "high_tasks_count": 0,
    "medium_tasks_count": 0,
    "low_tasks_count": 1,
    "tasks_with_recommendations": 1,
    "infeasible_tasks": []
  }
}
```

---

### Endpoint 3: Block Window Scheduling Only
- **HTTP Method**: `POST`
- **Path**: `/api/v1/recommend-windows`
- **Purpose**: Executes CP-SAT constrained optimization to compute ranked block window recommendations for maintenance tasks without returning full telemetry evaluation breakdowns.
- **HTTP Status Codes**:
  - `200 OK`: Successful scheduling computation.
  - `422 Unprocessable Entity`: Request validation failure.
  - `500 Internal Server Error`: Solver failure.
  - `503 Service Unavailable`: Model unavailable.

#### Request Fields:
Same schema as `POST /api/v1/evaluate` (`tasks`, `block_windows`, `max_recommendations`).

#### Example Response (200 OK):
```json
{
  "recommendations": {
    "TSK-2026-001": [
      {
        "task_id": "TSK-2026-001",
        "section_id": "NDL-MTJ-01",
        "department": "ENGINEERING",
        "block_id": "B001",
        "start_slot": 1,
        "end_slot": 3,
        "duration_minutes": 60,
        "required_manpower": 2,
        "maintenance_decision_score": 0.1247,
        "predicted_delay_minutes": 5.0,
        "feasibility_status": "FEASIBLE",
        "recommendation_reasons": ["LOW_OPERATIONAL_DELAY"],
        "coordination": {
          "is_coordinated": false,
          "coordinated_task_ids": [],
          "coordination_departments": [],
          "coordination_reason": null
        },
        "selection_status": "RECOMMENDED",
        "human_confirmation_required": true,
        "is_booked": false,
        "booking_status": "UNBOOKED",
        "rank": 1
      }
    ]
  },
  "recommendations_df": [
    {
      "task_id": "TSK-2026-001",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "block_id": "B001",
      "start_slot": 1,
      "end_slot": 3,
      "duration_minutes": 60,
      "required_manpower": 2,
      "maintenance_decision_score": 0.1247,
      "predicted_delay_minutes": 5.0,
      "feasibility_status": "FEASIBLE",
      "recommendation_reasons": ["LOW_OPERATIONAL_DELAY"],
      "selection_status": "RECOMMENDED",
      "human_confirmation_required": true,
      "is_booked": false,
      "booking_status": "UNBOOKED",
      "rank": 1,
      "is_coordinated": false,
      "coordinated_task_ids": [],
      "coordination_departments": [],
      "coordination_reason": null
    }
  ],
  "infeasible_tasks": [],
  "metrics": {
    "total_tasks": 1,
    "tasks_with_recommendations": 1,
    "total_recommendations": 1,
    "infeasible_task_count": 0
  }
}
```

---

## C. `/evaluate` Semantics: The Four Output Categories

Every task in `results` returns four distinct categories:

### 1. `risk`
- `failure_probability` *(float between 0.0 and 1.0)*: Calibrated probability that the asset will suffer a functional failure within 30 days.
- `predicted_failure_30d` *(boolean)*: True if `failure_probability >= 0.0120` (the fixed production threshold chosen to balance precision and operational recall).
- *Contract Note*: Does NOT contain heuristic confidence percentages or uncalibrated risk categories.

### 2. `priority`
- `priority_score` *(float between 0.0 and 1.0)*: Weighted multi-criteria maintenance urgency score.
- `priority_category` *(string)*: One of `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`.
  - `[0.00, 0.25)` -> `LOW`
  - `[0.25, 0.50)` -> `MEDIUM`
  - `[0.50, 0.75)` -> `HIGH`
  - `[0.75, 1.00]` -> `CRITICAL`
- `score_breakdown` *(object)*: Exact factor contributions:
  - `failure_risk_factor` (weight: 0.35)
  - `criticality_factor` (weight: 0.25)
  - `urgency_factor` (weight: 0.15)
  - `overdue_factor` (weight: 0.15)
  - `operational_factor` (weight: 0.10)
  - `total_score` (sum of weighted factors)

### 3. `explanation`
- `reason_tags` *(array of strings)*: Machine-parseable tags explaining decision drivers (e.g. `HIGH_FAILURE_RISK`, `CRITICAL_ASSET`, `OVERDUE_MAINTENANCE`, `ROUTINE_INSPECTION`).
- `score_breakdown`: Identical to priority breakdown.
- `explanation_summary` *(string)*: Natural language synthesis suitable for display on maintenance controller dashboards.

### 4. `scheduling`
- `recommended_windows` *(array of objects)*: Ranked list of up to 3 feasible block windows computed by CP-SAT.
- `total_recommendations` *(integer 0 to 3)*.
- `has_feasible_window` *(boolean)*: True if at least one conflict-free window exists within the planning horizon.

---

## D. Scheduling Semantics & Human-In-The-Loop Invariants

### 1. Recommendations Are NOT Bookings
The CP-SAT scheduling optimizer is strictly an **advisory decision-support engine**. It calculates mathematically feasible, non-conflicting time windows, but **never auto-books track occupancy**.

### 2. Invariant Fields (Enforced on Every Window)
Every item inside `recommended_windows` carries immutable verification flags:
- `selection_status = "RECOMMENDED"` (indicates candidate option)
- `human_confirmation_required = true` (mandatory controller approval flag)
- `is_booked = false` (track allocation is unreserved)
- `booking_status = "UNBOOKED"` (formal booking status is unassigned)

### 3. Recommendation Bounds
- **Maximum 3 recommendations**: Even if 10 windows are feasible, the API returns at most the Top-3 highest-ranked options.
- **Zero recommendations (`[]`)**: If a task requires more duration than available blocks or manpower exceeds track limits, `recommended_windows` is `[]`, `has_feasible_window` is `false`, and the task is flagged in `infeasible_tasks`.

---

## E. Error Handling Specification

All error responses are guaranteed to be structured JSON. **Python stack traces, memory addresses, and file paths are never exposed.**

### Error Categories:

| Status Code | Error Code | Meaning | When Triggered |
|-------------|------------|---------|----------------|
| **422** | Pydantic Validation Error | Invalid Client Request | Empty task list, missing `task_id`, negative duration/manpower, non-numeric telemetry types. |
| **500** | `INTERNAL_ERROR` | Controlled Internal Failure | Unexpected exception during ML scoring or solver execution. Detail contains sanitized error class name. |
| **503** | `MODEL_UNAVAILABLE` | Model Not Ready | Calibrated XGBoost production artifact is missing or corrupted. |
| **503** | `SERVICE_UNAVAILABLE` | Engine Init Failure | Engine health check failed during startup. |

#### Error Body Examples:

**422 Validation Error Example:**
```json
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "tasks"],
      "msg": "List should have at least 1 item after validation, not 0",
      "input": []
    }
  ]
}
```

**500 Internal Error Example:**
```json
{
  "error": "INTERNAL_ERROR",
  "detail": "ML evaluation failed: ValueError",
  "status_code": 500
}
```

**503 Model Unavailable Example:**
```json
{
  "error": "MODEL_UNAVAILABLE",
  "detail": "Production ML model is not loaded",
  "status_code": 503
}
```

---

## F. Client Integration Guidance (For Node Worker / Backend)

### 1. Request Headers
Always send:
```http
Content-Type: application/json
Accept: application/json
```

### 2. Timeouts
- Recommended client timeout: **5000 ms (5 seconds)**.
- ML inference and CP-SAT scheduling typically complete in under **150 ms** for batches of up to 10 tasks.
- In Node.js, use `AbortSignal.timeout(5000)`:
  ```typescript
  const response = await fetch(`${ML_SERVICE_URL}/api/v1/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(5000),
  });
  ```

### 3. Handling Connection & Network Failures
If the ML service process is down or unreachable:
- The Node HTTP client throws an `ECONNREFUSED` or `fetch failed` error.
- **Do not crash the Node Worker process.**
- Catch the error and mark the task status in the database as `PENDING_ML_EVALUATION` or fallback to rule-based offline scheduling.

### 4. HTTP Status Code Handling
- **`200`**: Parse `response.json()` and proceed.
- **`422`**: Client programming error. Log payload for inspection. **Do NOT retry automatically.**
- **`503`**: Service starting up or model loading. Retry up to 2 times with a 2-second delay.
- **`500`**: Unexpected engine error. Log error detail and escalate to ML service alert.

### 5. Architectural Boundaries
- Do not attempt to import Python modules or read `.pkl` files directly from Node.js or the backend.
- Do not invent frontend heuristics to override `failure_probability` or `priority_score`.
- Treat `schemas/ml_response.schema.json` as the frozen machine-readable contract.

---

## G. API Versioning

- The canonical endpoint prefix is `/api/v1/`.
- Future major revisions will be mounted at `/api/v2/` without breaking `/api/v1/`.
- All Node Worker and backend integrations must target `/api/v1/evaluate` and `/api/v1/recommend-windows`.
