# Railway-AI ML API Reference

HTTP interface for the Railway-AI ML engine, consumed by the Node.js worker service.

## Architecture

```
Smart-Railways/node-worker (Node.js + Express)
         │
         │ HTTP
         ▼
Railway-AI ML Service (Python + FastAPI)
         │
         ▼
   RailwayMLEngine
         │
         ├── Failure Risk (CalibratedClassifierCV)
         ├── Priority / Decision (5-factor weighted)
         ├── Explanation (rule-based tags)
         └── Scheduling (CP-SAT optimizer)
```

## Base URL

```
http://localhost:8000
```

## Local Development

Start the service using either `python -m src.api.main` (reads `ML_SERVICE_HOST` and `ML_SERVICE_PORT` from environment) or direct `uvicorn`:

```bash
cd /home/chirag/projects/railway-ai

# Option A: Python entrypoint (honors ML_SERVICE_HOST and ML_SERVICE_PORT)
PYTHONPATH=. .venv/bin/python -m src.api.main

# Option B: Direct uvicorn command
PYTHONPATH=. .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Auto-generated docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Endpoints

### GET /health

Check service and ML model availability.

**Response 200:**
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

**Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Service healthy, model loaded |
| 503 | ML engine failed to initialize |

---

### POST /api/v1/evaluate

Full ML evaluation: risk prediction + priority scoring + explanation + scheduling.

Delegates to `RailwayMLEngine.evaluate_for_backend()`.

**Request Body:**
```json
{
  "tasks": [
    {
      "task_id": "TSK-001",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "estimated_duration": 60,
      "required_manpower": 2,
      "predicted_delay_minutes": 5.0,
      "asset_age_years": 15.0,
      "condition_score": 5.0,
      "criticality": 4.0,
      "usage_factor": 0.7,
      "historical_failure_count": 3,
      "historical_downtime_hours": 50.0,
      "days_since_last_failure": 90.0
    }
  ],
  "block_windows": null,
  "max_recommendations": 3
}
```

**Task Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `task_id` | string | ✅ | — | Unique task identifier |
| `section_id` | string | | `"NDL-MTJ-01"` | Railway corridor section |
| `department` | string | | `"ENGINEERING"` | `ENGINEERING`, `S&T`, or `TRACTION` |
| `estimated_duration` | int | | `60` | Duration in minutes |
| `required_manpower` | int | | `2` | Personnel count |
| `predicted_delay_minutes` | float | | `0.0` | Predicted delay |
| `asset_age_years` | float | | null | Asset age |
| `condition_score` | float | | null | Condition score |
| `criticality` | float | | null | Criticality (1-10) |
| `usage_factor` | float | | null | Usage factor |
| `historical_failure_count` | int | | null | Past failure count |
| `historical_downtime_hours` | float | | null | Past downtime hours |
| `days_since_last_failure` | float | | null | Days since last failure |

**Response 200:**
```json
{
  "status": "SUCCESS",
  "engine_version": "0.1.0",
  "tasks_evaluated": 1,
  "results": [
    {
      "task_id": "TSK-001",
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
        "reason_tags": ["ROUTINE_INSPECTION"],
        "score_breakdown": { "..." : "same as priority" },
        "explanation_summary": "Task TSK-001 prioritized as LOW (score: 0.1247) driven by: ROUTINE_INSPECTION."
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
            "recommendation_reasons": ["LOWEST_OPERATIONAL_DELAY"],
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
        "total_recommendations": 1,
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

**Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Evaluation successful |
| 422 | Invalid request (missing fields, empty tasks) |
| 500 | Internal ML engine error |
| 503 | Production model not loaded |

---

### POST /api/v1/recommend-windows

Generate CP-SAT scheduling recommendations only.

Runs `engine.predict()` to score tasks, then delegates to `RailwayMLEngine.recommend_windows()`.

**Request Body:**
```json
{
  "tasks": [
    {
      "task_id": "SCHED-001",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "estimated_duration": 60,
      "required_manpower": 2,
      "predicted_delay_minutes": 5.0,
      "asset_age_years": 15.0,
      "condition_score": 5.0,
      "criticality": 4.0,
      "usage_factor": 0.7,
      "historical_failure_count": 3,
      "historical_downtime_hours": 50.0,
      "days_since_last_failure": 90.0
    }
  ],
  "block_windows": null,
  "max_recommendations": 3
}
```

**Response 200:**
```json
{
  "recommendations": {
    "SCHED-001": [
      {
        "task_id": "SCHED-001",
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
      "task_id": "SCHED-001",
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

**Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Scheduling successful |
| 422 | Invalid request |
| 500 | Scheduling engine error |
| 503 | Production model not loaded |

---

## Response Contract

All evaluate responses conform to `schemas/ml_response.schema.json`.

The 4 frozen output categories are:
1. **risk** — `failure_probability`, `predicted_failure_30d`
2. **priority** — `priority_score`, `priority_category`, `score_breakdown`
3. **explanation** — `reason_tags`, `score_breakdown`, `explanation_summary`
4. **scheduling** — `recommended_windows` (max 3), `total_recommendations`, `has_feasible_window`

Human-in-the-loop invariants apply to all scheduling windows:
- `selection_status = "RECOMMENDED"`
- `human_confirmation_required = true`
- `is_booked = false`
- `booking_status = "UNBOOKED"`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | (injected by host) | Cloud deployment port (e.g. Render) |
| `ML_SERVICE_HOST` | `0.0.0.0` | Bind address |
| `ML_SERVICE_PORT` | `8000` | Fallback port when starting via python entrypoint |
| `ML_CORS_ORIGINS` | (empty) | Comma-separated allowed CORS origins |

## Production Deployment (Render / Cloud)

### Deployment Commands
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
  ```

### Port Behavior
- When deployed on platforms such as Render, the host dynamically provides a `$PORT` environment variable.
- The Uvicorn start command passes `--port $PORT` directly.
- The service binds to `0.0.0.0` to accept incoming traffic from the external router.

### Service Endpoints for Operations
- `GET /health`: Health and readiness probe returning HTTP 200 with model and component status, or 503 if unready.
- `POST /api/v1/evaluate`: Full ML evaluation (risk prediction, priority scoring, explanation tags, CP-SAT block windows).
- `POST /api/v1/recommend-windows`: Maintenance block window scheduling recommendations via CP-SAT solver.

## Node Worker Integration

The Node.js worker (`Smart-Railways/node-worker`) will consume this API over HTTP. The Node Worker is **not modified** in this phase — this document specifies the contract for future integration.

```typescript
// Future Node Worker usage example:
const response = await fetch('http://ml-service:8000/api/v1/evaluate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ tasks: maintenanceTasks }),
});
const result = await response.json();
```
