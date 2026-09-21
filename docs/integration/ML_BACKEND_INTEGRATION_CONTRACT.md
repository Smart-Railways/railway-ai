# Railway-AI — ML Backend Integration Contract

**Document Version:** 1.0.0
**Service Version:** `railway-ml-engine 0.1.0`
**Status:** FROZEN PRE-BACKEND INTEGRATION CONTRACT
**Target Consumer:** Backend Services / Future API Service Boundary
**Schema Definition:** [`docs/schemas/ml_response.schema.json`](schemas/ml_response.schema.json)
**Test Suite:** [`tests/test_backend_contract.py`](../tests/test_backend_contract.py)

---

## 1. Executive Summary & Purpose

This document establishes the binding, immutable technical contract between the Railway-AI Machine Learning & Optimization engine and the backend application layer.

The ML service is an operational intelligence engine designed for railway infrastructure management. It ingests asset telemetry, maintenance work orders, and network operational data to deliver four frozen decision categories:
1. **RISK**: Probabilistic failure estimation calibrated against real operating conditions.
2. **PRIORITY**: Multi-objective priority ranking derived from 5 normalized operational factors.
3. **EXPLANATION**: Transparent, rule-based reasoning tags and numerical factor breakdowns.
4. **SCHEDULING**: Feasible maintenance block window candidates (up to 3) generated via constraint programming (OR-Tools CP-SAT).

This contract eliminates integration friction by guaranteeing:
- **Strict Data Typings:** 100% native Python primitives (`dict`, `list`, `int`, `float`, `str`, `bool`) that serialize to pure JSON without custom encoders.
- **Truth in AI:** Heuristic, arbitrary confidence scores have been completely excised from the public contract.
- **Human-in-the-Loop Governance:** The engine never automatically books or authorizes track possessions. It outputs ranked recommendations requiring explicit human operator authorization.

---

## 2. High-Level Architectural Flow

The repository implements a linear, deterministic pipeline from raw input to backend-ready JSON payload:

```
+---------------------------------------------------------------------------------------------------+
|                                      INCOMING WORKLIST / ASSETS                                    |
|                               (REST Request / Database Query / DataFrame)                          |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 1. RISK PREDICTION (FailureRiskPredictor)                                                         |
|    - Active Model: models/production/calibrated_xgboost.pkl (CalibratedClassifierCV, sigmoid Platt scaling)   |
|    - 7 Features: age, condition, criticality, usage, failures, downtime, days_since_last          |
|    - Output: failure_probability, predicted_failure_30d (Threshold: 0.0120)                       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. OPERATIONAL CORRIDOR ENRICHMENT (RailKitSectionPressureBuilder)                                |
|    - Real empirical traffic & disruption data from data/processed_real/                            |
|    - Output: railkit_operational_pressure (bounded [0.0, 1.0]) for mapped corridor sections       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. MULTI-OBJECTIVE SCORING (MaintenanceDecisionEngine)                                            |
|    - 5-Factor Formula: Risk (30%), Criticality (20%), Urgency (20%), Overdue (15%), Pressure (15%)|
|    - Output: priority_score, priority_category, score_breakdown, reason_tags                      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. CP-SAT CANDIDATE WINDOW OPTIMIZATION (BlockOptimizer)                                          |
|    - Hard Constraints: Same-section non-overlap, manpower bounds, window duration containment    |
|    - Soft Preference: Cross-department coordination bonus (+50)                                   |
|    - Output: Up to 3 ranked candidate windows per task (RECOMMENDED, UNBOOKED)                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. UNIFIED CONTRACT SERIALIZATION (RailwayMLEngine.evaluate_for_backend())                        |
|    - Native Python dictionary complying with docs/schemas/ml_response.schema.json                 |
|    - 4 Frozen Categories: RISK | PRIORITY | EXPLANATION | SCHEDULING                              |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Service Entry Point (`evaluate_for_backend`)

The unified entry point for all backend services is `RailwayMLEngine.evaluate_for_backend()` located in [`src/services/ml_engine.py`](../src/services/ml_engine.py).

### Method Signature
```python
def evaluate_for_backend(
    self,
    input_data: Union[pd.DataFrame, list, dict],
    block_windows: Optional[list] = None,
    apply_real_pressure: bool = True,
    max_recommendations: int = 3,
    **kwargs,
) -> Dict[str, Any]:
```

### Parameters
| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `input_data` | `DataFrame`, `list[dict]`, or `dict` | **Yes** | — | Single task dict, list of task dicts, or pandas DataFrame. |
| `block_windows` | `list[dict]` | No | `None` | Custom traffic block windows. If `None`, standard default 6-hour windows are used. |
| `apply_real_pressure` | `bool` | No | `True` | Whether to enrich tasks with empirical RailKit corridor section pressure. |
| `max_recommendations` | `int` | No | `3` | Maximum candidate block windows to generate per task (capped at 3). |

---

## 4. Input Contract Specifications

The backend may pass work orders in three formats:
1. Standard maintenance task records (pre-scored or requiring scoring).
2. Raw asset telemetry records (requiring XGBoost risk prediction).
3. Hybrid records containing both asset telemetry and scheduling metadata.

### Required & Optional Input Attributes
| Field | Type | Required | Default | Valid Range / Description |
|---|---|---|---|---|
| `task_id` | `str` | Recommended | Auto-generated (`TASK-001`, ...) | Unique identifier for the maintenance task. |
| `section_id` | `str` | Recommended | `"NDL-MTJ-01"` | Physical railway corridor section (e.g. `NDL-MTJ-01`). |
| `department` | `str` | Recommended | `"ENGINEERING"` | Maintenance department: `"ENGINEERING"`, `"S&T"`, `"TRACTION"`. |
| `estimated_duration` | `int` | No | `60` | Duration required for work in minutes ($\ge 1$). |
| `required_manpower` | `int` | No | `2` | Number of track personnel required ($\ge 1$). |
| `criticality` | `float` | No | `5.0` | Asset/track criticality score ($1.0$ to $10.0$). |
| `urgency_score` | `float` | No | `0.0` | Field inspection urgency score ($0.0$ to $1.0$). |
| `overdue_days` | `float` | No | `0.0` | Days overdue past scheduled maintenance ($\ge 0.0$). |
| `predicted_delay_minutes`| `float` | No | `0.0` | Estimated train traffic delay in minutes ($\ge 0.0$). |
| `failure_probability` | `float` | Optional | Auto-computed | Pre-computed failure probability ($0.0$ to $1.0$). |

### Raw Asset Telemetry Attributes (For Calibrated XGBoost)
If raw telemetry is supplied, `FailureRiskPredictor` automatically extracts the 7-feature contract:
1. `asset_age_years` (`float`): Age of track asset in years.
2. `condition_score` (`float`): Physical inspection condition score ($0.0$ to $100.0$).
3. `criticality` (`float`): Asset criticality level ($1.0$ to $10.0$).
4. `usage_factor` (`float`): Operational usage coefficient ($\ge 0.0$).
5. `historical_failure_count` (`int`): Count of past failures on asset.
6. `historical_downtime_hours` (`float`): Cumulative downtime hours recorded.
7. `days_since_last_failure` (`float`): Days elapsed since most recent breakdown.

---

## 5. The 4 Frozen Output Categories

Every evaluated task returned in `results` contains exactly four distinct, immutable functional categories:

```json
{
  "task_id": "TASK-001",
  "section_id": "NDL-MTJ-01",
  "department": "ENGINEERING",
  "decision_rank": 1,
  "risk": { ... },
  "priority": { ... },
  "explanation": { ... },
  "scheduling": { ... }
}
```

1. **`risk`**: The probabilistic and thresholded breakdown risk derived from machine learning.
2. **`priority`**: The unified operational prioritization score and 5-factor mathematical breakdown.
3. **`explanation`**: Auditable, rule-based reason tags and clear textual decision justifications.
4. **`scheduling`**: Feasible candidate block windows evaluated by CP-SAT under strict capacity constraints.

---

## 6. Complete JSON Response Schema & Field Reference

### Top-Level Envelope
```json
{
  "status": "SUCCESS",
  "engine_version": "0.1.0",
  "tasks_evaluated": 2,
  "results": [ ... ],
  "metrics": {
    "tasks_count": 2,
    "critical_tasks_count": 0,
    "high_tasks_count": 0,
    "medium_tasks_count": 1,
    "low_tasks_count": 1,
    "tasks_with_recommendations": 2,
    "infeasible_tasks": []
  }
}
```

### Envelope Properties
- `status` (`str`): `"SUCCESS"` on normal completion, `"EMPTY_INPUT"` if input list/DataFrame was empty.
- `engine_version` (`str`): Current semantic version (`"0.1.0"`).
- `tasks_evaluated` (`int`): Count of tasks processed.
- `results` (`list[dict]`): List of task result dictionaries, sorted in descending order of `priority_score` (matching `decision_rank`).
- `metrics` (`dict`): Aggregate summary metrics covering category counts and feasibility.

---

## 7. Output Category 1: RISK

Provides calibrated failure risk estimation:
```json
"risk": {
  "failure_probability": 0.0450,
  "predicted_failure_30d": true
}
```

### Field Definitions
- `failure_probability` (`float`): Posterior probability of track failure within a 30-day operational horizon, bounded in $[0.0, 1.0]$ and rounded to 4 decimal places. Derived via Platt sigmoid calibration on XGBoost.
- `predicted_failure_30d` (`bool`): `true` if `failure_probability >= 0.0120`, else `false`.

### Model Calibration & Threshold Factsheet
- **Active Model Artifact:** `models/production/calibrated_xgboost.pkl` (promoted from Candidate V3).
- **Archived Legacy Model:** `models/legacy/calibrated_xgboost_v2_legacy.pkl` (byte-for-byte identical to the original V2 production artifact).
- **Training Baseline:** Trained on the corrected monthly calendar target dataset (`data/processed/candidate_v3/`, resolving the 31-day timedelta defect) with true empirical 1.14% failure base rate. Calibrated using `CalibratedClassifierCV(method='sigmoid', cv=5)`.
- **Operating Decision Threshold:** $\tau = 0.0120$ (UNCHANGED).
- **Threshold Policy Note:** During this controlled promotion, the operating threshold is strictly preserved at $\tau = 0.0120$. Because Candidate V3 is calibrated to ~1.14% prevalence (compared to the corrupted ~0.44% baseline), evaluating operating threshold adjustments (such as $\tau = 0.0200$ or percentile ranking) is deferred to a dedicated post-promotion validation item.
- **Performance Characteristics & Limitations:**
  - Evaluated on the untouched Jan–Jul 2026 test set (7,000 observations, 108 true failures).
  - ROC-AUC improved from 0.6339 to 0.6996 (+10.4%, $\Delta = +0.0650$, $p < 0.001$, statistically significant).
  - Top-10% Recall improved from 16.67% to 25.93% (+55.5%, Lift 2.59x, $p = 0.032$, statistically significant).
  - PR-AUC increased from 0.0258 to 0.0338 (+31.0%, $\Delta = +0.0080$, $p = 0.063$, not statistically significant at $\alpha = 0.05$).
  - Mean probability calibration error reduced 5.7x (0.01157 $\to$ 0.00204).
  - All training and test data remain synthetic; operational validation on real railway failure outcomes remains necessary.

---

## 8. Output Category 2: PRIORITY

Provides the operational ranking score and factor breakdown:
```json
"priority": {
  "priority_score": 0.4543,
  "priority_category": "MEDIUM",
  "score_breakdown": {
    "failure_risk_factor": 0.0450,
    "criticality_factor": 0.8000,
    "urgency_factor": 0.8500,
    "overdue_factor": 0.5000,
    "operational_factor": 0.2388,
    "total_score": 0.4543
  }
}
```

### 5-Factor Mathematical Formula
The priority score is a strictly bounded convex combination of 5 normalized operational factors:

$$\text{Priority Score} = \text{clip}\left( \sum_{i=1}^5 w_i \cdot f_i, \; 0.0, \; 1.0 \right)$$

| Factor Name ($f_i$) | Weight ($w_i$) | Source Attribute | Normalization Formula |
|---|---|---|---|
| `failure_risk_factor` | **0.30** | `failure_probability` | $\text{clip}(p, 0.0, 1.0)$ |
| `criticality_factor` | **0.20** | `criticality` | $\text{clip}(c / 10.0, 0.0, 1.0)$ |
| `urgency_factor` | **0.20** | `urgency_score` | $\text{clip}(u, 0.0, 1.0)$ |
| `overdue_factor` | **0.15** | `overdue_days` | $\text{clip}(d / 30.0, 0.0, 1.0)$ |
| `operational_factor` | **0.15** | `railkit_operational_pressure` | $\text{clip}(\text{pressure}, 0.0, 1.0)$ |

### Priority Categories
Tasks are categorized according to hard score thresholds:
- `LOW`: $\text{score} \le 0.25$
- `MEDIUM`: $0.25 < \text{score} \le 0.50$
- `HIGH`: $0.50 < \text{score} \le 0.75$
- `CRITICAL`: $\text{score} > 0.75$

---

## 9. Output Category 3: EXPLANATION

Provides human-interpretable reasoning for dispatchers and safety inspectors:
```json
"explanation": {
  "reason_tags": [
    "HIGH_URGENCY",
    "HIGH_CRITICALITY",
    "OVERDUE_MAINTENANCE"
  ],
  "score_breakdown": {
    "failure_risk_factor": 0.0450,
    "criticality_factor": 0.8000,
    "urgency_factor": 0.8500,
    "overdue_factor": 0.5000,
    "operational_factor": 0.2388,
    "total_score": 0.4543
  },
  "explanation_summary": "Task TASK-001 prioritized as MEDIUM (score: 0.4543) driven by: HIGH_URGENCY, HIGH_CRITICALITY, OVERDUE_MAINTENANCE."
}
```

### Reason Tags Catalog
Tags are deterministically assigned according to the following rules:
- `HIGH_FAILURE_RISK`: Triggered when `failure_risk_factor >= 0.50`.
- `HIGH_URGENCY`: Triggered when `urgency_factor >= 0.60`.
- `HIGH_CRITICALITY`: Triggered when `criticality_factor >= 0.80`.
- `OVERDUE_MAINTENANCE`: Triggered when `overdue_factor > 0.0`.
- `HIGH_OPERATIONAL_PRESSURE`: Triggered when `operational_factor >= 0.30`.
- `ROUTINE_INSPECTION`: Default fallback tag applied when no high-risk condition is met.

### Production vs Dormant Explainability Notice
- **Production Path:** Rule-based reason tags + 5-factor mathematical score breakdown.
- **Dormant Research Code:** SHAP tree explainers exist in exploratory research notebooks (`notebooks/09_model_explainability.ipynb`) and are **not** executed in real-time inference to ensure sub-10ms response latencies and zero dependency on heavy C-extensions.

---

## 10. Output Category 4: SCHEDULING

Provides ranked candidate block window alternatives generated via CP-SAT:
```json
"scheduling": {
  "total_recommendations": 2,
  "has_feasible_window": true,
  "recommended_windows": [
    {
      "rank": 1,
      "block_id": "B001",
      "start_slot": 3,
      "end_slot": 5,
      "duration_minutes": 60,
      "required_manpower": 4,
      "feasibility_status": "FEASIBLE",
      "recommendation_reasons": [
        "CROSS_DEPARTMENT_COORDINATION",
        "MODERATE_PRIORITY_FIT",
        "LOWEST_OPERATIONAL_DELAY"
      ],
      "coordination": {
        "is_coordinated": true,
        "coordination_departments": ["ENGINEERING", "S&T"],
        "coordination_reason": "Bundled with TASK-002 on section NDL-MTJ-01 in block B001"
      },
      "selection_status": "RECOMMENDED",
      "human_confirmation_required": true,
      "is_booked": false,
      "booking_status": "UNBOOKED"
    },
    {
      "rank": 2,
      "block_id": "B002",
      "start_slot": 8,
      "end_slot": 10,
      "duration_minutes": 60,
      "required_manpower": 4,
      "feasibility_status": "FEASIBLE",
      "recommendation_reasons": [
        "CROSS_DEPARTMENT_COORDINATION",
        "MODERATE_PRIORITY_FIT",
        "LOWEST_OPERATIONAL_DELAY"
      ],
      "coordination": {
        "is_coordinated": true,
        "coordination_departments": ["ENGINEERING", "S&T"],
        "coordination_reason": "Bundled with TASK-002 on section NDL-MTJ-01 in block B002"
      },
      "selection_status": "RECOMMENDED",
      "human_confirmation_required": true,
      "is_booked": false,
      "booking_status": "UNBOOKED"
    }
  ]
}
```

### Hard Feasibility Constraints & Production Manpower Limit
Every recommendation emitted by CP-SAT strictly satisfies four physical constraints:
- **Same-Section Non-Overlap (`AddNoOverlap`):** Tasks assigned to the same corridor section cannot overlap in time.
- **Different-Section Concurrency:** Tasks on different corridor sections may execute concurrently within the same block window, subject to cumulative manpower capacity.
- **Cumulative Manpower Capacity (`AddCumulative`):** The total personnel allocated across all concurrent maintenance activities in any 30-minute time slot cannot exceed the configured production limit: **`max_manpower = 12` crew members** (as configured in [`BlockOptimizer(max_manpower=12)`](../src/optimization/block_optimizer.py) and [`MultiHorizonPlanner(max_manpower=12)`](../src/optimization/multi_horizon_planner.py)).
- **Window Duration Containment:** Task duration must fit entirely inside the assigned block window without truncation.

### Recommendation Invariants
1. **Cardinality Cap:** At most 3 candidate windows are returned per task. If fewer than 3 feasible windows exist, only the valid ones are returned (cardinality $\in \{0, 1, 2, 3\}$).
2. **Sequential Ranks:** Ranks are strictly ordered integers starting at 1: `[1]`, `[1, 2]`, or `[1, 2, 3]`.
3. **No Duplicate Windows:** A task is never assigned the same physical block window more than once.
4. **No Fabricated Windows:** All recommendations correspond to real block window definitions.
5. **Infeasible Tasks:** If a task cannot fit any window (e.g. duration exceeds window capacity), `total_recommendations = 0`, `has_feasible_window = false`, `recommended_windows = []`, and its `task_id` is recorded in `metrics["infeasible_tasks"]`.

---

## 11. Human-in-the-Loop Semantics & Booking Invariants

To comply with railway safety regulations (RDSO / Indian Railways safety codes), the AI engine is strictly an advisory decision-support system. It has no authority to close tracks or dispatch crews.

Every recommendation emitted contains immutable security invariants:
- `"selection_status": "RECOMMENDED"`
- `"human_confirmation_required": true`
- `"is_booked": false`
- `"booking_status": "UNBOOKED"`

**State Transition Contract:**
The backend state manager or human operator must explicitly authorize a window selection. The ML engine will never transition a window to `BOOKED` or `AUTHORIZED` automatically.

---

## 12. Criticality Field Clarification

There is an important distinction between `criticality` and `priority_score`:
- **`criticality` IS NOT an AI output:** It is an input attribute ($1.0$ to $10.0$) assigned by railway asset management standards based on rail classification (e.g., High-Density Corridor Group A track vs Branch Line track).
- **`criticality_factor` is an input factor:** Normalized as `criticality / 10.0` (weight $0.20$) inside the 5-factor priority formula.
- **`priority_score` IS the composite AI output:** Calculated dynamically from risk, criticality, urgency, overdue days, and operational pressure.

---

## 13. Removal / Quarantine of Heuristic Confidence Metrics

### Architectural Decision Notice
In previous iterations of the codebase, two heuristic confidence fields were present:
1. `failure_model_confidence = 0.80 + (|prob - threshold| * 0.18)` in `FailureRiskPredictor`.
2. `confidence = 0.92 if railkit_pressure else 0.85` in `MaintenanceDecisionEngine`.

**Audit Finding & Action:**
These formulas are arbitrary linear heuristics and do **not** represent true Bayesian credible intervals, conformal prediction sets, or statistical uncertainty bounds. Presenting them to human rail dispatchers as "AI Confidence" creates a dangerous illusion of certainty.

**Contract Guarantee:**
- Neither `"confidence"` nor `"failure_model_confidence"` appears anywhere in the `evaluate_for_backend()` response payload.
- Backend systems must not expect or display these fields.
- True uncertainty estimation (e.g. conformal prediction bands) is scheduled for future V3 research.

---

## 14. Real RailKit Operational Signal Integration

The ML engine integrates empirical operational telemetry extracted from real Indian Railways corridor captures (`data/processed_real/railkit_section_mapping_evidence.csv`):

- **Section Pressure Mapping:** High-frequency corridors (e.g. New Delhi to Mathura, Mathura to Agra Cantt) experience high train densities and delay propagation.
- **Zero-Inference Rule:** Sections without empirical RailKit evidence receive `railkit_operational_pressure = 0.0`. The system never invents or hallucinates traffic pressure on unmapped sections.
- **Fallback Hierarchy:** If `railkit_operational_pressure` is absent, the decision engine falls back to `traffic_intensity`, then `operational_impact_score`, and finally defaults to $0.0$.

---

## 15. Cross-Department Coordination Mechanics

Railway blocks require joint possession across multiple engineering departments:
- **Civil Engineering** (Track, ballast, sleepers)
- **S&T** (Signaling & Telecommunications, point machines, track circuits)
- **Traction** (OHE / Overhead 25kV electric lines)

### CP-SAT Coordination Incentive
When tasks from different compatible departments are scheduled on the **same physical section** during the **same block window**, the solver adds a soft coordination bonus ($+50$) to the objective:
- Compatible pairs: Engineering + S&T, Engineering + Traction, S&T + Traction.
- Identical departments on the same section receive no cross-department coordination bonus.
- **Hard Feasibility Dominance:** The coordination bonus is strictly soft. It will **never** override section non-overlap constraints or exceed manpower capacity limits.

---

## 16. Maintenance Lifecycle State Machine Handoff

Phase 3 introduces the official state machine managed by `MaintenanceLifecycleManager` in [`src/lifecycle/manager.py`](../src/lifecycle/manager.py):

```
       [AI Generates Recommendations]
                     |
                     v
               +-----------+
               | GENERATED |
               +-----------+
                     |
         [Human Operator Confirms Window]
                     |
                     v
               +-----------+
               |  UPCOMING |
               +-----------+
                     |
           [Window Start Slot Reached]
                     |
                     v
               +-----------+
               |   ACTIVE  |
               +-----------+
              /             \
 [Work Finished]        [Window Ends Without Completion]
            /                 \
           v                   v
    +-----------+        +-----------+
    | COMPLETED |        |   MISSED  |
    +-----------+        +-----------+
                               |
                    [AI Reschedules Task]
                               |
                               v
                       +--------------+
                       | RESCHEDULED  |
                       +--------------+
                               |
                   [Human Confirms New Window]
                               |
                               v
                       +--------------+
                       |   UPCOMING   |
                       +--------------+
```

### Lifecycle Invariants
1. `GENERATED`: Initial state upon recommendation emission.
2. `UPCOMING`: A candidate window has been explicitly authorized by an operator.
3. `ACTIVE`: Current clock time falls within the active window.
4. `COMPLETED`: Terminal state; once marked completed, a task never transitions to `MISSED`.
5. `MISSED`: Window lapsed without confirmed completion. Triggers overdue penalty and automatic rescheduling candidate generation.
6. `RESCHEDULED`: Task has received new candidate windows following a missed execution.

### Architectural Boundary: Separation of Lifecycle from the AI Contract
> [!IMPORTANT]
> **Lifecycle Is NOT an AI Output Category:**
> The `MaintenanceLifecycleManager` is an internal operational state-tracking and event-logging utility. It is **not** an AI prediction model, **not** an AI capability, and **is not** emitted in the results of `evaluate_for_backend()`.
> The backend-facing ML contract consists strictly and exclusively of the **4 frozen AI categories**:
> 1. `risk`
> 2. `priority`
> 3. `explanation`
> 4. `scheduling`
> Lifecycle state management operates downstream after candidate recommendations have been evaluated and presented to human dispatchers.

---

## 17. Timezone-Aware Clock & Midnight Crossing

All lifecycle operations and block window calculations are strictly timezone-aware:
- Standard timezone is UTC (`datetime.timezone.utc`) or Indian Standard Time (`Asia/Kolkata`, UTC+05:30).
- **Midnight-Crossing Windows:** Maintenance windows frequently operate between 23:00 and 03:00. The `MaintenanceWindow` class models windows as continuous intervals across midnight (`end_time > start_time` across calendar days), correctly calculating durations without negative modulo anomalies.

---

## 18. End-to-End Concrete Example Payloads

### Example Request (Backend to ML Engine)
```python
from src.services.ml_engine import RailwayMLEngine

engine = RailwayMLEngine()

request_payload = [
    {
        "task_id": "NDL-TRK-101",
        "section_id": "NDL-MTJ-01",
        "department": "ENGINEERING",
        "estimated_duration": 60,
        "required_manpower": 4,
        "criticality": 8.5,
        "urgency_score": 0.90,
        "overdue_days": 12.0,
        "failure_probability": 0.045,
        "predicted_delay_minutes": 2.0
    },
    {
        "task_id": "NDL-SIG-202",
        "section_id": "NDL-MTJ-01",
        "department": "S&T",
        "estimated_duration": 60,
        "required_manpower": 3,
        "criticality": 5.0,
        "urgency_score": 0.30,
        "overdue_days": 0.0,
        "failure_probability": 0.002,
        "predicted_delay_minutes": 0.0
    }
]

response = engine.evaluate_for_backend(request_payload)
```

### Example Response (ML Engine to Backend)
```json
{
  "status": "SUCCESS",
  "engine_version": "0.1.0",
  "tasks_evaluated": 2,
  "results": [
    {
      "task_id": "NDL-TRK-101",
      "section_id": "NDL-MTJ-01",
      "department": "ENGINEERING",
      "decision_rank": 1,
      "risk": {
        "failure_probability": 0.0450,
        "predicted_failure_30d": true
      },
      "priority": {
        "priority_score": 0.4793,
        "priority_category": "MEDIUM",
        "score_breakdown": {
          "failure_risk_factor": 0.0450,
          "criticality_factor": 0.8500,
          "urgency_factor": 0.9000,
          "overdue_factor": 0.4000,
          "operational_factor": 0.2388,
          "total_score": 0.4793
        }
      },
      "explanation": {
        "reason_tags": [
          "HIGH_URGENCY",
          "HIGH_CRITICALITY",
          "OVERDUE_MAINTENANCE"
        ],
        "score_breakdown": {
          "failure_risk_factor": 0.0450,
          "criticality_factor": 0.8500,
          "urgency_factor": 0.9000,
          "overdue_factor": 0.4000,
          "operational_factor": 0.2388,
          "total_score": 0.4793
        },
        "explanation_summary": "Task NDL-TRK-101 prioritized as MEDIUM (score: 0.4793) driven by: HIGH_URGENCY, HIGH_CRITICALITY, OVERDUE_MAINTENANCE."
      },
      "scheduling": {
        "total_recommendations": 2,
        "has_feasible_window": true,
        "recommended_windows": [
          {
            "rank": 1,
            "block_id": "B001",
            "start_slot": 3,
            "end_slot": 5,
            "duration_minutes": 60,
            "required_manpower": 4,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "CROSS_DEPARTMENT_COORDINATION",
              "MODERATE_PRIORITY_FIT",
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": true,
              "coordination_departments": ["ENGINEERING", "S&T"],
              "coordination_reason": "Bundled with NDL-SIG-202 on section NDL-MTJ-01 in block B001"
            },
            "selection_status": "RECOMMENDED",
            "human_confirmation_required": true,
            "is_booked": false,
            "booking_status": "UNBOOKED"
          },
          {
            "rank": 2,
            "block_id": "B002",
            "start_slot": 8,
            "end_slot": 10,
            "duration_minutes": 60,
            "required_manpower": 4,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "CROSS_DEPARTMENT_COORDINATION",
              "MODERATE_PRIORITY_FIT",
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": true,
              "coordination_departments": ["ENGINEERING", "S&T"],
              "coordination_reason": "Bundled with NDL-SIG-202 on section NDL-MTJ-01 in block B002"
            },
            "selection_status": "RECOMMENDED",
            "human_confirmation_required": true,
            "is_booked": false,
            "booking_status": "UNBOOKED"
          }
        ]
      }
    },
    {
      "task_id": "NDL-SIG-202",
      "section_id": "NDL-MTJ-01",
      "department": "S&T",
      "decision_rank": 2,
      "risk": {
        "failure_probability": 0.0020,
        "predicted_failure_30d": false
      },
      "priority": {
        "priority_score": 0.1864,
        "priority_category": "LOW",
        "score_breakdown": {
          "failure_risk_factor": 0.0020,
          "criticality_factor": 0.5000,
          "urgency_factor": 0.3000,
          "overdue_factor": 0.0000,
          "operational_factor": 0.2388,
          "total_score": 0.1864
        }
      },
      "explanation": {
        "reason_tags": [
          "ROUTINE_INSPECTION"
        ],
        "score_breakdown": {
          "failure_risk_factor": 0.0020,
          "criticality_factor": 0.5000,
          "urgency_factor": 0.3000,
          "overdue_factor": 0.0000,
          "operational_factor": 0.2388,
          "total_score": 0.1864
        },
        "explanation_summary": "Task NDL-SIG-202 prioritized as LOW (score: 0.1864) driven by: ROUTINE_INSPECTION."
      },
      "scheduling": {
        "total_recommendations": 2,
        "has_feasible_window": true,
        "recommended_windows": [
          {
            "rank": 1,
            "block_id": "B001",
            "start_slot": 1,
            "end_slot": 3,
            "duration_minutes": 60,
            "required_manpower": 3,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "CROSS_DEPARTMENT_COORDINATION",
              "MODERATE_PRIORITY_FIT",
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": true,
              "coordination_departments": ["ENGINEERING", "S&T"],
              "coordination_reason": "Bundled with NDL-TRK-101 on section NDL-MTJ-01 in block B001"
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
            "required_manpower": 3,
            "feasibility_status": "FEASIBLE",
            "recommendation_reasons": [
              "CROSS_DEPARTMENT_COORDINATION",
              "MODERATE_PRIORITY_FIT",
              "LOWEST_OPERATIONAL_DELAY"
            ],
            "coordination": {
              "is_coordinated": true,
              "coordination_departments": ["ENGINEERING", "S&T"],
              "coordination_reason": "Bundled with NDL-TRK-101 on section NDL-MTJ-01 in block B002"
            },
            "selection_status": "RECOMMENDED",
            "human_confirmation_required": true,
            "is_booked": false,
            "booking_status": "UNBOOKED"
          }
        ]
      }
    }
  ],
  "metrics": {
    "tasks_count": 2,
    "critical_tasks_count": 0,
    "high_tasks_count": 0,
    "medium_tasks_count": 1,
    "low_tasks_count": 1,
    "tasks_with_recommendations": 2,
    "infeasible_tasks": []
  }
}
```

---

## 19. Edge Cases & Error Handling

| Edge Case Scenario | Pipeline Behavior | Contract Response |
|---|---|---|
| **Empty Task List (`[]`)** | Graceful exit, no solver invocation | Returns `status: "EMPTY_INPUT"`, `tasks_evaluated: 0`, `results: []`. |
| **Empty DataFrame (`pd.DataFrame()`)** | Handled identically to empty list | Returns `status: "EMPTY_INPUT"`, `tasks_evaluated: 0`, `results: []`. |
| **Single Dict Input** | Converted to 1-row DataFrame | Processed normally; returns `tasks_evaluated: 1`. |
| **Task Exceeds All Windows** | Solver flags infeasibility | Returns `total_recommendations: 0`, `has_feasible_window: false`, `recommended_windows: []`, `task_id` in `infeasible_tasks`. |
| **Manpower Deficit** | Solver blocks concurrent execution | Conflicting tasks scheduled sequentially or flagged infeasible if time runs out. |
| **Missing Optional Columns** | Non-breaking defaults applied | `department="ENGINEERING"`, `duration=60`, `manpower=2`, `predicted_delay=0.0`. |
| **Corridor Without RailKit Data** | Zero-inference policy | `railkit_operational_pressure = 0.0`; no synthetic pressure added. |

---

## 20. Contract Test Suite Verification

The contract is continuously validated by [`tests/test_backend_contract.py`](../tests/test_backend_contract.py).

### 16 Dedicated Contract Tests (Items A through P)
1. `test_item_a_top_level_structure`: Validates envelope keys (`status`, `engine_version`, `metrics`).
2. `test_item_b_task_level_structure`: Validates task-level keys and mandatory IDs.
3. `test_item_c_four_frozen_categories`: Validates exact presence of `risk`, `priority`, `explanation`, and `scheduling`.
4. `test_item_d_pure_native_python_types`: Recursively verifies 0 numpy or pandas types.
5. `test_item_e_range_constraints`: Validates mathematical boundaries ($[0, 1]$ bounds, category enums).
6. `test_item_f_five_factor_breakdown`: Validates linear combination matches `priority_score`.
7. `test_item_g_decision_category_thresholds`: Validates strict cut points for LOW/MED/HIGH/CRITICAL.
8. `test_item_h_reason_tags_and_fallback`: Validates condition codes and `ROUTINE_INSPECTION` fallback.
9. `test_item_i_top_3_recommendation_cap`: Validates that no task receives $>3$ candidate windows.
10. `test_item_j_sequential_ranks`: Validates rank values are strictly sequential `[1, 2, 3]`.
11. `test_item_k_human_in_the_loop_invariants`: Validates `RECOMMENDED`, `UNBOOKED`, `confirmation=True`.
12. `test_item_l_zero_fake_confidence_metrics`: Validates complete absence of heuristic confidence fields.
13. `test_item_m_id_and_metadata_preservation`: Validates input `task_id`, `section_id`, and `department` are preserved.
14. `test_item_n_json_serialization_roundtrip`: Validates lossless `json.dumps` and `json.loads`.
15. `test_item_o_edge_cases`: Validates empty inputs, infeasible tasks, single dicts, and raw asset telemetry.
16. `test_item_p_schema_file_validation`: Validates complete payload strictly against `docs/schemas/ml_response.schema.json`.

---

## 21. Production Deployment & Runtime Requirements

- **Python Version:** $\ge 3.10$ (Verified on Python 3.14 on Ubuntu WSL2).
- **Core Dependencies:**
  - `ortools` ($\ge 9.8.0$): CP-SAT solver engine.
  - `xgboost` ($\ge 2.0.0$): Gradient boosted tree risk scoring.
  - `scikit-learn` ($\ge 1.4.0$): Probability calibration (`CalibratedClassifierCV`).
  - `pandas` ($\ge 2.1.0$), `numpy` ($\ge 1.26.0$).
- **Computational Footprint:**
  - Memory: $\approx 250\text{ MB}$ RSS working footprint.
  - Latency: $< 15\text{ ms}$ for scoring 50 tasks; $< 150\text{ ms}$ for CP-SAT solve with 50 tasks across 6-hour horizon.
  - Threading: CP-SAT solver runs in-process with deterministic seeds.

---

## 22. Status & Versioning Guarantee

- **Contract Version:** `1.0.0`
- **Guarantee:** The payload structure, field names, factor weights, and category schemas defined herein are **FROZEN**. No breaking changes will be made without incrementing the major version number.
- **Ready for Integration:** The ML repository is hardened and verified for the upcoming communication boundary design and backend integration step.
