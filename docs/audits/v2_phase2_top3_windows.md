# V2 Phase 2 Audit Report: Top-3 Block Window Recommendations & Coordination

## 1. Existing Optimizer Behavior Before Phase 2

Prior to Phase 2, the CP-SAT planning layer in `BlockOptimizer` (`src/optimization/block_optimizer.py`) operated under a single-schedule paradigm:
- Given a set of maintenance tasks and block windows, CP-SAT solved a single global optimization problem.
- It produced **at most ONE assigned block window** per task.
- Tasks not selected in the single optimal schedule were excluded with no alternatives provided.
- The output dataframe had columns `plan_sequence`, `task_id`, `section_id`, `department`, `block_id`, `start_slot`, `end_slot`, `duration_minutes`, `required_manpower`, `maintenance_decision_score`, `predicted_delay_minutes`.
- Cross-department coordination bonus (`coordination_bonus = 50`) existed as an unused parameter in `MultiHorizonPlanner`, but was not wired into the single-window optimization objective in `BlockOptimizer`.
- Output was structured as an auto-booked/scheduled plan rather than a recommendation set for human-in-the-loop decision making.

---

## 2. New Recommendation Architecture

In Phase 2, the planning layer was enhanced with `optimize_with_recommendations()` in `BlockOptimizer`, alongside extensions in `PlanningService` and `RailwayMLEngine`:
- **Role Definition**: CP-SAT acts strictly as a **decision-support recommendation engine**. It never mutates task records or automatically confirms maintenance slots.
- **Top-3 Window Capacity**: For each task in the worklist, the engine evaluates available block windows and returns **at most 3 ranked recommendations** (`rank 1`, `rank 2`, `rank 3`).
- **Flexible Result Cardinality**:
  - If $\ge 3$ feasible windows exist: returns the top 3 ranked candidates.
  - If 2 feasible windows exist: returns exactly 2 candidates (`rank 1`, `rank 2`).
  - If 1 feasible window exists: returns exactly 1 candidate (`rank 1`).
  - If 0 feasible windows exist: returns an empty list `[]` and documents the task and reason in `infeasible_tasks`.
- **Output Interfaces**:
  - `recommendations`: Dict mapping `task_id -> List[Dict]`.
  - `recommendations_df`: Flat Pandas DataFrame representing all recommendations for convenient downstream tabular consumption.
  - `infeasible_tasks`: List of dicts documenting unschedulable tasks and physical constraints violated.
  - `metrics`: Summary statistics (`total_tasks`, `tasks_with_recommendations`, `total_recommendations`, `infeasible_task_count`).

---

## 3. Candidate Generation Approach

Candidate block windows are generated without fabrication or synthetic duplication:
1. **Physical Feasibility Filter**: For each task $t$ with duration $d_t$ slots and required manpower $m_t$:
   - A block window $w = [S_w, E_w]$ is physically feasible if $d_t \le (E_w - S_w)$ and $m_t \le \text{max\_manpower}$.
   - If either condition is violated across all windows, the task is marked infeasible with explicit reasons.
2. **CP-SAT Window Placement & Joint Solves**:
   - For each block window $w$, CP-SAT solves the exact discrete-slot placement of all eligible tasks.
   - Computes optimal start slot $s_{t, w} \in [S_w, E_w - d_t]$ and end slot $e_{t, w} = s_{t, w} + d_t$.
   - Evaluates multi-task co-location, ensuring section non-overlap and cumulative manpower limits are strictly honored.
3. **Standalone Fallback Evaluation**:
   - If task $t$ physically fits in window $w$ but was preempted in joint scheduling due to competition with a higher-priority task on the same section or manpower saturation, a standalone placement in window $w$ is evaluated at its pure objective score (priority minus delay penalty, without coordination bonus).
   - This ensures the human operator can still see window $w$ as a valid physical option for task $t$ if they choose not to execute the competing task.

---

## 4. Ranking Methodology

Candidates for each task are ranked using the verified objective terms:
$$\text{Score}(t, w) = \text{priority}_t - \text{delay\_penalty}_t + \text{coordination\_bonus}(t, w)$$
where:
- $\text{priority}_t = \text{round}(\text{maintenance\_decision\_score}_t \times 1000)$
- $\text{delay\_penalty}_t = \text{round}(\text{predicted\_delay\_minutes}_t \times 10)$
- $\text{coordination\_bonus}(t, w) = 50$ if coordinated with another department on the same section in window $w$ (0 otherwise).
*(Note: An arbitrary -100 preemption penalty present in early drafting was audited and removed, as it distorted delay trade-offs across competing alternative windows).*

**Deterministic Tie-Breaking**:
1. Highest `Score(t, w)` descending.
2. Earliest `start_slot` ascending.
3. Lexicographic `block_id` ascending.

The sorted candidates are truncated to at most `max_recommendations` (3) and labeled with sequential integer ranks `1`, `2`, `3`.

---

## 5. Hard Constraints Preserved

Every returned recommendation strictly obeys all railway safety and physical constraints:
1. **Section Non-Overlap**: Tasks on the same physical section cannot occupy the track at the same time slot:
   $$\text{model.AddNoOverlap}([\text{task\_intervals on same section}])$$
   If two tasks on the same section are both placed in window $w$, they must run sequentially ($d_1 + d_2 \le E_w - S_w$).
2. **Manpower Capacity**: Concurrent manpower across all active tasks at any 30-minute slot $s$ cannot exceed `max_manpower` (default 12):
   $$\text{model.AddCumulative}(\text{intervals}, \text{manpowers}, \text{max\_manpower})$$
3. **Block Window Containment**: Every task must fit wholly inside its window boundaries:
   $$\text{start\_slot} \ge S_w \quad \text{and} \quad \text{end\_slot} \le E_w$$
4. **Duration Fit**: Task duration is never truncated or squeezed to force fit a block.

---

## 6. Coordination Preference

Phase 2 implements a soft optimization preference for cross-department coordination:
- **Scenario**: Two or more tasks belong to different departments (`ENGINEERING`, `S&T`, `TRACTION`) on the same section and can be safely accommodated in the same block window.
- **Formulation**:
  ```python
  c_var = model.NewBoolVar(f"coord_{w_id}_{idx1}_{idx2}")
  model.Add(c_var <= task_sel[idx1])
  model.Add(c_var <= task_sel[idx2])
  objective_terms.append(self.coordination_bonus * c_var)
  ```
- **Metadata Exposing**: When coordination occurs, recommendations expose:
  - `is_coordinated: True`
  - `coordinated_task_ids: ["TASK-002"]`
  - `coordination_departments: ["ENGINEERING", "S&T"]`
  - `coordination_reason: "Bundled with TASK-002 on section NDL-MTJ-01 in block B001"`
- **Strict Precedence of Safety**: Coordination is a **soft bonus**. It is never forced when section non-overlap or manpower limits would be violated.

---

## 7. Human-in-the-Loop Semantics

To prevent erroneous autonomous slot bookings, every recommendation data structure carries explicit human control attributes:
- `selection_status`: `"RECOMMENDED"` (never `"BOOKED"` or `"ACTIVE"`).
- `human_confirmation_required`: `True`.
- `is_booked`: `False`.
- `booking_status`: `"UNBOOKED"`.
- CP-SAT does not update the database, trigger work orders, or modify maintenance state machines.

---

## 8. Edge Cases Handled and Verified

| # | Edge Case Scenario | Handled Behavior | Status |
|---|---|---|:---:|
| 1 | Zero feasible windows | Empty list `[]` returned; documented in `infeasible_tasks` with reason | Verified |
| 2 | Exactly one feasible window | Exactly 1 recommendation returned (`rank 1`) | Verified |
| 3 | Exactly two feasible windows | Exactly 2 recommendations returned (`rank 1`, `rank 2`) | Verified |
| 4 | More than three feasible windows | Top 3 recommendations returned (`rank 1`, `rank 2`, `rank 3`) | Verified |
| 5 | Identical candidate solutions | Deduplication by `block_id`; no duplicate windows returned | Verified |
| 6 | Same-section conflicting tasks | `AddNoOverlap` prevents overlap; only sequential execution permitted | Verified |
| 7 | Different-section concurrency | Tasks execute in parallel at same slot if manpower permits | Verified |
| 8 | Manpower saturation | Concurrent tasks constrained by `AddCumulative`; peak $\le$ max_manpower | Verified |
| 9 | Duration longer than window | Filtered out as infeasible; listed in `infeasible_tasks` | Verified |
| 10 | Multiple departments in same section | Coordination bonus awarded; full metadata exposed | Verified |
| 11 | Coordination prevented by manpower | Hard cumulative constraint overrides coordination bonus | Verified |
| 12 | Coordination prevented by section duration | Hard `NoOverlap` constraint overrides coordination bonus | Verified |
| 13 | Empty worklist | Returns empty recommendation dictionary and empty DataFrame | Verified |
| 14 | Missing required columns | Raises `ValueError` with list of missing columns | Verified |
| 15 | Deterministic repeated execution | Identical inputs produce 100% identical outputs | Verified |

---

## 9. Test Results

- **New Phase 2 Test Suite**: `tests/test_top3_recommendations.py` — **37 passed in 3.14s** (expanded with 15 audit tests: 6 ranking invariants, 7 coordination scenarios, independent validator, and preemption correctness).
- **Full Test Suite**: **112 passed in 5.41s** across 6 test modules:
  - `tests/test_top3_recommendations.py`: 37 passed
  - `tests/test_planning_regression.py`: 33 passed
  - `tests/test_failure_predictor.py`: 12 passed
  - `tests/test_multi_horizon_and_engine.py`: 10 passed
  - `tests/test_section_evidence.py`: 20 passed
  - Overall pass rate: 100% (112/112 tests green).

---

## 10. Known Limitations

1. **Horizon Scope**: `optimize_with_recommendations()` operates on discrete block windows within a configurable planning horizon (default 6 hours, configurable up to multi-day).
2. **Department Granularity**: Coordination logic currently pairs tasks on the exact same `section_id`. Cross-corridor or adjacent-station yard coordination can be considered in future extensions.
3. **No Dynamic Delay Feedback Loop**: Recommendations reflect static `predicted_delay_minutes` provided at invocation time. Dynamic rescheduling during active execution belongs to downstream operational triggers.

---

## 11. Explicit Confirmation of Strict Scope Boundaries

As required by project instructions:
- **Phase 3 was NOT implemented**:
  - No real-time clock tickers or background workers were added.
  - No state machine (`UPCOMING` / `ACTIVE` / `COMPLETED` / `MISSED`) was created.
  - No automatic missed-window detection or priority escalation logic was implemented.
  - No Celery tasks, background threads, or live pollers were created.
- **System Boundaries Preserved**:
  - No frontend code was touched.
  - No backend Django models or database migrations were run.
  - No API or MCP tools were altered.
  - No Git operations (commit, push, checkout, branch, merge) were executed.
- Phase 2 implementation is strictly confined to local Python algorithmic planning modules in `src/optimization/`, `src/services/`, and tests in `tests/`.
