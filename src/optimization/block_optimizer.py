import math

import pandas as pd
from ortools.sat.python import cp_model


class BlockOptimizer:
    """
    CP-SAT maintenance block optimizer.

    Preserves the core constraints from notebook 23 while making
    planning parameters configurable.

    The optimizer decides which maintenance tasks to select and
    where to place them inside available block windows.
    """

    SLOT_MINUTES = 30

    def __init__(
        self,
        planning_hours=6,
        max_manpower=12,
        max_time_seconds=30,
        num_search_workers=8,
        score_scale=1000,
        delay_penalty_scale=10,
        coordination_bonus=50,
    ):
        if planning_hours <= 0:
            raise ValueError("planning_hours must be positive")

        if max_manpower <= 0:
            raise ValueError("max_manpower must be positive")

        self.planning_hours = planning_hours
        self.max_manpower = max_manpower
        self.max_time_seconds = max_time_seconds
        self.num_search_workers = num_search_workers
        self.score_scale = score_scale
        self.delay_penalty_scale = delay_penalty_scale
        self.coordination_bonus = coordination_bonus

        self.num_slots = (
            planning_hours * 60
        ) // self.SLOT_MINUTES

    @staticmethod
    def _numeric(series, default=0.0):
        return pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(default)

    def optimize(
        self,
        tasks: pd.DataFrame,
        block_windows=None,
    ) -> pd.DataFrame:
        """
        Optimize a maintenance task dataframe.

        Required columns:
            task_id
            section_id
            estimated_duration
            required_manpower
            maintenance_decision_score
            predicted_delay_minutes

        block_windows format:

            [
                {
                    "block_id": "B001",
                    "start_slot": 1,
                    "end_slot": 5,
                },
                ...
            ]

        start_slot is inclusive.
        end_slot is exclusive.
        """

        if not isinstance(tasks, pd.DataFrame):
            raise TypeError("tasks must be a pandas DataFrame")

        required_columns = [
            "task_id",
            "section_id",
            "estimated_duration",
            "required_manpower",
            "maintenance_decision_score",
            "predicted_delay_minutes",
        ]

        missing = [
            column
            for column in required_columns
            if column not in tasks.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if tasks.empty:
            return pd.DataFrame(
                columns=[
                    "task_id",
                    "section_id",
                    "department",
                    "block_id",
                    "start_slot",
                    "end_slot",
                    "duration_minutes",
                    "required_manpower",
                    "maintenance_decision_score",
                    "predicted_delay_minutes",
                ]
            )

        if block_windows is None:
            block_windows = [
                {
                    "block_id": "B001",
                    "start_slot": 1,
                    "end_slot": 5,
                },
                {
                    "block_id": "B002",
                    "start_slot": 6,
                    "end_slot": 10,
                },
            ]

        normalized_windows = self._validate_block_windows(
            block_windows
        )

        optimizer_tasks = tasks.copy().reset_index(drop=True)

        optimizer_tasks["estimated_duration"] = self._numeric(
            optimizer_tasks["estimated_duration"]
        ).clip(lower=0)

        optimizer_tasks["required_manpower"] = self._numeric(
            optimizer_tasks["required_manpower"]
        ).clip(lower=0)

        optimizer_tasks["maintenance_decision_score"] = (
            self._numeric(
                optimizer_tasks[
                    "maintenance_decision_score"
                ]
            )
            .clip(0, 1)
        )

        optimizer_tasks["predicted_delay_minutes"] = (
            self._numeric(
                optimizer_tasks[
                    "predicted_delay_minutes"
                ]
            )
            .clip(lower=0)
        )

        optimizer_tasks["duration_slots"] = (
            optimizer_tasks["estimated_duration"]
            .apply(
                lambda value: max(
                    1,
                    math.ceil(
                        value / self.SLOT_MINUTES
                    ),
                )
            )
            .astype(int)
        )

        model = cp_model.CpModel()

        task_selected = {}
        task_start = {}
        task_end = {}
        task_intervals = {}
        manpower_intervals = {}
        manpower_demands = {}

        for i, row in optimizer_tasks.iterrows():
            duration_slots = int(row["duration_slots"])

            task_selected[i] = model.NewBoolVar(
                f"task_selected_{i}"
            )

            task_start[i] = model.NewIntVar(
                0,
                self.num_slots - 1,
                f"task_start_{i}",
            )

            task_end[i] = model.NewIntVar(
                0,
                self.num_slots,
                f"task_end_{i}",
            )

            model.Add(
                task_end[i]
                == task_start[i] + duration_slots
            )

            model.Add(
                task_end[i] <= self.num_slots
            ).OnlyEnforceIf(task_selected[i])

            allowed_starts = self._allowed_starts(
                duration_slots,
                normalized_windows,
            )

            if allowed_starts:
                model.AddAllowedAssignments(
                    [task_start[i]],
                    [[slot] for slot in allowed_starts],
                ).OnlyEnforceIf(task_selected[i])

            else:
                model.Add(
                    task_selected[i] == 0
                )

            interval = model.NewOptionalIntervalVar(
                task_start[i],
                duration_slots,
                task_end[i],
                task_selected[i],
                f"task_interval_{i}",
            )

            task_intervals[i] = interval

            manpower = max(
                0,
                int(
                    math.ceil(
                        row["required_manpower"]
                    )
                ),
            )

            manpower_intervals[i] = interval
            manpower_demands[i] = manpower

        # Preserve manpower capacity.
        if manpower_intervals:
            model.AddCumulative(
                list(manpower_intervals.values()),
                list(manpower_demands.values()),
                self.max_manpower,
            )

        # Preserve section-level non-overlap.
        for section_id, group in optimizer_tasks.groupby(
            "section_id"
        ):
            task_ids = group.index.tolist()

            section_intervals = [
                task_intervals[i]
                for i in task_ids
            ]

            if len(section_intervals) > 1:
                model.AddNoOverlap(
                    section_intervals
                )

        # Existing objective:
        #
        #   priority = decision_score * 1000
        #   delay_penalty = predicted_delay * 10
        #   coefficient = priority - delay_penalty
        #
        # Keep this behavior explicit and configurable.
        objective_terms = []

        for i, row in optimizer_tasks.iterrows():
            priority = round(
                float(
                    row[
                        "maintenance_decision_score"
                    ]
                )
                * self.score_scale
            )

            delay_penalty = round(
                float(
                    row[
                        "predicted_delay_minutes"
                    ]
                )
                * self.delay_penalty_scale
            )

            coefficient = (
                priority - delay_penalty
            )

            objective_terms.append(
                coefficient * task_selected[i]
            )

        model.Maximize(
            sum(objective_terms)
        )

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = (
            self.max_time_seconds
        )

        solver.parameters.num_search_workers = (
            self.num_search_workers
        )

        status = solver.Solve(model)

        if status not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            return pd.DataFrame(
                columns=[
                    "task_id",
                    "section_id",
                    "department",
                    "block_id",
                    "start_slot",
                    "end_slot",
                    "duration_minutes",
                    "required_manpower",
                    "maintenance_decision_score",
                    "predicted_delay_minutes",
                ]
            )

        selected_rows = []

        for i, row in optimizer_tasks.iterrows():
            if not solver.BooleanValue(
                task_selected[i]
            ):
                continue

            start_slot = solver.Value(
                task_start[i]
            )

            end_slot = solver.Value(
                task_end[i]
            )

            block_id = self._find_block(
                start_slot,
                end_slot,
                normalized_windows,
            )

            selected_rows.append(
                {
                    "task_id": row["task_id"],
                    "section_id": row["section_id"],
                    "department": row.get(
                        "department",
                        None,
                    ),
                    "block_id": block_id,
                    "start_slot": start_slot,
                    "end_slot": end_slot,
                    "duration_minutes": (
                        end_slot - start_slot
                    )
                    * self.SLOT_MINUTES,
                    "required_manpower": row[
                        "required_manpower"
                    ],
                    "maintenance_decision_score": row[
                        "maintenance_decision_score"
                    ],
                    "predicted_delay_minutes": row[
                        "predicted_delay_minutes"
                    ],
                }
            )

        result = pd.DataFrame(selected_rows)

        if not result.empty:
            result = result.sort_values(
                [
                    "block_id",
                    "start_slot",
                    "section_id",
                ]
            ).reset_index(drop=True)

            result.insert(
                0,
                "plan_sequence",
                range(1, len(result) + 1),
            )

        return result

    def optimize_with_recommendations(
        self,
        tasks: pd.DataFrame,
        block_windows=None,
        max_recommendations: int = 3,
    ) -> dict:
        """
        Generate ranked feasible candidate block window recommendations per maintenance task.

        Phase 2 Semantics:
        CP-SAT is strictly a recommendation engine. It does NOT automatically book,
        confirm, or mutate maintenance task state. Final slot selection is made by
        the human operator.

        Returns at most max_recommendations (default 3) per task, labeled rank 1, 2, 3.
        If only 2 feasible windows exist, returns 2.
        If only 1 exists, returns 1.
        If 0 exist, returns an empty list for that task and records it in infeasible_tasks.
        Never fabricates or duplicates candidate windows.

        Cross-department coordination:
        Soft optimization preference rewarding same-section co-location of compatible
        tasks from different departments (ENGINEERING, S&T, TRACTION) in the same block window,
        without violating hard safety or capacity constraints.
        """
        if not isinstance(tasks, pd.DataFrame):
            raise TypeError("tasks must be a pandas DataFrame")

        if max_recommendations <= 0:
            raise ValueError("max_recommendations must be positive")

        if tasks.empty:
            return {
                "recommendations": {},
                "recommendations_df": pd.DataFrame(columns=[
                    "task_id", "section_id", "department", "rank", "block_id",
                    "start_slot", "end_slot", "duration_minutes", "required_manpower",
                    "maintenance_decision_score", "predicted_delay_minutes",
                    "feasibility_status", "recommendation_reasons", "is_coordinated",
                    "coordinated_task_ids", "coordination_departments", "coordination_reason",
                    "selection_status", "human_confirmation_required", "is_booked", "booking_status"
                ]),
                "infeasible_tasks": [],
                "metrics": {
                    "total_tasks": 0,
                    "tasks_with_recommendations": 0,
                    "total_recommendations": 0,
                    "infeasible_task_count": 0,
                }
            }

        required_columns = [
            "task_id",
            "section_id",
            "estimated_duration",
            "required_manpower",
            "maintenance_decision_score",
            "predicted_delay_minutes",
        ]

        missing = [
            column
            for column in required_columns
            if column not in tasks.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        if block_windows is None:
            block_windows = [
                {
                    "block_id": "B001",
                    "start_slot": 1,
                    "end_slot": 5,
                },
                {
                    "block_id": "B002",
                    "start_slot": 6,
                    "end_slot": 10,
                },
            ]

        normalized_windows = self._validate_block_windows(
            block_windows
        )

        optimizer_tasks = tasks.copy().reset_index(drop=True)

        optimizer_tasks["estimated_duration"] = self._numeric(
            optimizer_tasks["estimated_duration"]
        ).clip(lower=0)

        optimizer_tasks["required_manpower"] = self._numeric(
            optimizer_tasks["required_manpower"]
        ).clip(lower=0)

        optimizer_tasks["maintenance_decision_score"] = (
            self._numeric(
                optimizer_tasks["maintenance_decision_score"]
            ).clip(0, 1)
        )

        optimizer_tasks["predicted_delay_minutes"] = (
            self._numeric(
                optimizer_tasks["predicted_delay_minutes"]
            ).clip(lower=0)
        )

        optimizer_tasks["duration_slots"] = (
            optimizer_tasks["estimated_duration"]
            .apply(
                lambda value: max(
                    1,
                    math.ceil(value / self.SLOT_MINUTES),
                )
            )
            .astype(int)
        )

        optimizer_tasks["manpower_int"] = (
            optimizer_tasks["required_manpower"]
            .apply(
                lambda value: max(
                    0,
                    int(math.ceil(value)),
                )
            )
            .astype(int)
        )

        recommendations_by_task = {
            row["task_id"]: []
            for _, row in optimizer_tasks.iterrows()
        }
        infeasible_tasks = []
        task_candidate_windows = {}

        for idx, row in optimizer_tasks.iterrows():
            t_id = row["task_id"]
            d_slots = int(row["duration_slots"])
            m_req = int(row["manpower_int"])

            if m_req > self.max_manpower:
                infeasible_tasks.append({
                    "task_id": t_id,
                    "section_id": row["section_id"],
                    "reason": (
                        f"Required manpower ({m_req}) exceeds "
                        f"maximum available capacity ({self.max_manpower})"
                    ),
                })
                task_candidate_windows[t_id] = []
                continue

            valid_windows = [
                w
                for w in normalized_windows
                if d_slots <= (w["end_slot"] - w["start_slot"])
            ]

            if not valid_windows:
                infeasible_tasks.append({
                    "task_id": t_id,
                    "section_id": row["section_id"],
                    "reason": (
                        f"Task duration ({int(row['estimated_duration'])} min / "
                        f"{d_slots} slots) exceeds all available block windows"
                    ),
                })
                task_candidate_windows[t_id] = []
            else:
                task_candidate_windows[t_id] = valid_windows

        window_placements = {
            w["block_id"]: {}
            for w in normalized_windows
        }

        for w in normalized_windows:
            w_id = w["block_id"]
            w_start = int(w["start_slot"])
            w_end = int(w["end_slot"])

            fitting_indices = [
                idx
                for idx, r in optimizer_tasks.iterrows()
                if any(
                    cand["block_id"] == w_id
                    for cand in task_candidate_windows.get(r["task_id"], [])
                )
            ]

            if not fitting_indices:
                continue

            sub_tasks = optimizer_tasks.loc[fitting_indices].copy()

            model = cp_model.CpModel()
            task_sel = {}
            task_st = {}
            task_en = {}
            task_inv = {}
            mp_inv = {}
            mp_dem = {}

            for idx in fitting_indices:
                r = optimizer_tasks.loc[idx]
                d_slots = int(r["duration_slots"])

                task_sel[idx] = model.NewBoolVar(f"sel_{w_id}_{idx}")
                task_st[idx] = model.NewIntVar(
                    w_start,
                    w_end - d_slots,
                    f"st_{w_id}_{idx}",
                )
                task_en[idx] = model.NewIntVar(
                    w_start + d_slots,
                    w_end,
                    f"en_{w_id}_{idx}",
                )
                model.Add(task_en[idx] == task_st[idx] + d_slots)

                inv = model.NewOptionalIntervalVar(
                    task_st[idx],
                    d_slots,
                    task_en[idx],
                    task_sel[idx],
                    f"inv_{w_id}_{idx}",
                )
                task_inv[idx] = inv
                mp_inv[idx] = inv
                mp_dem[idx] = int(r["manpower_int"])

            # 1. Section non-overlap in window
            for sec_id, grp in sub_tasks.groupby("section_id"):
                sec_idxs = grp.index.tolist()
                if len(sec_idxs) > 1:
                    model.AddNoOverlap([task_inv[i] for i in sec_idxs])

            # 2. Cumulative manpower in window
            if mp_inv:
                model.AddCumulative(
                    list(mp_inv.values()),
                    list(mp_dem.values()),
                    self.max_manpower,
                )

            # 3. Cross-department coordination bonus
            coord_vars = {}
            for i_pos in range(len(fitting_indices)):
                idx1 = fitting_indices[i_pos]
                for j_pos in range(i_pos + 1, len(fitting_indices)):
                    idx2 = fitting_indices[j_pos]
                    r1 = optimizer_tasks.loc[idx1]
                    r2 = optimizer_tasks.loc[idx2]

                    sec1 = r1["section_id"]
                    sec2 = r2["section_id"]
                    dept1 = str(r1.get("department", "") or "").strip().upper()
                    dept2 = str(r2.get("department", "") or "").strip().upper()

                    if (
                        sec1 == sec2
                        and dept1
                        and dept2
                        and dept1 != dept2
                        and dept1 != "UNKNOWN"
                        and dept2 != "UNKNOWN"
                    ):
                        c_var = model.NewBoolVar(f"coord_{w_id}_{idx1}_{idx2}")
                        model.Add(c_var <= task_sel[idx1])
                        model.Add(c_var <= task_sel[idx2])
                        coord_vars[(idx1, idx2)] = c_var

            # 4. Objective
            obj_terms = []
            for idx in fitting_indices:
                r = optimizer_tasks.loc[idx]
                p = round(float(r["maintenance_decision_score"]) * self.score_scale)
                d = round(float(r["predicted_delay_minutes"]) * self.delay_penalty_scale)
                coeff = p - d
                obj_terms.append(coeff * task_sel[idx])

            for (idx1, idx2), c_var in coord_vars.items():
                obj_terms.append(self.coordination_bonus * c_var)

            model.Maximize(sum(obj_terms))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = min(self.max_time_seconds, 10)
            solver.parameters.num_search_workers = self.num_search_workers

            status = solver.Solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                for idx in fitting_indices:
                    r = optimizer_tasks.loc[idx]
                    t_id = r["task_id"]
                    if solver.BooleanValue(task_sel[idx]):
                        s_val = solver.Value(task_st[idx])
                        e_val = solver.Value(task_en[idx])

                        coord_with = []
                        for (i1, i2), c_var in coord_vars.items():
                            if solver.BooleanValue(c_var):
                                if i1 == idx:
                                    coord_with.append(optimizer_tasks.loc[i2, "task_id"])
                                elif i2 == idx:
                                    coord_with.append(optimizer_tasks.loc[i1, "task_id"])

                        window_placements[w_id][t_id] = {
                            "start_slot": s_val,
                            "end_slot": e_val,
                            "is_coordinated": len(coord_with) > 0,
                            "coordinated_task_ids": coord_with,
                            "objective_score": (
                                round(float(r["maintenance_decision_score"]) * self.score_scale)
                                - round(float(r["predicted_delay_minutes"]) * self.delay_penalty_scale)
                                + (self.coordination_bonus if len(coord_with) > 0 else 0)
                            ),
                        }

        all_recommendations_rows = []

        for idx, row in optimizer_tasks.iterrows():
            t_id = row["task_id"]
            sec_id = row["section_id"]
            dept = row.get("department", None)
            valid_windows = task_candidate_windows.get(t_id, [])

            if not valid_windows:
                continue

            candidates_for_task = []
            for w in valid_windows:
                w_id = w["block_id"]
                placement = window_placements[w_id].get(t_id)

                if placement:
                    s_slot = placement["start_slot"]
                    e_slot = placement["end_slot"]
                    is_coord = placement["is_coordinated"]
                    coord_ids = placement["coordinated_task_ids"]
                    obj_score = placement["objective_score"]
                else:
                    d_slots = int(row["duration_slots"])
                    s_slot = int(w["start_slot"])
                    e_slot = int(w["start_slot"]) + d_slots
                    is_coord = False
                    coord_ids = []
                    obj_score = (
                        round(float(row["maintenance_decision_score"]) * self.score_scale)
                        - round(float(row["predicted_delay_minutes"]) * self.delay_penalty_scale)
                    )

                coord_depts = []
                coord_reason = None
                if is_coord:
                    matched_rows = optimizer_tasks[optimizer_tasks["task_id"].isin(coord_ids)]
                    coord_depts = sorted(list(set(
                        [dept] + [str(d).strip().upper() for d in matched_rows.get("department", []) if d]
                    )))
                    coord_reason = f"Bundled with {', '.join(coord_ids)} on section {sec_id} in block {w_id}"

                reasons = []
                if is_coord:
                    reasons.append("CROSS_DEPARTMENT_COORDINATION")
                if float(row["maintenance_decision_score"]) >= 0.7:
                    reasons.append("OPTIMAL_PRIORITY_FIT")
                elif float(row["maintenance_decision_score"]) >= 0.4:
                    reasons.append("MODERATE_PRIORITY_FIT")
                if float(row["predicted_delay_minutes"]) == 0:
                    reasons.append("LOWEST_OPERATIONAL_DELAY")
                elif float(row["predicted_delay_minutes"]) <= 5:
                    reasons.append("LOW_OPERATIONAL_DELAY")
                if not reasons:
                    reasons.append("FEASIBLE_MAINTENANCE_WINDOW")

                candidates_for_task.append({
                    "task_id": t_id,
                    "section_id": sec_id,
                    "department": dept,
                    "block_id": w_id,
                    "start_slot": s_slot,
                    "end_slot": e_slot,
                    "duration_minutes": (e_slot - s_slot) * self.SLOT_MINUTES,
                    "required_manpower": row["required_manpower"],
                    "maintenance_decision_score": row["maintenance_decision_score"],
                    "predicted_delay_minutes": row["predicted_delay_minutes"],
                    "feasibility_status": "FEASIBLE",
                    "recommendation_reasons": reasons,
                    "coordination": {
                        "is_coordinated": is_coord,
                        "coordinated_task_ids": coord_ids,
                        "coordination_departments": coord_depts,
                        "coordination_reason": coord_reason,
                    },
                    "selection_status": "RECOMMENDED",
                    "human_confirmation_required": True,
                    "is_booked": False,
                    "booking_status": "UNBOOKED",
                    "_score": obj_score,
                    "_start": s_slot,
                })

            candidates_for_task.sort(
                key=lambda c: (-c["_score"], c["_start"], c["block_id"])
            )

            top_candidates = candidates_for_task[:max_recommendations]

            for rank_idx, cand in enumerate(top_candidates, start=1):
                cand["rank"] = rank_idx
                del cand["_score"]
                del cand["_start"]
                recommendations_by_task[t_id].append(cand)

                flat_row = cand.copy()
                coord_info = flat_row.pop("coordination")
                flat_row["is_coordinated"] = coord_info["is_coordinated"]
                flat_row["coordinated_task_ids"] = coord_info["coordinated_task_ids"]
                flat_row["coordination_departments"] = coord_info["coordination_departments"]
                flat_row["coordination_reason"] = coord_info["coordination_reason"]
                all_recommendations_rows.append(flat_row)

        rec_df = pd.DataFrame(all_recommendations_rows)

        return {
            "recommendations": recommendations_by_task,
            "recommendations_df": rec_df,
            "infeasible_tasks": infeasible_tasks,
            "metrics": {
                "total_tasks": len(optimizer_tasks),
                "tasks_with_recommendations": sum(
                    1 for recs in recommendations_by_task.values() if len(recs) > 0
                ),
                "total_recommendations": len(all_recommendations_rows),
                "infeasible_task_count": len(infeasible_tasks),
            },
        }

    def _validate_block_windows(
        self,
        block_windows,
    ):
        normalized = []

        for window in block_windows:
            if not isinstance(window, dict):
                raise TypeError(
                    "Each block window must be a dictionary"
                )

            required = [
                "block_id",
                "start_slot",
                "end_slot",
            ]

            missing = [
                key
                for key in required
                if key not in window
            ]

            if missing:
                raise ValueError(
                    f"Block window missing: {missing}"
                )

            start = int(window["start_slot"])
            end = int(window["end_slot"])

            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid block window: {window}"
                )

            if end > self.num_slots:
                raise ValueError(
                    f"Block window exceeds planning horizon: "
                    f"{window}"
                )

            normalized.append(
                {
                    "block_id": str(
                        window["block_id"]
                    ),
                    "start_slot": start,
                    "end_slot": end,
                }
            )

        if not normalized:
            raise ValueError(
                "At least one block window is required"
            )

        return normalized

    @staticmethod
    def _allowed_starts(
        duration_slots,
        block_windows,
    ):
        starts = set()

        for window in block_windows:
            for start in range(
                window["start_slot"],
                window["end_slot"]
                - duration_slots
                + 1,
            ):
                starts.add(start)

        return sorted(starts)

    @staticmethod
    def _find_block(
        start_slot,
        end_slot,
        block_windows,
    ):
        for window in block_windows:
            if (
                start_slot
                >= window["start_slot"]
                and end_slot
                <= window["end_slot"]
            ):
                return window["block_id"]

        return None
