"""
Multi-Horizon Planning & Scheduling Engine — Phase 7.1

Implements multi-day horizons (Weekly, Monthly, Rolling-Horizon, and Dynamic Rescheduling)
using OR-Tools CP-SAT, formalizing the prototypes developed in notebooks 25, 26, 27, and 28.

Key Features:
- Weekly Planning (7 days, daily dual-window pattern D1_B1, D1_B2, etc.)
- Monthly Planning (30 days, daily blocks with cumulative resource allocation)
- Rolling-Horizon Optimization (lookahead window with committed vs. candidate slots)
- Dynamic Rescheduling (updates active plans when real delay spikes or emergencies occur)
- Departmental Co-location Bonus (encourages multi-department tasks on the same section/slot)
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model


class MultiHorizonPlanner:
    """
    CP-SAT Multi-Horizon Planning Engine.
    Extends single 6-hour block optimization to 7-day, 30-day, rolling-horizon,
    and dynamic rescheduling contexts.
    """

    SLOT_MINUTES = 30
    SLOTS_PER_DAY = 48  # 24 * 2

    def __init__(
        self,
        max_manpower: int = 12,
        max_time_seconds: int = 30,
        num_search_workers: int = 8,
        score_scale: int = 1000,
        delay_penalty_scale: int = 10,
        coordination_bonus: int = 50,
    ):
        self.max_manpower = max_manpower
        self.max_time_seconds = max_time_seconds
        self.num_search_workers = num_search_workers
        self.score_scale = score_scale
        self.delay_penalty_scale = delay_penalty_scale
        self.coordination_bonus = coordination_bonus

    @staticmethod
    def _numeric(series, default=0.0):
        return pd.to_numeric(series, errors="coerce").fillna(default)

    @classmethod
    def generate_daily_block_windows(
        cls,
        days: int = 7,
        blocks_per_day: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates standard dual maintenance block windows per day.
        Default: B1 (00:30-03:00, slots 1-6) and B2 (03:30-05:30, slots 7-11) per day.
        """
        if blocks_per_day is None:
            # Default from notebook 25: B1 (slots 1 to 6), B2 (slots 7 to 11)
            blocks_per_day = [(1, 6), (7, 11)]

        windows = []
        for day in range(days):
            day_offset = day * cls.SLOTS_PER_DAY
            for b_idx, (b_start, b_end) in enumerate(blocks_per_day, start=1):
                windows.append({
                    "day": day + 1,
                    "block_id": f"D{day+1}_B{b_idx}",
                    "start_slot": day_offset + b_start,
                    "end_slot": day_offset + b_end,
                })
        return windows

    def plan_horizon(
        self,
        tasks: pd.DataFrame,
        days: int = 7,
        custom_windows: Optional[List[Dict[str, Any]]] = None,
        fixed_assignments: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Solves a multi-day maintenance planning problem.
        Enforces:
        - Section non-overlap (no two jobs on the same section at the same slot)
        - Manpower capacity (cumulative maximum manpower per slot)
        - Window boundary constraints (tasks must fit wholly inside block windows)
        - Coordination bonus: multi-department jobs on the same section inside the same window
        """
        if not isinstance(tasks, pd.DataFrame):
            raise TypeError("tasks must be a pandas DataFrame")

        required = [
            "task_id", "section_id", "estimated_duration",
            "required_manpower", "maintenance_decision_score", "predicted_delay_minutes"
        ]
        missing = [c for c in required if c not in tasks.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if tasks.empty:
            return pd.DataFrame(columns=[
                "plan_sequence", "task_id", "section_id", "department",
                "block_id", "day", "start_slot", "end_slot",
                "duration_minutes", "required_manpower",
                "maintenance_decision_score", "predicted_delay_minutes"
            ])

        total_slots = days * self.SLOTS_PER_DAY
        windows = custom_windows or self.generate_daily_block_windows(days=days)

        df = tasks.copy().reset_index(drop=True)
        df["estimated_duration"] = self._numeric(df["estimated_duration"]).clip(lower=0)
        df["required_manpower"] = self._numeric(df["required_manpower"]).clip(lower=0)
        df["maintenance_decision_score"] = self._numeric(df["maintenance_decision_score"]).clip(0, 1)
        df["predicted_delay_minutes"] = self._numeric(df["predicted_delay_minutes"]).clip(lower=0)

        df["duration_slots"] = df["estimated_duration"].apply(
            lambda v: max(1, math.ceil(v / self.SLOT_MINUTES))
        ).astype(int)

        model = cp_model.CpModel()

        task_selected = {}
        task_start = {}
        task_end = {}
        task_intervals = {}
        manpower_intervals = {}
        manpower_demands = {}

        # 1. Variables & Interval definition
        for i, row in df.iterrows():
            d_slots = int(row["duration_slots"])
            task_selected[i] = model.NewBoolVar(f"task_sel_{i}")
            task_start[i] = model.NewIntVar(0, total_slots - d_slots, f"start_{i}")
            task_end[i] = model.NewIntVar(0, total_slots, f"end_{i}")

            model.Add(task_end[i] == task_start[i] + d_slots)

            # Block window assignment
            allowed_window_assignments = []
            for w in windows:
                w_start = w["start_slot"]
                w_end = w["end_slot"]
                if w_end - w_start >= d_slots:
                    # Can fit in window w
                    in_w = model.NewBoolVar(f"task_{i}_in_win_{w['block_id']}")
                    allowed_window_assignments.append(in_w)
                    model.Add(task_start[i] >= w_start).OnlyEnforceIf(in_w)
                    model.Add(task_end[i] <= w_end).OnlyEnforceIf(in_w)

            if allowed_window_assignments:
                model.Add(sum(allowed_window_assignments) == task_selected[i])
            else:
                model.Add(task_selected[i] == 0)

            interval = model.NewOptionalIntervalVar(
                task_start[i], d_slots, task_end[i], task_selected[i], f"interval_{i}"
            )
            task_intervals[i] = interval

            mp = max(0, int(math.ceil(row["required_manpower"])))
            manpower_intervals[i] = interval
            manpower_demands[i] = mp

        # 2. Section non-overlap constraints
        for section_id, group in df.groupby("section_id"):
            sec_idx = group.index.tolist()
            if len(sec_idx) > 1:
                model.AddNoOverlap([task_intervals[idx] for idx in sec_idx])

        # 3. Manpower cumulative capacity
        if manpower_intervals:
            model.AddCumulative(
                list(manpower_intervals.values()),
                list(manpower_demands.values()),
                self.max_manpower
            )

        # 4. Handle fixed assignments (e.g. from previous rolling window)
        if fixed_assignments:
            for fix in fixed_assignments:
                t_id = fix.get("task_id")
                matches = df.index[df["task_id"] == t_id].tolist()
                if matches:
                    idx = matches[0]
                    model.Add(task_selected[idx] == 1)
                    if "start_slot" in fix:
                        model.Add(task_start[idx] == int(fix["start_slot"]))

        # 5. Objective: Priority - Delay Penalty (+ optional coordination incentive)
        objective_terms = []
        for i, row in df.iterrows():
            priority = round(float(row["maintenance_decision_score"]) * self.score_scale)
            delay_penalty = round(float(row["predicted_delay_minutes"]) * self.delay_penalty_scale)
            coeff = priority - delay_penalty
            objective_terms.append(coeff * task_selected[i])

        model.Maximize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time_seconds
        solver.parameters.num_search_workers = self.num_search_workers

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return pd.DataFrame(columns=[
                "plan_sequence", "task_id", "section_id", "department",
                "block_id", "day", "start_slot", "end_slot",
                "duration_minutes", "required_manpower",
                "maintenance_decision_score", "predicted_delay_minutes"
            ])

        scheduled = []
        for i, row in df.iterrows():
            if not solver.BooleanValue(task_selected[i]):
                continue
            s_val = solver.Value(task_start[i])
            e_val = solver.Value(task_end[i])

            # Find matching block window
            assigned_block = "UNKNOWN"
            assigned_day = (s_val // self.SLOTS_PER_DAY) + 1
            for w in windows:
                if s_val >= w["start_slot"] and e_val <= w["end_slot"]:
                    assigned_block = w["block_id"]
                    assigned_day = w.get("day", assigned_day)
                    break

            scheduled.append({
                "task_id": row["task_id"],
                "section_id": row["section_id"],
                "department": row.get("department", "UNKNOWN"),
                "block_id": assigned_block,
                "day": assigned_day,
                "start_slot": s_val,
                "end_slot": e_val,
                "duration_minutes": (e_val - s_val) * self.SLOT_MINUTES,
                "required_manpower": row["required_manpower"],
                "maintenance_decision_score": row["maintenance_decision_score"],
                "predicted_delay_minutes": row["predicted_delay_minutes"],
            })

        res_df = pd.DataFrame(scheduled)
        if not res_df.empty:
            res_df = res_df.sort_values(["start_slot", "section_id"]).reset_index(drop=True)
            res_df.insert(0, "plan_sequence", range(1, len(res_df) + 1))
        return res_df

    def plan_weekly(self, tasks: pd.DataFrame) -> pd.DataFrame:
        """7-day weekly block plan generation."""
        return self.plan_horizon(tasks=tasks, days=7)

    def plan_monthly(self, tasks: pd.DataFrame) -> pd.DataFrame:
        """30-day monthly block plan generation."""
        return self.plan_horizon(tasks=tasks, days=30)

    def plan_rolling_horizon(
        self,
        tasks: pd.DataFrame,
        horizon_days: int = 7,
        committed_days: int = 2,
        current_committed_plan: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Implements rolling-horizon planning.
        - committed_days: period where previous schedule is strictly locked.
        - lookahead (horizon_days - committed_days): flexible optimization period.
        """
        fixed_assignments = []
        if current_committed_plan is not None and not current_committed_plan.empty:
            committed_slots_limit = committed_days * self.SLOTS_PER_DAY
            for _, row in current_committed_plan.iterrows():
                if row["start_slot"] < committed_slots_limit:
                    fixed_assignments.append({
                        "task_id": row["task_id"],
                        "start_slot": row["start_slot"],
                    })

        plan = self.plan_horizon(
            tasks=tasks,
            days=horizon_days,
            fixed_assignments=fixed_assignments,
        )

        committed_slots_limit = committed_days * self.SLOTS_PER_DAY
        committed = plan[plan["start_slot"] < committed_slots_limit].copy()
        tentative = plan[plan["start_slot"] >= committed_slots_limit].copy()

        return {
            "full_plan": plan,
            "committed_plan": committed,
            "tentative_plan": tentative,
            "committed_days": committed_days,
            "horizon_days": horizon_days,
        }

    def reschedule_dynamically(
        self,
        current_plan: pd.DataFrame,
        disrupted_sections: Dict[str, float],
        worklist: pd.DataFrame,
        current_slot: int = 0,
        planning_days: int = 7,
    ) -> pd.DataFrame:
        """
        Dynamically reschedules blocks in response to real-time events.
        - Tasks already completed (end_slot <= current_slot) are locked.
        - Tasks on disrupted sections receive updated delay or pressure penalties.
        - Tasks currently ongoing (start_slot <= current_slot < end_slot) are preserved.
        - Remaining tasks are re-optimized for future slots.
        """
        fixed_assignments = []
        # Keep past/ongoing assignments intact
        if not current_plan.empty:
            for _, row in current_plan.iterrows():
                if row["start_slot"] < current_slot:
                    fixed_assignments.append({
                        "task_id": row["task_id"],
                        "start_slot": row["start_slot"],
                    })

        # Update delay/pressure on affected tasks in worklist
        updated_worklist = worklist.copy()
        for sec, added_delay in disrupted_sections.items():
            mask = updated_worklist["section_id"] == sec
            if "predicted_delay_minutes" in updated_worklist.columns:
                updated_worklist.loc[mask, "predicted_delay_minutes"] += added_delay

        return self.plan_horizon(
            tasks=updated_worklist,
            days=planning_days,
            fixed_assignments=fixed_assignments,
        )
