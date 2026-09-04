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

        # Preserve the prototype's global non-overlap rule.
        if task_intervals:
            model.AddNoOverlap(
                list(task_intervals.values())
            )

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
