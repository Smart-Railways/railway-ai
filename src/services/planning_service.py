import pandas as pd

from src.decision.maintenance_decision_engine import (
    MaintenanceDecisionEngine,
)
from src.optimization.block_optimizer import BlockOptimizer


class PlanningService:
    """
    End-to-end local maintenance planning orchestration.

    Pipeline:
        maintenance worklist
            -> decision scoring
            -> optimizer-contract normalization
            -> CP-SAT block optimization

    RailKit operational pressure may be supplied explicitly,
    but this service never infers section mappings from station
    presence.
    """

    def __init__(
        self,
        decision_engine=None,
        optimizer=None,
    ):
        self.decision_engine = (
            decision_engine or MaintenanceDecisionEngine()
        )

        self.optimizer = (
            optimizer or BlockOptimizer()
        )

    @staticmethod
    def _validate_worklist(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "worklist must be a pandas DataFrame"
            )

        required = [
            "task_id",
            "section_id",
            "estimated_duration",
            "required_manpower",
            "failure_probability",
            "urgency_score",
            "criticality",
            "overdue_days",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Worklist missing required columns: {missing}"
            )

        if df.empty:
            raise ValueError("Worklist is empty")

        return df.copy()

    @staticmethod
    def _prepare_optimizer_input(df):
        """
        Normalize optional inputs required by the optimizer.

        Missing task-level predicted delay means that no
        task-specific delay prediction is currently available.
        Therefore the optimizer receives an explicit zero penalty
        rather than an invented prediction.
        """
        result = df.copy()

        if "predicted_delay_minutes" not in result.columns:
            result["predicted_delay_minutes"] = 0.0

        result["predicted_delay_minutes"] = (
            pd.to_numeric(
                result["predicted_delay_minutes"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        return result

    def score(self, worklist, railkit_pressure=None):
        """
        Score maintenance tasks.

        If railkit_pressure is supplied, it is applied explicitly
        as the operational factor for the provided worklist.

        No section mapping is inferred here.
        """
        df = self._validate_worklist(worklist)

        if railkit_pressure is not None:
            pressure = float(railkit_pressure)

            if not 0.0 <= pressure <= 1.0:
                raise ValueError(
                    "railkit_pressure must be between 0 and 1"
                )

            df["railkit_operational_pressure"] = pressure

        return self.decision_engine.transform(df)

    def plan(self, worklist, railkit_pressure=None):
        """
        Generate an optimized maintenance block plan.
        """
        scored = self.score(
            worklist,
            railkit_pressure=railkit_pressure,
        )

        optimizer_input = self._prepare_optimizer_input(
            scored
        )

        return self.optimizer.optimize(
            optimizer_input
        )

    def plan_with_validated_mapping(
        self,
        worklist,
        section_pressure,
    ):
        """
        Apply explicitly validated section-level RailKit
        operational pressure and optimize.

        section_pressure must be a mapping:

            {
                "NDL-MTJ-01": 0.31,
                "MTJ-AGC-01": 0.54,
            }

        Sections absent from the mapping receive no inferred
        RailKit pressure.
        """
        df = self._validate_worklist(worklist)

        if not isinstance(section_pressure, dict):
            raise TypeError(
                "section_pressure must be a dictionary"
            )

        df["railkit_operational_pressure"] = (
            df["section_id"]
            .map(section_pressure)
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        scored = self.decision_engine.transform(df)

        optimizer_input = self._prepare_optimizer_input(
            scored
        )

        return self.optimizer.optimize(
            optimizer_input
        )
