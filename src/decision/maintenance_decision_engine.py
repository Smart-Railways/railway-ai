import numpy as np
import pandas as pd


class MaintenanceDecisionEngine:
    """
    Unified maintenance decision scoring.

    Score range: [0, 1].

    The engine preserves the existing multi-objective formulation
    while allowing real RailKit operational pressure to replace
    the operational-impact component when explicitly supplied.
    """

    WEIGHTS = {
        "failure_risk_factor": 0.25,
        "urgency_factor": 0.20,
        "criticality_factor": 0.15,
        "overdue_factor": 0.10,
        "duration_factor": 0.05,
        "traffic_factor": 0.10,
        "operational_factor": 0.15,
    }

    @staticmethod
    def _numeric_column(
        df: pd.DataFrame,
        column: str,
        default: float = 0.0,
    ) -> pd.Series:
        """
        Return a numeric Series for a column.

        Missing columns become a Series of the correct length,
        avoiding scalar .get(...).fillna(...) failures.
        """
        if column not in df.columns:
            return pd.Series(
                default,
                index=df.index,
                dtype=float,
            )

        return pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(default)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        result = df.copy()

        result["failure_risk_factor"] = (
            self._numeric_column(
                result,
                "failure_probability",
            )
            .clip(0, 1)
        )

        result["urgency_factor"] = (
            self._numeric_column(
                result,
                "urgency_score",
            )
            .clip(0, 1)
        )

        result["criticality_factor"] = (
            self._numeric_column(
                result,
                "criticality",
            )
            .div(10)
            .clip(0, 1)
        )

        result["overdue_factor"] = (
            self._numeric_column(
                result,
                "overdue_days",
            )
            .div(30)
            .clip(0, 1)
        )

        result["duration_factor"] = (
            self._numeric_column(
                result,
                "estimated_duration",
            )
            .div(180)
            .clip(0, 1)
        )

        result["traffic_factor"] = (
            self._numeric_column(
                result,
                "traffic_intensity",
            )
            .clip(0, 1)
        )

        if "railkit_operational_pressure" in result.columns:
            result["operational_factor"] = (
                self._numeric_column(
                    result,
                    "railkit_operational_pressure",
                )
                .clip(0, 1)
            )

        elif "operational_impact_score" in result.columns:
            result["operational_factor"] = (
                self._numeric_column(
                    result,
                    "operational_impact_score",
                )
                .clip(0, 1)
            )

        else:
            result["operational_factor"] = 0.0

        result["maintenance_decision_score"] = (
            self.WEIGHTS["failure_risk_factor"]
            * result["failure_risk_factor"]
            + self.WEIGHTS["urgency_factor"]
            * result["urgency_factor"]
            + self.WEIGHTS["criticality_factor"]
            * result["criticality_factor"]
            + self.WEIGHTS["overdue_factor"]
            * result["overdue_factor"]
            + self.WEIGHTS["duration_factor"]
            * result["duration_factor"]
            + self.WEIGHTS["traffic_factor"]
            * result["traffic_factor"]
            + self.WEIGHTS["operational_factor"]
            * result["operational_factor"]
        ).clip(0, 1)

        result["decision_category"] = pd.cut(
            result["maintenance_decision_score"],
            bins=[
                -np.inf,
                0.25,
                0.50,
                0.75,
                np.inf,
            ],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            ],
        )

        result = result.sort_values(
            [
                "maintenance_decision_score",
                "failure_risk_factor",
                "criticality_factor",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).reset_index(drop=True)

        result.insert(
            0,
            "decision_rank",
            np.arange(1, len(result) + 1),
        )

        # Generate explainable decision reason codes
        reason_codes = []
        confidence_levels = []
        for _, row in result.iterrows():
            reasons = []
            if row["failure_risk_factor"] >= 0.5:
                reasons.append("HIGH_FAILURE_RISK")
            if row["urgency_factor"] >= 0.6:
                reasons.append("HIGH_URGENCY")
            if row["criticality_factor"] >= 0.8:
                reasons.append("HIGH_CRITICALITY")
            if row["overdue_factor"] > 0:
                reasons.append("OVERDUE_MAINTENANCE")
            if row["operational_factor"] >= 0.3:
                reasons.append("HIGH_OPERATIONAL_PRESSURE")
            if not reasons:
                reasons.append("ROUTINE_INSPECTION")
            reason_codes.append(";".join(reasons))

            # Confidence scoring: high when real operational or empirical data exists
            if row.get("railkit_operational_pressure") is not None and not pd.isna(row.get("railkit_operational_pressure")):
                confidence_levels.append(0.92)
            else:
                confidence_levels.append(0.85)

        result["decision_reasons"] = reason_codes
        result["confidence"] = confidence_levels

        return result
