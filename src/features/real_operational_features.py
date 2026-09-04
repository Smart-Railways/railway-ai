import pandas as pd


class RealOperationalFeatureBuilder:
    """
    Build ML-ready operational features from real RailKit snapshots.

    Uses the canonical RailKit operational-pressure formulation:
        60% maximum observed delay
        25% average observed delay
        15% upcoming route pressure

    Does not modify existing synthetic datasets.
    """

    def transform(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(snapshots, pd.DataFrame):
            raise TypeError("snapshots must be a pandas DataFrame")

        df = snapshots.copy()

        required_columns = [
            "max_observed_delay_minutes",
            "average_observed_delay_minutes",
            "upcoming_station_count",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        numeric_cols = [
            "max_observed_delay_minutes",
            "average_observed_delay_minutes",
            "upcoming_station_count",
            "delay_observation_count",
            "total_stations",
            "timeline_station_count",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).fillna(0.0)

        df["railkit_delay_risk"] = (
            df["max_observed_delay_minutes"]
            .clip(lower=0, upper=60)
            / 60.0
        )

        df["railkit_delay_intensity"] = (
            df["average_observed_delay_minutes"]
            .clip(lower=0, upper=60)
            / 60.0
        )

        df["railkit_route_pressure"] = (
            df["upcoming_station_count"]
            .clip(lower=0, upper=100)
            / 100.0
        )

        df["railkit_operational_pressure"] = (
            0.60 * df["railkit_delay_risk"]
            + 0.25 * df["railkit_delay_intensity"]
            + 0.15 * df["railkit_route_pressure"]
        ).clip(0.0, 1.0)

        df["railkit_data_available"] = (
            df["train_number"].notna().astype(int)
            if "train_number" in df.columns
            else 0
        )

        return df
