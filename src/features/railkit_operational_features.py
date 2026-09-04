import pandas as pd


class RailKitOperationalFeatures:
    """
    Derives operational-risk features from normalized RailKit
    live-tracking context.

    These features are intended for downstream impact scoring
    and block optimization, not direct train-control decisions.
    """

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        result = df.copy()

        required = [
            "railkit_max_observed_delay_minutes",
            "railkit_average_observed_delay_minutes",
            "railkit_upcoming_station_count",
        ]

        for column in required:
            if column not in result.columns:
                result[column] = 0.0

        max_delay = pd.to_numeric(
            result["railkit_max_observed_delay_minutes"],
            errors="coerce",
        ).fillna(0.0)

        avg_delay = pd.to_numeric(
            result["railkit_average_observed_delay_minutes"],
            errors="coerce",
        ).fillna(0.0)

        upcoming = pd.to_numeric(
            result["railkit_upcoming_station_count"],
            errors="coerce",
        ).fillna(0.0)

        # Saturates at 60 minutes.
        result["railkit_delay_risk"] = (
            max_delay.clip(lower=0, upper=60) / 60.0
        )

        # Average delay contributes a smaller secondary signal.
        avg_delay_signal = (
            avg_delay.clip(lower=0, upper=60) / 60.0
        )

        # More upcoming stations means more opportunity for
        # operational interaction, but this is deliberately capped.
        route_pressure = (
            upcoming.clip(lower=0, upper=100) / 100.0
        )

        result["railkit_operational_pressure"] = (
            0.60 * result["railkit_delay_risk"]
            + 0.25 * avg_delay_signal
            + 0.15 * route_pressure
        ).clip(0.0, 1.0)

        result["railkit_data_available"] = (
            result[
                "railkit_current_station_code"
            ].notna().astype(int)
            if "railkit_current_station_code" in result.columns
            else 0
        )

        return result
