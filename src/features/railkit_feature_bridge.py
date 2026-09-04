import pandas as pd


class RailKitFeatureBridge:
    """
    Converts normalized RailKit operational context into
    ML-ready operational features.

    RailKit features are intentionally kept separate from
    asset-level features until an explicit mapping is available.
    """

    FEATURE_COLUMNS = [
        "railkit_train_number",
        "railkit_current_station_code",
        "railkit_total_stations",
        "railkit_upcoming_station_count",
        "railkit_stoppage_count",
        "railkit_intermediate_station_count",
        "railkit_current_distance_km",
        "railkit_max_observed_delay_minutes",
        "railkit_average_observed_delay_minutes",
        "railkit_has_delay",
    ]

    def transform(self, context):
        """
        Convert one normalized RailKit context dictionary
        into a one-row pandas DataFrame.
        """

        row = {
            "railkit_train_number": str(
                context.get("train_number", "")
            ),
            "railkit_current_station_code": context.get(
                "current_station_code"
            ),
            "railkit_total_stations": context.get(
                "total_stations", 0
            ),
            "railkit_upcoming_station_count": context.get(
                "upcoming_station_count", 0
            ),
            "railkit_stoppage_count": context.get(
                "stoppage_count", 0
            ),
            "railkit_intermediate_station_count": context.get(
                "intermediate_station_count", 0
            ),
            "railkit_current_distance_km": context.get(
                "current_distance_km"
            ),
            "railkit_max_observed_delay_minutes": context.get(
                "max_observed_delay_minutes", 0.0
            ),
            "railkit_average_observed_delay_minutes": context.get(
                "average_observed_delay_minutes", 0.0
            ),
            "railkit_has_delay": int(
                bool(context.get("has_delay", False))
            ),
        }

        return pd.DataFrame(
            [row],
            columns=self.FEATURE_COLUMNS,
        )

