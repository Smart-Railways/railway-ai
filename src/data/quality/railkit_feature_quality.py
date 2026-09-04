import pandas as pd


class RailKitFeatureQuality:
    """Validate the real RailKit-derived feature dataset."""

    REQUIRED_COLUMNS = [
        "snapshot_id",
        "train_number",
        "current_station_code",
        "max_observed_delay_minutes",
        "average_observed_delay_minutes",
        "railkit_delay_risk",
        "railkit_delay_intensity",
        "railkit_operational_pressure",
        "railkit_data_available",
    ]

    def validate(self, df: pd.DataFrame) -> dict:
        errors = []
        warnings = []

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            errors.append(
                f"Missing required columns: {missing}"
            )

        if df.empty:
            errors.append("Feature dataset is empty.")
            return {
                "passed": False,
                "errors": errors,
                "warnings": warnings,
            }

        numeric_ranges = {
            "railkit_delay_risk": (0, 1),
            "railkit_delay_intensity": (0, 1),
            "railkit_operational_pressure": (0, 1),
            "railkit_data_available": (0, 1),
        }

        for column, (low, high) in numeric_ranges.items():
            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if values.isna().any():
                errors.append(
                    f"{column} contains non-numeric or missing values."
                )

            if ((values < low) | (values > high)).any():
                errors.append(
                    f"{column} contains values outside [{low}, {high}]."
                )

        if "max_observed_delay_minutes" in df.columns:
            delays = pd.to_numeric(
                df["max_observed_delay_minutes"],
                errors="coerce",
            )

            if (delays < 0).any():
                errors.append(
                    "Negative observed delays detected."
                )

        if "snapshot_id" in df.columns:
            if df["snapshot_id"].duplicated().any():
                errors.append(
                    "Duplicate snapshot_id values detected."
                )

        if "train_number" in df.columns:
            if df["train_number"].isna().any():
                errors.append(
                    "Missing train numbers detected."
                )

        if "current_station_code" in df.columns:
            if df["current_station_code"].isna().any():
                warnings.append(
                    "Some snapshots have no current station code."
                )

        return {
            "passed": len(errors) == 0,
            "errors": errors or None,
            "warnings": warnings or None,
        }
