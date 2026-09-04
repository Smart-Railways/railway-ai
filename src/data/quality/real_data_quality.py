from pathlib import Path

import pandas as pd


class RealDataQualityReport:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, message):
        self.warnings.append(message)

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self):
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class RealDataQualityChecker:
    REQUIRED_COLUMNS = {
        "assets": {
            "asset_id",
            "section_id",
            "department",
            "asset_type",
            "installation_year",
            "criticality",
            "condition_score",
            "failure_count",
            "last_maintenance_date",
        },
        "failures": {
            "failure_id",
            "asset_id",
            "failure_date",
            "failure_type",
            "severity",
            "downtime_hours",
            "repair_duration_hours",
        },
        "maintenance": {
            "task_id",
            "asset_id",
            "section_id",
            "department",
            "maintenance_type",
            "defect_type",
            "severity",
            "criticality",
            "created_date",
            "due_date",
            "overdue_days",
            "estimated_duration",
            "required_manpower",
            "status",
        },
        "trains": {
            "movement_id",
            "train_id",
            "section_id",
            "train_type",
            "entry_time",
            "exit_time",
            "priority",
        },
        "freight": {
            "forecast_id",
            "section_id",
            "date",
            "hour",
            "expected_freight_count",
            "forecast_confidence",
        },
    }

    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)

    def load(self, name):
        return pd.read_csv(self.raw_dir / f"{name}.csv")

    def check_required_columns(self, name, df, report):
        required = self.REQUIRED_COLUMNS[name]
        missing = required - set(df.columns)

        if missing:
            report.error(
                f"{name}: missing columns: {sorted(missing)}"
            )

    def check_duplicates(self, name, df, id_column, report):
        if id_column not in df.columns:
            return

        duplicate_count = df[id_column].duplicated().sum()

        if duplicate_count:
            report.error(
                f"{name}: {duplicate_count} duplicate {id_column} values"
            )

    def check_range(
        self,
        name,
        df,
        column,
        minimum,
        maximum,
        report,
    ):
        if column not in df.columns:
            return

        values = pd.to_numeric(df[column], errors="coerce")

        invalid = values.notna() & (
            (values < minimum) | (values > maximum)
        )

        count = int(invalid.sum())

        if count:
            report.error(
                f"{name}: {count} values outside "
                f"{column} range [{minimum}, {maximum}]"
            )

    def check_dates(self, name, df, columns, report):
        for column in columns:
            if column not in df.columns:
                continue

            parsed = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            invalid = parsed.isna() & df[column].notna()

            count = int(invalid.sum())

            if count:
                report.error(
                    f"{name}: {count} invalid dates in {column}"
                )

    def run(self):
        report = RealDataQualityReport()

        datasets = {}

        for name in self.REQUIRED_COLUMNS:
            path = self.raw_dir / f"{name}.csv"

            if not path.exists():
                report.error(
                    f"{name}: file not found: {path}"
                )
                continue

            df = self.load(name)
            datasets[name] = df

            if df.empty:
                report.error(f"{name}: dataset is empty")

            self.check_required_columns(
                name,
                df,
                report,
            )

        if "assets" in datasets:
            assets = datasets["assets"]

            self.check_duplicates(
                "assets",
                assets,
                "asset_id",
                report,
            )

            self.check_range(
                "assets",
                assets,
                "criticality",
                1,
                10,
                report,
            )

            self.check_range(
                "assets",
                assets,
                "condition_score",
                0,
                100,
                report,
            )

            self.check_dates(
                "assets",
                assets,
                ["last_maintenance_date"],
                report,
            )

        if "failures" in datasets:
            failures = datasets["failures"]

            self.check_duplicates(
                "failures",
                failures,
                "failure_id",
                report,
            )

            self.check_range(
                "failures",
                failures,
                "severity",
                1,
                10,
                report,
            )

            self.check_range(
                "failures",
                failures,
                "downtime_hours",
                0,
                float("inf"),
                report,
            )

            self.check_range(
                "failures",
                failures,
                "repair_duration_hours",
                0,
                float("inf"),
                report,
            )

            self.check_dates(
                "failures",
                failures,
                ["failure_date"],
                report,
            )

            if "assets" in datasets:
                known_assets = set(
                    datasets["assets"]["asset_id"]
                )

                unknown_assets = set(
                    failures["asset_id"]
                ) - known_assets

                if unknown_assets:
                    report.error(
                        "failures: unknown asset IDs: "
                        f"{sorted(unknown_assets)}"
                    )

        if "maintenance" in datasets:
            maintenance = datasets["maintenance"]

            self.check_duplicates(
                "maintenance",
                maintenance,
                "task_id",
                report,
            )

            self.check_range(
                "maintenance",
                maintenance,
                "severity",
                1,
                10,
                report,
            )

            self.check_range(
                "maintenance",
                maintenance,
                "criticality",
                1,
                10,
                report,
            )

            self.check_range(
                "maintenance",
                maintenance,
                "required_manpower",
                0,
                float("inf"),
                report,
            )

            self.check_dates(
                "maintenance",
                maintenance,
                ["created_date", "due_date"],
                report,
            )

            if "assets" in datasets:
                known_assets = set(
                    datasets["assets"]["asset_id"]
                )

                unknown_assets = set(
                    maintenance["asset_id"]
                ) - known_assets

                if unknown_assets:
                    report.error(
                        "maintenance: unknown asset IDs: "
                        f"{sorted(unknown_assets)}"
                    )

        if "trains" in datasets:
            trains = datasets["trains"]

            self.check_duplicates(
                "trains",
                trains,
                "movement_id",
                report,
            )

            self.check_dates(
                "trains",
                trains,
                ["entry_time", "exit_time"],
                report,
            )

            entry = pd.to_datetime(
                trains["entry_time"],
                errors="coerce",
            )

            exit_ = pd.to_datetime(
                trains["exit_time"],
                errors="coerce",
            )

            invalid = (
                entry.notna()
                & exit_.notna()
                & (exit_ < entry)
            )

            if invalid.any():
                report.error(
                    f"trains: {int(invalid.sum())} movements "
                    "have exit before entry"
                )

        if "freight" in datasets:
            freight = datasets["freight"]

            self.check_duplicates(
                "freight",
                freight,
                "forecast_id",
                report,
            )

            self.check_range(
                "freight",
                freight,
                "hour",
                0,
                23,
                report,
            )

            self.check_range(
                "freight",
                freight,
                "expected_freight_count",
                0,
                float("inf"),
                report,
            )

            self.check_range(
                "freight",
                freight,
                "forecast_confidence",
                0,
                1,
                report,
            )

            self.check_dates(
                "freight",
                freight,
                ["date"],
                report,
            )

        return report
