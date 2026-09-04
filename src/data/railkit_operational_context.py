import json
from datetime import datetime
from pathlib import Path

from src.data.railkit_adapter import RailKitAdapter
from src.data.railkit_normalizer import RailKitNormalizer


class RailKitOperationalContext:
    """
    Fetches live RailKit data, normalizes it, and optionally
    persists the resulting operational context.
    """

    def __init__(self, output_dir="data/processed/railkit"):
        self.adapter = RailKitAdapter()
        self.normalizer = RailKitNormalizer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_train_context(self, train_number, date):
        raw = self.adapter.get_live_train_context(
            train_number,
            date,
        )

        normalized = self.normalizer.normalize_live_tracking(raw)

        return normalized

    def save_train_context(self, context):
        train_number = str(context["train_number"])
        journey_date = str(context["journey_date"])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = (
            f"train_{train_number}_"
            f"{journey_date.replace('-', '')}_"
            f"{timestamp}.json"
        )

        path = self.output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)

        return path

    def fetch_and_save(self, train_number, date):
        context = self.fetch_train_context(
            train_number,
            date,
        )

        path = self.save_train_context(context)

        return context, path
