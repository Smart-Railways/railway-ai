import json
from datetime import datetime
from pathlib import Path

from src.data.railkit_adapter import RailKitAdapter


class RailKitCapture:
    """
    Capture raw RailKit operational responses without modifying them.

    Raw responses are stored separately from processed ML data so that
    normalization/features can evolve without losing the original data.
    """

    def __init__(self, output_dir="data/raw_real/railkit"):
        self.adapter = RailKitAdapter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_live_train(self, train_number, date):
        captured_at = datetime.now().strftime("%Y%m%d_%H%M%S")

        response = self.adapter.get_live_train_context(
            train_number,
            date,
        )

        filename = (
            f"live_train_{train_number}_"
            f"{date.replace('-', '')}_"
            f"{captured_at}.json"
        )

        path = self.output_dir / filename

        payload = {
            "source": "railkit",
            "endpoint_type": "live_train_tracking",
            "train_number": str(train_number),
            "requested_date": date,
            "captured_at": datetime.now().isoformat(),
            "response": response,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return path
