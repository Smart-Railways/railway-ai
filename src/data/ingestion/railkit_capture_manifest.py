import json
from pathlib import Path


class RailKitCaptureManifest:
    def __init__(self, raw_dir="data/raw_real/railkit"):
        self.raw_dir = Path(raw_dir)

    def build(self):
        records = []

        for path in sorted(self.raw_dir.glob("live_train_*.json")):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)

            response = payload.get("response", {})
            data = response.get("data", {})

            records.append({
                "file": str(path),
                "source": payload.get("source"),
                "endpoint_type": payload.get("endpoint_type"),
                "train_number": payload.get("train_number"),
                "requested_date": payload.get("requested_date"),
                "captured_at": payload.get("captured_at"),
                "journey_date": data.get("date"),
                "current_station_code": data.get(
                    "currentStationCode"
                ),
                "timeline_station_count": len(
                    data.get("timeline", []) or []
                ),
            })

        return records

    def save(self, output="data/raw_real/railkit/manifest.json"):
        records = self.build()

        output_path = Path(output)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "0.1.0",
                    "capture_count": len(records),
                    "captures": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return output_path
