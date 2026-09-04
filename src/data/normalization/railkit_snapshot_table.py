import json
from pathlib import Path

import pandas as pd


class RailKitSnapshotTable:
    def __init__(
        self,
        raw_dir="data/raw_real/railkit",
        output="data/processed_real/railkit_train_snapshots.csv",
    ):
        self.raw_dir = Path(raw_dir)
        self.output = Path(output)

    def build(self):
        records = []

        for path in sorted(
            self.raw_dir.glob("live_train_*.json")
        ):
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)

            response = payload.get("response", {})
            data = response.get("data", {})

            timeline = data.get("timeline", []) or []

            delays = []

            for station in timeline:
                for movement_key in ("arrival", "departure"):
                    movement = station.get(
                        movement_key,
                        {},
                    ) or {}

                    delay = movement.get("delay")

                    if delay is None or delay == "":
                        continue

                    text = str(delay).lower().strip()

                    if text == "on time":
                        delays.append(0.0)
                        continue

                    digits = "".join(
                        c for c in text
                        if c.isdigit() or c == "."
                    )

                    if digits:
                        delays.append(float(digits))

            snapshot_id = f"{data.get("trainNo")}_{payload.get("captured_at")}"

            records.append({
                "snapshot_id": snapshot_id,
                "capture_file": str(path),
                "captured_at": payload.get("captured_at"),
                "train_number": data.get("trainNo"),
                "train_name": data.get("trainName"),
                "journey_date": data.get("date"),
                "current_station_code": data.get(
                    "currentStationCode"
                ),
                "status_note": data.get("statusNote"),
                "total_stations": data.get(
                    "totalStations"
                ),
                "timeline_station_count": len(timeline),
                "max_observed_delay_minutes": (
                    max(delays) if delays else 0.0
                ),
                "average_observed_delay_minutes": (
                    sum(delays) / len(delays)
                    if delays
                    else 0.0
                ),
                "delay_observation_count": len(delays),
                "has_delay": int(
                    any(delay > 0 for delay in delays)
                ),
            })

        df = pd.DataFrame(records)

        self.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            self.output,
            index=False,
        )

        return df, self.output
