from datetime import datetime


class RailKitNormalizer:
    """
    Converts raw RailKit live-tracking responses into
    ML/optimization-friendly operational features.
    """

    @staticmethod
    def _safe_float(value, default=None):
        try:
            if value in ("", None):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _delay_minutes(value):
        """
        Converts common RailKit delay representations into minutes.

        Semantics:
        - missing/empty values -> None (no observation)
        - "On Time" -> 0.0
        - numeric delay values -> minutes
        - unparseable values -> None
        """
        if value is None:
            return None

        text = str(value).strip().lower()

        if not text:
            return None

        if text == "on time":
            return 0.0

        digits = ""
        for char in text:
            if char.isdigit() or char == ".":
                digits += char

        try:
            return float(digits) if digits else None
        except ValueError:
            return None

    def normalize_live_tracking(self, response):
        """
        Normalize a RailKit live-tracking response.

        Returns a compact dictionary suitable for the ML
        feature/decision layer.
        """

        data = response.get("data", {}) if isinstance(response, dict) else {}

        timeline = data.get("timeline", []) or []

        current_station = data.get("currentStationCode")
        upcoming_stations = []
        current_station_data = None

        delays = []
        stoppage_count = 0
        intermediate_count = 0

        for station in timeline:
            status = station.get("status")
            station_type = station.get("type")

            if status == "current":
                current_station_data = station

            if status == "upcoming":
                upcoming_stations.append(station.get("stationCode"))

            if station_type == "stoppage":
                stoppage_count += 1

            if station_type == "intermediate":
                intermediate_count += 1

            arrival = station.get("arrival", {})
            departure = station.get("departure", {})

            arrival_delay = self._delay_minutes(
                arrival.get("delay")
            )
            departure_delay = self._delay_minutes(
                departure.get("delay")
            )

            delays.extend([arrival_delay, departure_delay])

        valid_delays = [d for d in delays if d is not None]

        max_delay = max(valid_delays) if valid_delays else 0.0
        avg_delay = (
            sum(valid_delays) / len(valid_delays)
            if valid_delays
            else 0.0
        )

        current_distance = None
        if current_station_data:
            current_distance = self._safe_float(
                current_station_data.get("distanceKm")
            )

        return {
            "train_number": data.get("trainNo"),
            "train_name": data.get("trainName"),
            "journey_date": data.get("date"),
            "current_station_code": current_station,
            "status_note": data.get("statusNote"),
            "total_stations": data.get("totalStations", 0),
            "timeline_station_count": len(timeline),
            "upcoming_station_count": len(upcoming_stations),
            "stoppage_count": stoppage_count,
            "intermediate_station_count": intermediate_count,
            "current_distance_km": current_distance,
            "max_observed_delay_minutes": max_delay,
            "average_observed_delay_minutes": avg_delay,
            "has_delay": max_delay > 0,
            "last_update": data.get("lastUpdate"),
            "normalized_at": datetime.utcnow().isoformat(),
        }
