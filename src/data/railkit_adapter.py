from datetime import datetime
from src.data.railkit_client import RailKitClient


class RailKitAdapter:
    """
    Converts RailKit responses into a small, stable structure
    that the ML pipeline can consume.
    """

    def __init__(self):
        self.client = RailKitClient()

    def get_station_context(self, station_code):
        """Fetch basic station information."""
        response = self.client.station_lookup(station_code)

        return {
            "station_code": station_code,
            "success": response.get("success", False),
            "data": response.get("data"),
        }

    def get_train_context(self, train_number):
        """Fetch train information."""
        response = self.client.train_lookup(train_number)

        return {
            "train_number": str(train_number),
            "success": response.get("success", False),
            "data": response.get("data"),
        }

    def get_live_train_context(self, train_number, date):
        """
        Fetch live train tracking information.

        date format: DD-MM-YYYY
        """
        response = self.client.live_tracking(train_number, date)

        return {
            "train_number": str(train_number),
            "journey_date": date,
            "success": response.get("success", False),
            "data": response.get("data"),
            "retrieved_at": datetime.utcnow().isoformat(),
        }
