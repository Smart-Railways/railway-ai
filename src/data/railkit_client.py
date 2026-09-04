import os
import requests
from dotenv import load_dotenv


load_dotenv()


class RailKitClient:
    def __init__(self):
        self.base_url = os.getenv("RAILKIT_BASE_URL")
        self.api_key = os.getenv("RAILKIT_API_KEY")

        if not self.base_url:
            raise ValueError("RAILKIT_BASE_URL is not set")

        if not self.api_key:
            raise ValueError("RAILKIT_API_KEY is not set")

        self.headers = {
            "x-api-key": self.api_key
        }

    def get(self, endpoint, params=None):
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        response = requests.get(
            url,
            headers=self.headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    def station_lookup(self, station_code):
        return self.get(
            f"/api/station/{station_code}"
        )

    def train_lookup(self, train_number):
        return self.get(
            f"/api/train/{train_number}"
        )

    def live_tracking(self, train_number, date):
        endpoint = f"/api/trackTrain/{train_number}/{date}"
        return self.get(endpoint)

    def search_trains(self, from_station, to_station, date=None):
        params = {
            "from": from_station,
            "to": to_station
        }

        if date:
            params["date"] = date

        return self.get(
            "/api/searchTrainBetweenStations",
            params=params
        )
