from src.data.railkit_client import RailKitClient


client = RailKitClient()

data = client.get("/api/station/NDLS")

print("RailKit request successful")
print("Response type:", type(data).__name__)

if isinstance(data, dict):
    print("Response keys:", list(data.keys()))
else:
    print("Response received")
