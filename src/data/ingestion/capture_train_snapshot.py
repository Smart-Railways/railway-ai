from datetime import datetime

from src.data.ingestion.railkit_capture import RailKitCapture


def capture(train_number):
    date = datetime.now().strftime("%d-%m-%Y")

    capture = RailKitCapture()

    path = capture.capture_live_train(
        train_number=str(train_number),
        date=date,
    )

    print("TRAIN SNAPSHOT CAPTURED")
    print("Train:", train_number)
    print("Date:", date)
    print("Saved:", path)


if __name__ == "__main__":
    capture("12301")
