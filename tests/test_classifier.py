import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.noise_classifier import classify_noise


def test_classifier():

    files = [
        "data/noise/fan.wav",
        "data/noise/machine.wav",
        "data/noise/traffic.wav"
    ]

    print("\n===== NOISE CLASSIFIER TEST =====")

    for relative_path in files:

        file_path = os.path.join(
            PROJECT_ROOT,
            relative_path
        )

        print("\n----------------------------")
        print("Testing:", relative_path)

        if not os.path.exists(file_path):
            print("⚠ File not found. Skipping.")
            continue

        try:
            result = classify_noise(file_path)
            print("Final Result:", result)

        except Exception as error:
            print("ERROR:", error)


if __name__ == "__main__":
    test_classifier()