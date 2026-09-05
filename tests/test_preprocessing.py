import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_audio


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_speech",
    "voice1.wav"
)


def test_preprocessing():

    print("\n================================")
    print("     SONIC SHIELD PREPROCESSING")
    print("================================\n")

    if not os.path.exists(INPUT_FILE):
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        return

    print("Input file:")
    print(INPUT_FILE)

    print("\nRunning preprocessing...\n")

    try:

        audio, sample_rate = preprocess_audio(
            INPUT_FILE
        )

        print("--------------------------------")
        print("PREPROCESSING RESULTS")
        print("--------------------------------")

        print("Sample rate :", sample_rate)
        print("Audio samples:", len(audio))
        print("Audio shape :", audio.shape)
        print("Audio dtype :", audio.dtype)

        print("\n================================")
        print("✓ PREPROCESSING TEST PASSED")
        print("================================\n")

    except Exception as e:

        print("\n================================")
        print("ERROR")
        print("================================")

        print(type(e).__name__)
        print(str(e))


if __name__ == "__main__":
    test_preprocessing()