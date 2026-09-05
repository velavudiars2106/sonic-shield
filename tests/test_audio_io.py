import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio_io import load_audio, save_audio


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_speech",
    "voice1.wav"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "enhanced",
    "audio_io_test.wav"
)


def test_audio_io():

    print("\n================================")
    print("       SONIC SHIELD AUDIO I/O")
    print("================================\n")

    if not os.path.exists(INPUT_FILE):
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        return

    print("Loading:")
    print(INPUT_FILE)

    audio, sample_rate = load_audio(INPUT_FILE)

    print("\nAudio loaded successfully.")
    print("Sample rate:", sample_rate)
    print("Audio samples:", len(audio))

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    save_audio(
        OUTPUT_FILE,
        audio,
        sample_rate
    )

    print("\nAudio saved successfully.")
    print("Output:", OUTPUT_FILE)

    if os.path.exists(OUTPUT_FILE):
        print("\n✓ AUDIO I/O TEST PASSED")
    else:
        print("\n✗ AUDIO I/O TEST FAILED")


if __name__ == "__main__":
    test_audio_io()