import os
import sys

# ---------------------------------------------
# PROJECT ROOT
# ---------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.vad import voice_activity_detection


# ---------------------------------------------
# INPUT FILE
# ---------------------------------------------

input_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "noisy",
    "noisy_voice1.wav"
)


# ---------------------------------------------
# CHECK FILE
# ---------------------------------------------

if not os.path.exists(input_file):

    print("\nERROR: Audio file not found:")
    print(input_file)
    sys.exit(1)


# ---------------------------------------------
# LOAD AUDIO
# ---------------------------------------------

import soundfile as sf

audio, sample_rate = sf.read(
    input_file,
    dtype="float32"
)


# ---------------------------------------------
# VAD TEST
# ---------------------------------------------

print("\n================================")
print("       SONIC SHIELD VAD TEST")
print("================================\n")

print("Input file:")
print(input_file)

print("\nSample rate:", sample_rate)
print("Audio samples:", len(audio))

print("\nRunning Voice Activity Detection...\n")


try:

    result = voice_activity_detection(
        audio,
        sample_rate
    )

    voice_frames = sum(result)
    silence_frames = len(result) - voice_frames

    print("--------------------------------")
    print("VAD RESULTS")
    print("--------------------------------")

    print("Total frames   :", len(result))
    print("Voice frames   :", voice_frames)
    print("Silence frames :", silence_frames)

    print("\n================================")
    print("✓ VAD TEST COMPLETED")
    print("================================\n")


except Exception as e:

    print("\n================================")
    print("ERROR")
    print("================================")

    print(type(e).__name__)
    print(str(e))