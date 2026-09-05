import os
import sys


# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from src.preprocessing import preprocess_audio
from src.noise_classifier import classify_noise
from src.vad import voice_activity_detection
from src.spectral_subtraction import reduce_noise


def enhance_speech(
    input_file,
    output_file
):

    print("\n================================")
    print("     SONIC SHIELD AI PIPELINE")
    print("================================\n")


    # -----------------------------
    # CHECK INPUT FILE
    # -----------------------------

    if not os.path.exists(input_file):

        print("ERROR: Input file not found!")

        print("Expected file:")

        print(input_file)

        return None


    # -----------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------

    output_directory = os.path.dirname(
        output_file
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )


    # -----------------------------
    # STEP 1
    # PREPROCESSING
    # -----------------------------

    print("STEP 1: Preprocessing...")

    audio, sample_rate = preprocess_audio(
        input_file
    )

    print(
        "Sample Rate:",
        sample_rate
    )

    print(
        "Audio Samples:",
        len(audio)
    )


    # -----------------------------
    # STEP 2
    # NOISE CLASSIFICATION
    # -----------------------------

    print(
        "\nSTEP 2: Noise Classification..."
    )

    noise_result = classify_noise(
        audio,
        sample_rate
    )

    noise_type = noise_result[
        "noise_type"
    ]

    print("\n--- NOISE ANALYSIS ---")

    print(
        "Energy:",
        round(
            noise_result["energy"],
            6
        )
    )

    print(
        "Spectral Centroid:",
        round(
            noise_result[
                "spectral_centroid"
            ],
            2
        ),
        "Hz"
    )

    print(
        "Detected Noise Type:",
        noise_type
    )


    # -----------------------------
    # STEP 3
    # VOICE ACTIVITY DETECTION
    # -----------------------------

    print(
        "\nSTEP 3: Voice Activity Detection..."
    )

    vad_result = voice_activity_detection(
        audio,
        sample_rate
    )

    voice_frames = sum(
        vad_result
    )

    silence_frames = (
        len(vad_result)
        - voice_frames
    )

    print(
        "Voice Frames:",
        voice_frames
    )

    print(
        "Silence Frames:",
        silence_frames
    )


    # -----------------------------
    # STEP 4
    # NOISE REDUCTION
    # -----------------------------

    print(
        "\nSTEP 4: Noise Reduction..."
    )

    noise_strength = 1.5

    print(
        "Noise Reduction Strength:",
        noise_strength
    )

    enhanced_file = reduce_noise(
        input_file=input_file,
        output_file=output_file,
        noise_strength=noise_strength
    )


    # -----------------------------
    # FINAL RESULT
    # -----------------------------

    print("\n================================")

    print(
        "SPEECH ENHANCEMENT COMPLETE"
    )

    print("================================")

    print(
        "\nInput File:"
    )

    print(input_file)

    print(
        "\nDetected Noise:"
    )

    print(noise_type)

    print(
        "\nOutput File:"
    )

    print(enhanced_file)

    return enhanced_file


# =================================
# MAIN PROGRAM
# =================================

if __name__ == "__main__":

    input_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "noisy",
        "noisy_voice1.wav"
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "enhanced",
        "pipeline_enhanced_voice1.wav"
    )

    enhance_speech(
        input_file,
        output_file
    )