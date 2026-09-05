import os
import sys
import numpy as np

# -------------------------------------------------
# PROJECT PATH
# -------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.evaluation import read_wav, calculate_snr


# -------------------------------------------------
# AUDIO LENGTH MATCHING
# -------------------------------------------------

def match_length(audio1, audio2):
    """
    Make two audio signals the same length.
    """

    min_length = min(
        len(audio1),
        len(audio2)
    )

    return (
        audio1[:min_length],
        audio2[:min_length]
    )


# -------------------------------------------------
# MAIN EVALUATION
# -------------------------------------------------

def test_evaluation():

    # ---------------------------------------------
    # FILE PATHS
    # ---------------------------------------------

    clean_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "clean_speech",
        "voice1.wav"
    )

    noisy_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "noisy",
        "noisy_voice1.wav"
    )

    enhanced_file = os.path.join(
        PROJECT_ROOT,
        "data",
        "enhanced",
        "pipeline_enhanced_voice1.wav"
    )

    print("\n================================")
    print("     SONIC SHIELD AUDIO")
    print("        EVALUATION")
    print("================================\n")

    # ---------------------------------------------
    # CHECK FILES
    # ---------------------------------------------

    print("Checking files...\n")

    for name, path in [
        ("Clean", clean_file),
        ("Noisy", noisy_file),
        ("Enhanced", enhanced_file)
    ]:

        if not os.path.exists(path):

            print("ERROR: File not found:")
            print(path)
            return

        print(f"✓ {name} file found")

    # ---------------------------------------------
    # READ AUDIO
    # ---------------------------------------------

    print("\nLoading audio files...\n")

    clean_audio, clean_sr = read_wav(
        clean_file
    )

    noisy_audio, noisy_sr = read_wav(
        noisy_file
    )

    enhanced_audio, enhanced_sr = read_wav(
        enhanced_file
    )

    # ---------------------------------------------
    # AUDIO INFORMATION
    # ---------------------------------------------

    print("Audio Information")
    print("----------------------------")

    print(
        f"Clean    : {len(clean_audio)} samples"
    )

    print(
        f"Noisy    : {len(noisy_audio)} samples"
    )

    print(
        f"Enhanced : {len(enhanced_audio)} samples"
    )

    print()

    print(
        f"Clean sample rate    : {clean_sr} Hz"
    )

    print(
        f"Noisy sample rate    : {noisy_sr} Hz"
    )

    print(
        f"Enhanced sample rate : {enhanced_sr} Hz"
    )

    # ---------------------------------------------
    # SAMPLE RATE CHECK
    # ---------------------------------------------

    if not (
        clean_sr ==
        noisy_sr ==
        enhanced_sr
    ):

        print(
            "\nWARNING: Sample rates are different."
        )

    # ---------------------------------------------
    # MATCH AUDIO LENGTH
    # ---------------------------------------------

    clean_noisy, noisy_for_snr = match_length(
        clean_audio,
        noisy_audio
    )

    clean_enhanced, enhanced_for_snr = match_length(
        clean_audio,
        enhanced_audio
    )

    # ---------------------------------------------
    # CALCULATE SNR
    # ---------------------------------------------

    print("\nCalculating SNR...\n")

    snr_noisy = calculate_snr(
        clean_noisy,
        noisy_for_snr
    )

    snr_enhanced = calculate_snr(
        clean_enhanced,
        enhanced_for_snr
    )

    improvement = (
        snr_enhanced -
        snr_noisy
    )

    # ---------------------------------------------
    # RESULTS
    # ---------------------------------------------

    print("\n================================")
    print("          SNR RESULTS")
    print("================================")

    print(
        f"Noisy Audio SNR     : "
        f"{snr_noisy:.2f} dB"
    )

    print(
        f"Enhanced Audio SNR  : "
        f"{snr_enhanced:.2f} dB"
    )

    print(
        f"SNR Improvement     : "
        f"{improvement:.2f} dB"
    )

    print("================================")

    # ---------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------

    if improvement > 0:

        print(
            "\n✓ Speech enhancement improved SNR."
        )

    elif improvement == 0:

        print(
            "\n⚠ No SNR improvement detected."
        )

    else:

        print(
            "\n⚠ SNR decreased after enhancement."
        )

    print(
        "\n✓ EVALUATION COMPLETED"
    )

    print("================================\n")


# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    test_evaluation()