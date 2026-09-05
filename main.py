import os
import numpy as np
import soundfile as sf

from src.preprocessing import preprocess_audio
from src.noise_mixing import add_real_noise
from src.noise_classifier import classify_noise
from src.vad import voice_activity_detection
from src.spectral_subtraction import spectral_subtraction
from src.wiener_filter import wiener_denoise


# ==========================================
# SONIC SHIELD - COMPLETE AUDIO PIPELINE
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# INPUT FILES
# ==========================================

clean_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "clean_speech",
    "voice1.wav"
)

noise_file = os.path.join(
    PROJECT_ROOT,
    "data",
    "noise",
    "fan.wav"
)


# ==========================================
# OUTPUT FOLDERS
# ==========================================

noisy_folder = os.path.join(
    PROJECT_ROOT,
    "data",
    "noisy"
)

enhanced_folder = os.path.join(
    PROJECT_ROOT,
    "data",
    "enhanced"
)

os.makedirs(noisy_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)


# ==========================================
# OUTPUT FILES
# ==========================================

noisy_file = os.path.join(
    noisy_folder,
    "noisy_voice1.wav"
)

spectral_file = os.path.join(
    enhanced_folder,
    "spectral_voice1.wav"
)

final_file = os.path.join(
    enhanced_folder,
    "sonic_shield_output.wav"
)


# ==========================================
# START
# ==========================================

print("\n========================================")
print("       SONIC SHIELD STARTING")
print("========================================")


# ==========================================
# STEP 1 - PREPROCESS CLEAN SPEECH
# ==========================================

print("\n[1] PREPROCESSING AUDIO...")

try:

    clean_audio, sample_rate = preprocess_audio(
        clean_file
    )

    print("Audio preprocessing completed.")
    print("Sample rate:", sample_rate)
    print("Samples:", len(clean_audio))

except Exception as e:

    print("\nERROR during preprocessing:")
    print(e)
    raise SystemExit


# ==========================================
# STEP 2 - ADD REAL NOISE
# ==========================================

print("\n[2] ADDING REAL NOISE...")

try:

    noisy_audio = add_real_noise(
        clean_audio,
        noise_file,
        noise_level=0.3
    )

    sf.write(
        noisy_file,
        noisy_audio,
        sample_rate,
        subtype="PCM_16"
    )

    print("Noisy audio saved:")
    print(noisy_file)

except Exception as e:

    print("\nERROR during noise mixing:")
    print(e)
    raise SystemExit


# ==========================================
# STEP 3 - VOICE ACTIVITY DETECTION
# ==========================================

print("\n[3] VOICE ACTIVITY DETECTION...")

try:

    vad_result = voice_activity_detection(
        noisy_audio,
        sample_rate
    )

    vad_result = np.asarray(vad_result)

    voice_frames = np.sum(vad_result == 1)
    silence_frames = np.sum(vad_result == 0)

    print("Total frames:", len(vad_result))
    print("Voice frames:", voice_frames)
    print("Silence frames:", silence_frames)

except Exception as e:

    print("\nERROR during VAD:")
    print(e)

    # Continue the pipeline even if VAD has a problem
    voice_frames = 0
    silence_frames = 0


# ==========================================
# STEP 4 - NOISE CLASSIFICATION
# ==========================================

print("\n[4] NOISE CLASSIFICATION...")

try:

    # IMPORTANT:
    # classify_noise() needs audio + sample rate.
    # Therefore load/preprocess the noise first.

    noise_audio, noise_sample_rate = preprocess_audio(
        noise_file
    )

    noise_type = classify_noise(
        noise_audio,
        noise_sample_rate
    )

    print("Detected noise:", noise_type)

except Exception as e:

    print("\nERROR during noise classification:")
    print(e)

    noise_type = "unknown"


# ==========================================
# STEP 5 - SPECTRAL SUBTRACTION
# ==========================================

print("\n[5] SPECTRAL SUBTRACTION...")

try:

    spectral_subtraction(
        noisy_file,
        spectral_file,
        noise_strength=1.2
    )

    print("Spectral subtraction completed.")
    print("Spectral output saved:")
    print(spectral_file)

except Exception as e:

    print("\nERROR during spectral subtraction:")
    print(e)
    raise SystemExit


# ==========================================
# STEP 6 - WIENER FILTER
# ==========================================

print("\n[6] WIENER FILTERING...")

try:

    spectral_audio, spectral_sample_rate = sf.read(
        spectral_file
    )

    # Convert stereo to mono
    if spectral_audio.ndim > 1:

        spectral_audio = np.mean(
            spectral_audio,
            axis=1
        )

    final_audio = wiener_denoise(
        spectral_audio,
        spectral_sample_rate
    )

    # Prevent clipping
    final_audio = np.asarray(
        final_audio,
        dtype=np.float32
    )

    max_value = np.max(
        np.abs(final_audio)
    )

    if max_value > 1.0:

        final_audio = (
            final_audio / max_value
        )

    sf.write(
        final_file,
        final_audio,
        spectral_sample_rate,
        subtype="PCM_16"
    )

    print("Wiener filtering completed.")
    print("Final enhanced audio saved:")
    print(final_file)

except Exception as e:

    print("\nERROR during Wiener filtering:")
    print(e)
    raise SystemExit


# ==========================================
# FINAL RESULT
# ==========================================

print("\n========================================")
print("   SONIC SHIELD PIPELINE COMPLETED!")
print("========================================")

print("\nNoise detected:", noise_type)

print("Voice frames:", voice_frames)
print("Silence frames:", silence_frames)

print("\nFinal enhanced audio:")
print(final_file)

print("\n========================================")
print("       ✓ SUCCESS")
print("========================================\n")