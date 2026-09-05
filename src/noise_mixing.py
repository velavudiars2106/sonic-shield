import numpy as np
import soundfile as sf


def add_real_noise(clean_audio, noise_path, noise_level=0.3):

    # Convert clean audio to NumPy array
    clean_audio = np.asarray(clean_audio, dtype=np.float32)

    # Convert clean audio stereo -> mono
    if clean_audio.ndim == 2:
        clean_audio = np.mean(clean_audio, axis=1)

    # Load real noise
    noise, noise_sr = sf.read(noise_path, dtype="float32")

    # Convert noise stereo -> mono
    if noise.ndim == 2:
        noise = np.mean(noise, axis=1)

    # Make sure arrays are 1-dimensional
    clean_audio = clean_audio.flatten()
    noise = noise.flatten()

    # Check for empty noise file
    if len(noise) == 0:
        raise ValueError("Noise audio file is empty.")

    # Repeat noise if shorter than clean speech
    if len(noise) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise)))
        noise = np.tile(noise, repeats)

    # Cut noise to exactly match clean audio length
    noise = noise[:len(clean_audio)]

    # Normalize noise
    max_value = np.max(np.abs(noise))

    if max_value > 0:
        noise = noise / max_value

    # Add real noise
    noisy_audio = clean_audio + (noise_level * noise)

    # Prevent clipping
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

    return noisy_audio