import numpy as np
import soundfile as sf


def preprocess_audio(input_file, target_sample_rate=48000):

    print("Loading audio...")

    audio, sample_rate = sf.read(input_file)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Normalize audio
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    print("Audio preprocessing completed.")

    return audio, sample_rate