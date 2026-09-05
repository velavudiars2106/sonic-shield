import soundfile as sf
import numpy as np


def load_audio(file_path):
    """
    Load an audio file.
    """

    audio, sample_rate = sf.read(file_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    print("Audio loaded successfully")
    print("Sample rate:", sample_rate)
    print("Duration:", round(len(audio) / sample_rate, 2), "seconds")

    return audio, sample_rate


def save_audio(file_path, audio, sample_rate):
    """
    Save audio to a WAV file.
    """

    sf.write(file_path, audio, sample_rate)

    print("Audio saved successfully:", file_path)


def get_audio_info(audio, sample_rate):

    duration = len(audio) / sample_rate

    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "samples": len(audio)
    }