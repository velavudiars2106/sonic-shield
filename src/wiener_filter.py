import numpy as np
from scipy import signal


def wiener_denoise(audio, sample_rate):
    """
    Simple Wiener-based noise reduction.
    """

    # Convert audio to float
    audio = audio.astype(np.float32)

    # Apply Wiener filter
    enhanced = signal.wiener(audio, mysize=29)

    # Normalize audio
    max_value = np.max(np.abs(enhanced))

    if max_value > 0:
        enhanced = enhanced / max_value

    # Prevent clipping
    enhanced = np.clip(enhanced, -1.0, 1.0)

    return enhanced