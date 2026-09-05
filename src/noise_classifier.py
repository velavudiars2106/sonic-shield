import numpy as np


def classify_noise(audio, sample_rate):

    # Calculate signal energy
    energy = np.mean(audio ** 2)

    # Calculate spectral centroid
    spectrum = np.abs(np.fft.rfft(audio))

    frequencies = np.fft.rfftfreq(
        len(audio),
        d=1 / sample_rate
    )

    if np.sum(spectrum) == 0:
        spectral_centroid = 0
    else:
        spectral_centroid = np.sum(
            frequencies * spectrum
        ) / np.sum(spectrum)

    # Simple rule-based classification
    if spectral_centroid < 800:
        noise_type = "FAN"

    elif spectral_centroid < 2500:
        noise_type = "MACHINE"

    else:
        noise_type = "TRAFFIC"

    result = {
        "noise_type": noise_type,
        "energy": float(energy),
        "spectral_centroid": float(spectral_centroid)
    }

    return result