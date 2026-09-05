import numpy as np
import soundfile as sf


def reduce_noise(
    input_file,
    output_file,
    noise_strength=1.5
):

    print("Loading noisy audio...")

    audio, sample_rate = sf.read(input_file)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    print("Estimating noise...")

    # Use first part of audio as noise estimate
    noise_samples = int(
        min(len(audio), sample_rate * 0.5)
    )

    noise = audio[:noise_samples]

    noise_energy = np.mean(
        noise ** 2
    )

    # Frame processing
    frame_size = 1024
    hop_size = 512

    enhanced_audio = np.zeros_like(audio)

    window = np.hanning(frame_size)

    for start in range(
        0,
        len(audio) - frame_size,
        hop_size
    ):

        frame = audio[
            start:start + frame_size
        ]

        windowed_frame = frame * window

        spectrum = np.fft.rfft(
            windowed_frame
        )

        magnitude = np.abs(spectrum)

        phase = np.angle(spectrum)

        # Estimate noise magnitude
        noise_magnitude = np.sqrt(
            noise_energy
        )

        # Spectral subtraction
        enhanced_magnitude = (
            magnitude
            - noise_strength
            * noise_magnitude
        )

        # Prevent negative values
        enhanced_magnitude = np.maximum(
            enhanced_magnitude,
            0
        )

        # Reconstruct spectrum
        enhanced_spectrum = (
            enhanced_magnitude
            * np.exp(1j * phase)
        )

        enhanced_frame = np.fft.irfft(
            enhanced_spectrum
        )

        # Add enhanced frame
        enhanced_audio[
            start:start + frame_size
        ] += enhanced_frame * window

    # Normalize
    max_value = np.max(
        np.abs(enhanced_audio)
    )

    if max_value > 0:
        enhanced_audio = (
            enhanced_audio
            / max_value
            * 0.95
        )

    # Save
    sf.write(
        output_file,
        enhanced_audio,
        sample_rate
    )

    print("Noise reduction completed.")

    return output_file


# Compatibility function
def spectral_subtraction(
    input_file,
    output_file,
    noise_strength=1.5
):

    return reduce_noise(
        input_file,
        output_file,
        noise_strength
    )