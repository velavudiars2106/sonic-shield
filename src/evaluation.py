def read_wav(filename):

    import wave
    import numpy as np

    with wave.open(filename, "rb") as wav_file:

        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()

        frames = wav_file.readframes(num_frames)

    # 8-bit WAV
    if sample_width == 1:

        audio = np.frombuffer(
            frames,
            dtype=np.uint8
        )

        audio = (
            audio.astype(np.float32) - 128
        ) / 128.0


    # 16-bit WAV
    elif sample_width == 2:

        audio = np.frombuffer(
            frames,
            dtype=np.int16
        ).astype(np.float32)

        audio = audio / 32768.0


    # 24-bit WAV
    elif sample_width == 3:

        raw = np.frombuffer(
            frames,
            dtype=np.uint8
        )

        raw = raw.reshape(
            -1,
            3
        )

        audio = (
            raw[:, 0].astype(np.int32)
            |
            raw[:, 1].astype(np.int32) << 8
            |
            raw[:, 2].astype(np.int32) << 16
        )

        # Convert 24-bit signed integer
        audio = np.where(
            audio & 0x800000,
            audio - 0x1000000,
            audio
        )

        audio = (
            audio.astype(np.float32)
            / 8388608.0
        )


    # 32-bit WAV
    elif sample_width == 4:

        audio = np.frombuffer(
            frames,
            dtype=np.int32
        ).astype(np.float32)

        audio = (
            audio
            / 2147483648.0
        )


    else:

        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width * 8}-bit"
        )


    # Convert stereo to mono
    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(
            axis=1
        )


    return audio, sample_rate
def calculate_snr(clean_audio, test_audio):
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    """

    import numpy as np

    # Make both arrays the same length
    min_length = min(len(clean_audio), len(test_audio))

    clean_audio = clean_audio[:min_length]
    test_audio = test_audio[:min_length]

    # Noise/error signal
    noise = clean_audio - test_audio

    # Calculate signal power
    signal_power = np.mean(clean_audio ** 2)

    # Calculate noise power
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power < 1e-12:
        return 100.0

    snr = 10 * np.log10(
        (signal_power + 1e-12) /
        (noise_power + 1e-12)
    )

    return float(snr)