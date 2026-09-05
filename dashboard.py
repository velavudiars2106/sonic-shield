import os
import tempfile

import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_audio
from src.noise_classifier import classify_noise
from src.vad import voice_activity_detection
from src.spectral_subtraction import spectral_subtraction
from src.wiener_filter import wiener_denoise


# ============================================================
# SONIC SHIELD
# INTERACTIVE SPEECH ENHANCEMENT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Sonic Shield",
    page_icon="🎧",
    layout="wide"
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_snr(clean, test):

    length = min(
        len(clean),
        len(test)
    )

    clean = clean[:length]
    test = test[:length]

    noise = test - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-12:
        return 100.0

    return 10 * np.log10(
        signal_power / noise_power
    )


def calculate_rms(audio):

    if audio is None or len(audio) == 0:
        return 0.0

    return float(
        np.sqrt(np.mean(audio ** 2))
    )


def get_noise_type(audio, sample_rate):

    try:

        result = classify_noise(
            audio,
            sample_rate
        )

        if isinstance(result, dict):

            return result.get(
                "noise_type",
                "UNKNOWN"
            )

        return str(result)

    except Exception:

        return "UNKNOWN"


def get_spectral_centroid(audio, sample_rate):

    if len(audio) == 0:
        return 0.0

    spectrum = np.abs(
        np.fft.rfft(audio)
    )

    frequencies = np.fft.rfftfreq(
        len(audio),
        1 / sample_rate
    )

    total = np.sum(spectrum)

    if total <= 0:
        return 0.0

    return float(
        np.sum(
            frequencies * spectrum
        ) / total
    )


# ============================================================
# HEADER
# ============================================================

st.title("🎧 SONIC SHIELD")

st.subheader(
    "AI-Based Speech Enhancement & Intelligent Noise Reduction"
)

st.write(
    "Upload a noisy WAV recording and let Sonic Shield "
    "analyze and enhance the speech."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Processing Settings")

    noise_strength = st.slider(
        "Noise Reduction Strength",
        min_value=0.5,
        max_value=2.0,
        value=1.2,
        step=0.1
    )

    st.info(
        "Higher values apply stronger noise reduction."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("🎤 1. Upload Noisy Speech")

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


if uploaded_file is None:

    st.info(
        "👆 Upload a noisy WAV file to begin."
    )

    st.stop()


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".wav"
) as temp_input:

    temp_input.write(
        uploaded_file.getbuffer()
    )

    input_path = temp_input.name


# ============================================================
# LOAD INPUT AUDIO
# ============================================================

try:

    input_audio, sample_rate = sf.read(
        input_path,
        dtype="float32"
    )

except Exception as error:

    st.error(
        f"Could not read the WAV file: {error}"
    )

    os.remove(input_path)

    st.stop()


# Convert stereo to mono

if input_audio.ndim > 1:

    input_audio = np.mean(
        input_audio,
        axis=1
    )


# ============================================================
# INPUT INFORMATION
# ============================================================

st.header("🎵 2. Input Audio")

st.audio(
    uploaded_file,
    format="audio/wav"
)

duration = (
    len(input_audio) /
    sample_rate
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Sample Rate",
        f"{sample_rate} Hz"
    )

with col2:

    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

with col3:

    st.metric(
        "Audio Samples",
        f"{len(input_audio):,}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

st.header("🤖 3. AI Analysis")

analyze_button = st.button(
    "🔍 Analyze Noise",
    use_container_width=True
)


if analyze_button:

    with st.spinner(
        "Analyzing audio..."
    ):

        noise_type = get_noise_type(
            input_audio,
            sample_rate
        )

        rms = calculate_rms(
            input_audio
        )

        centroid = get_spectral_centroid(
            input_audio,
            sample_rate
        )

        vad_result = voice_activity_detection(
            input_audio,
            sample_rate
        )

        vad_result = np.asarray(
            vad_result
        )

        voice_frames = int(
            np.sum(vad_result == 1)
        )

        total_frames = len(
            vad_result
        )

        st.session_state[
            "noise_type"
        ] = noise_type

        st.session_state[
            "rms"
        ] = rms

        st.session_state[
            "centroid"
        ] = centroid

        st.session_state[
            "voice_frames"
        ] = voice_frames

        st.session_state[
            "total_frames"
        ] = total_frames

        st.success(
            "✓ Audio analysis completed."
        )


# ============================================================
# SHOW ANALYSIS
# ============================================================

if "noise_type" in st.session_state:

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Detected Noise",
            st.session_state[
                "noise_type"
            ]
        )

    with col2:

        st.metric(
            "Audio Level",
            f"{st.session_state['rms']:.4f}"
        )

    with col3:

        st.metric(
            "Spectral Centroid",
            f"{st.session_state['centroid']:.1f} Hz"
        )

    with col4:

        st.metric(
            "Voice Frames",
            st.session_state[
                "voice_frames"
            ]
        )


# ============================================================
# ENHANCEMENT
# ============================================================

st.divider()

st.header("✨ 4. Speech Enhancement")

enhance_button = st.button(
    "🚀 ENHANCE SPEECH",
    type="primary",
    use_container_width=True
)


if enhance_button:

    output_dir = tempfile.mkdtemp()

    spectral_output = os.path.join(
        output_dir,
        "spectral_enhanced.wav"
    )

    final_output = os.path.join(
        output_dir,
        "sonic_shield_enhanced.wav"
    )

    try:

        with st.spinner(
            "Running Sonic Shield enhancement..."
        ):

            # ------------------------------------
            # Spectral subtraction
            # ------------------------------------

            spectral_subtraction(
                input_path,
                spectral_output,
                noise_strength=noise_strength
            )

            # ------------------------------------
            # Load spectral output
            # ------------------------------------

            spectral_audio, spectral_sr = sf.read(
                spectral_output,
                dtype="float32"
            )

            if spectral_audio.ndim > 1:

                spectral_audio = np.mean(
                    spectral_audio,
                    axis=1
                )

            # ------------------------------------
            # Wiener filtering
            # ------------------------------------

            final_audio = wiener_denoise(
                spectral_audio,
                spectral_sr
            )

            final_audio = np.asarray(
                final_audio,
                dtype=np.float32
            )

            # Prevent clipping

            maximum = np.max(
                np.abs(final_audio)
            )

            if maximum > 1:

                final_audio = (
                    final_audio / maximum
                )

            # ------------------------------------
            # Save final output
            # ------------------------------------

            sf.write(
                final_output,
                final_audio,
                spectral_sr,
                subtype="PCM_16"
            )

        st.session_state[
            "enhanced_file"
        ] = final_output

        st.success(
            "🎉 Speech enhancement completed!"
        )

    except Exception as error:

        st.error(
            f"Enhancement failed: {error}"
        )


# ============================================================
# ENHANCED RESULT
# ============================================================

if "enhanced_file" in st.session_state:

    enhanced_file = st.session_state[
        "enhanced_file"
    ]

    enhanced_audio, enhanced_sr = sf.read(
        enhanced_file,
        dtype="float32"
    )

    if enhanced_audio.ndim > 1:

        enhanced_audio = np.mean(
            enhanced_audio,
            axis=1
        )


    # ==========================================
    # AUDIO OUTPUT
    # ==========================================

    st.divider()

    st.header("🎧 5. Enhanced Speech")

    with open(
        enhanced_file,
        "rb"
    ) as audio_file:

        enhanced_bytes = (
            audio_file.read()
        )

    st.audio(
        enhanced_bytes,
        format="audio/wav"
    )


    # ==========================================
    # DOWNLOAD
    # ==========================================

    st.download_button(
        label="⬇️ Download Enhanced WAV",
        data=enhanced_bytes,
        file_name="sonic_shield_enhanced.wav",
        mime="audio/wav",
        use_container_width=True
    )


    # ==========================================
    # PERFORMANCE
    # ==========================================

    st.divider()

    st.header("📊 6. Enhancement Results")

    st.info(
        "SNR comparison requires a clean reference recording. "
        "For uploaded real-world speech, the dashboard reports "
        "audio-level and enhancement information instead."
    )

    input_rms = calculate_rms(
        input_audio
    )

    enhanced_rms = calculate_rms(
        enhanced_audio
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Input RMS",
            f"{input_rms:.4f}"
        )

    with col2:

        st.metric(
            "Enhanced RMS",
            f"{enhanced_rms:.4f}"
        )

    with col3:

        st.metric(
            "Sample Rate",
            f"{enhanced_sr} Hz"
        )


    # ==========================================
    # WAVEFORM
    # ==========================================

    st.divider()

    st.header("📈 7. Waveform Comparison")

    max_samples = 100000

    noisy_display = (
        input_audio[:max_samples]
    )

    enhanced_display = (
        enhanced_audio[:max_samples]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("🔴 Noisy Input")

        fig1, ax1 = plt.subplots(
            figsize=(8, 3)
        )

        time1 = (
            np.arange(
                len(noisy_display)
            ) / sample_rate
        )

        ax1.plot(
            time1,
            noisy_display
        )

        ax1.set_xlabel(
            "Time (seconds)"
        )

        ax1.set_ylabel(
            "Amplitude"
        )

        ax1.grid(True)

        st.pyplot(fig1)

        plt.close(fig1)


    with col2:

        st.write("🟢 Enhanced Output")

        fig2, ax2 = plt.subplots(
            figsize=(8, 3)
        )

        time2 = (
            np.arange(
                len(enhanced_display)
            ) / enhanced_sr
        )

        ax2.plot(
            time2,
            enhanced_display
        )

        ax2.set_xlabel(
            "Time (seconds)"
        )

        ax2.set_ylabel(
            "Amplitude"
        )

        ax2.grid(True)

        st.pyplot(fig2)

        plt.close(fig2)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ SONIC SHIELD | AI Speech Enhancement System | SIH 2026"
)


# ============================================================
# CLEAN TEMP INPUT
# ============================================================

# Don't delete input_path here because Streamlit
# reruns the script and enhancement may still need it.