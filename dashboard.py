import os
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# SONIC SHIELD DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Sonic Shield",
    page_icon="🎧",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CLEAN_FILE = os.path.join(
    DATA_DIR,
    "clean_speech",
    "voice1.wav"
)

NOISY_FILE = os.path.join(
    DATA_DIR,
    "noisy",
    "noisy_voice1.wav"
)

ENHANCED_FILE = os.path.join(
    DATA_DIR,
    "enhanced",
    "sonic_shield_output.wav"
)


# ============================================================
# FUNCTIONS
# ============================================================

def load_audio(filename):

    if not os.path.exists(filename):
        return None, None

    try:
        audio, sample_rate = sf.read(filename)

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio, sample_rate

    except Exception:
        return None, None


def calculate_snr(clean, test):

    min_length = min(len(clean), len(test))

    clean = clean[:min_length]
    test = test[:min_length]

    noise = test - clean

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return 100.0

    return 10 * np.log10(
        signal_power / noise_power
    )


def calculate_rms(audio):

    if audio is None or len(audio) == 0:
        return 0

    return np.sqrt(
        np.mean(audio ** 2)
    )


def detect_noise_type():

    # Current pipeline uses fan.wav.
    # This can later be connected to your ML classifier.

    return "FAN / MACHINE"


# ============================================================
# HEADER
# ============================================================

st.title("🎧 SONIC SHIELD")

st.subheader(
    "AI-Based Speech Enhancement & Intelligent Noise Reduction"
)

st.write(
    "Real-time analysis of noisy speech and enhanced speech."
)


st.divider()


# ============================================================
# LOAD AUDIO
# ============================================================

clean_audio, clean_sr = load_audio(CLEAN_FILE)

noisy_audio, noisy_sr = load_audio(NOISY_FILE)

enhanced_audio, enhanced_sr = load_audio(ENHANCED_FILE)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("🟢 System Status")

col1, col2, col3 = st.columns(3)

with col1:

    if clean_audio is not None:
        st.success("Clean speech available")
    else:
        st.error("Clean speech missing")


with col2:

    if noisy_audio is not None:
        st.success("Noisy speech available")
    else:
        st.error("Noisy speech missing")


with col3:

    if enhanced_audio is not None:
        st.success("Enhanced speech available")
    else:
        st.error("Enhanced speech missing")


# ============================================================
# CHECK FILES
# ============================================================

if clean_audio is None:

    st.error(
        "Clean audio not found: "
        + CLEAN_FILE
    )

    st.stop()


if noisy_audio is None:

    st.error(
        "Noisy audio not found. "
        "Run main.py first."
    )

    st.stop()


if enhanced_audio is None:

    st.warning(
        "Enhanced audio not found. "
        "Run main.py first."
    )

    st.stop()


# ============================================================
# AUDIO INFORMATION
# ============================================================

st.divider()

st.subheader("🎵 Audio Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Sample Rate",
        f"{clean_sr} Hz"
    )

with col2:

    duration = len(clean_audio) / clean_sr

    st.metric(
        "Duration",
        f"{duration:.2f} sec"
    )

with col3:

    st.metric(
        "Samples",
        f"{len(clean_audio):,}"
    )


# ============================================================
# NOISE INFORMATION
# ============================================================

st.divider()

st.subheader("🔊 Noise Analysis")

noise_type = detect_noise_type()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Detected Noise",
        noise_type
    )

with col2:

    noisy_rms = calculate_rms(noisy_audio)

    st.metric(
        "Noise Audio Level",
        f"{noisy_rms:.4f}"
    )


# ============================================================
# SNR CALCULATION
# ============================================================

snr_noisy = calculate_snr(
    clean_audio,
    noisy_audio
)

snr_enhanced = calculate_snr(
    clean_audio,
    enhanced_audio
)

snr_improvement = (
    snr_enhanced - snr_noisy
)


# ============================================================
# SNR DASHBOARD
# ============================================================

st.divider()

st.subheader("📊 Speech Enhancement Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Noisy SNR",
        f"{snr_noisy:.2f} dB"
    )

with col2:

    st.metric(
        "Enhanced SNR",
        f"{snr_enhanced:.2f} dB"
    )

with col3:

    st.metric(
        "SNR Improvement",
        f"{snr_improvement:.2f} dB"
    )


if snr_improvement > 0:

    st.success(
        "✓ Speech enhancement improved the SNR."
    )

else:

    st.warning(
        "⚠ Enhancement did not improve the SNR."
    )


# ============================================================
# AUDIO PLAYERS
# ============================================================

st.divider()

st.subheader("🎧 Audio Comparison")

col1, col2 = st.columns(2)

with col1:

    st.write("### 🔴 Noisy Speech")

    st.audio(
        NOISY_FILE,
        format="audio/wav"
    )


with col2:

    st.write("### 🟢 Enhanced Speech")

    st.audio(
        ENHANCED_FILE,
        format="audio/wav"
    )


# ============================================================
# WAVEFORMS
# ============================================================

st.divider()

st.subheader("📈 Waveform Comparison")


# Limit displayed samples for performance

MAX_SAMPLES = 100000


clean_display = clean_audio[:MAX_SAMPLES]

noisy_display = noisy_audio[:MAX_SAMPLES]

enhanced_display = enhanced_audio[:MAX_SAMPLES]


# ------------------------------------------------------------
# Noisy waveform
# ------------------------------------------------------------

st.write("🔴 Noisy Speech Waveform")

fig1, ax1 = plt.subplots(
    figsize=(12, 3)
)

time_noisy = np.arange(
    len(noisy_display)
) / noisy_sr

ax1.plot(
    time_noisy,
    noisy_display
)

ax1.set_xlabel("Time (seconds)")

ax1.set_ylabel("Amplitude")

ax1.grid(True)

st.pyplot(fig1)

plt.close(fig1)


# ------------------------------------------------------------
# Enhanced waveform
# ------------------------------------------------------------

st.write("🟢 Enhanced Speech Waveform")

fig2, ax2 = plt.subplots(
    figsize=(12, 3)
)

time_enhanced = np.arange(
    len(enhanced_display)
) / enhanced_sr

ax2.plot(
    time_enhanced,
    enhanced_display
)

ax2.set_xlabel("Time (seconds)")

ax2.set_ylabel("Amplitude")

ax2.grid(True)

st.pyplot(fig2)

plt.close(fig2)


# ============================================================
# FINAL RESULT
# ============================================================

st.divider()

st.subheader("🏆 Sonic Shield Result")

st.success(
    "Speech enhancement pipeline completed successfully."
)

st.write(
    "Final enhanced audio:"
)

st.code(
    ENHANCED_FILE
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SONIC SHIELD • AI Speech Enhancement System • SIH 2026"
)