import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import io
import matplotlib.pyplot as plt

from models.echo_detector import (
    detect_echo_single, detect_echo_multiframe, EchoDetectionError,
    MIN_DELAY_MS_DEFAULT, MAX_DELAY_MS_DEFAULT, CORR_THRESHOLD_DEFAULT,
)

SUPPORTED_TYPES = ["wav", "mp3"]


def _load_any_audio(uploaded_file):
    """
    Load WAV directly via soundfile; fall back to librosa for MP3 (and
    anything soundfile can't decode). Returns float32 samples + sample rate,
    preserving channel structure (mono stays 1-D, stereo stays (n, ch)).
    """
    raw_bytes = uploaded_file.getvalue()

    try:
        signal, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception:
        signal, sample_rate = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=False)
        signal = signal.T if signal.ndim > 1 else signal
        signal = signal.astype(np.float32)

    return signal, sample_rate


def _plot_waveform(signal, sample_rate):
    mono = signal if signal.ndim == 1 else signal.mean(axis=1)
    t = np.arange(len(mono)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(t, mono, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    return fig


def _plot_autocorrelation(result):
    autocorr = result["autocorrelation"]
    sample_rate = result["sample_rate"]
    lags_ms = np.arange(len(autocorr)) / sample_rate * 1000

    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    ax.plot(lags_ms, autocorr, linewidth=0.8, color="#1f77b4")
    ax.axvspan(result["min_lag"] / sample_rate * 1000,
              result["max_lag"] / sample_rate * 1000,
              color="orange", alpha=0.08, label="Search range")
    ax.axhline(result["corr_threshold"], color="gray", linestyle="--",
              linewidth=0.8, label=f"Threshold ({result['corr_threshold']})")

    if result["detected"]:
        delay_ms = result["delay_ms"]
        idx = int(delay_ms / 1000 * sample_rate)
        if idx < len(autocorr):
            ax.scatter([delay_ms], [autocorr[idx]], color="red", zorder=5,
                      label=f"Detected echo ({delay_ms:.0f} ms)")

    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Autocorrelation (FFT-based)")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def _plot_spectrum(signal, sample_rate):
    mono = signal if signal.ndim == 1 else signal.mean(axis=1)
    n = min(len(mono), sample_rate * 6)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    magnitude = np.abs(np.fft.rfft(mono[:n]))

    fig, ax = plt.subplots(figsize=(10, 2.8), constrained_layout=True)
    ax.plot(freqs, magnitude, linewidth=0.6, color="#53ddcb")
    ax.set_xlim(0, min(5000, sample_rate / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Spectrum (supportive visualization)")
    return fig


def render_echo_detector():
    st.title("🔊 Echo Detector")
    st.write(
        "Analyzes an audio signal for a significant echo using **FFT-based "
        "autocorrelation** -- not a simple frequency check. An echo shows up "
        "as a secondary correlation peak at a non-zero time lag."
    )
    st.caption("Accepted formats: WAV, MP3")

    audio_file = st.file_uploader("Upload audio", type=SUPPORTED_TYPES, key="echo_upload")

    col1, col2, col3 = st.columns(3)
    min_delay_ms = col1.number_input("Minimum echo delay (ms)", 1, 2000, MIN_DELAY_MS_DEFAULT)
    max_delay_ms = col2.number_input("Maximum echo delay (ms)", 10, 5000, MAX_DELAY_MS_DEFAULT)
    corr_threshold = col3.number_input("Correlation threshold", 0.01, 1.0,
                                       CORR_THRESHOLD_DEFAULT, step=0.01, format="%.2f")

    use_multiframe = st.checkbox(
        "Use multi-frame analysis (more reliable, checks consistency across "
        "several overlapping segments)", value=True
    )

    if st.button("🔍 Detect Echo"):
        if audio_file is None:
            st.error("⚠ Please upload an audio file first.")
            st.stop()

        try:
            signal, sample_rate = _load_any_audio(audio_file)
        except Exception as e:
            st.error(f"⚠ Unsupported audio format or corrupted file: {e}")
            st.stop()

        try:
            if use_multiframe:
                result = detect_echo_multiframe(
                    signal, sample_rate,
                    min_delay_ms=min_delay_ms, max_delay_ms=max_delay_ms,
                    corr_threshold=corr_threshold,
                )
            else:
                result = detect_echo_single(
                    signal, sample_rate,
                    min_delay_ms=min_delay_ms, max_delay_ms=max_delay_ms,
                    corr_threshold=corr_threshold,
                )
        except EchoDetectionError as e:
            st.error(f"⚠ {e}")
            st.stop()

        st.markdown("---")
        st.subheader("Result")

        if result["status"] == "echo_detected":
            st.success("Echo Detected: **YES** ✓")
        elif result["status"] == "possible_echo":
            st.warning("Possible Echo Detected (weak evidence)")
        else:
            st.info("Echo Detected: **NO**")

        st.caption(result["message"])

        if result["detected"]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Estimated Delay", f"{result['delay_ms']:.0f} ms")
            m2.metric("Echo Strength", f"{result['strength']:.3f}")
            m3.metric("Peak Prominence", f"{result['prominence']:.3f}")
            if result.get("consistency") is not None:
                m4.metric("Consistency", f"{result['consistency']*100:.0f}%",
                          help=f"{result.get('frames_with_peak', 0)}/"
                               f"{result.get('frames_analyzed', 1)} frames agreed")
            st.caption(
                "Echo strength is an **estimated relative correlation strength**, "
                "not necessarily the exact physical reflection amplitude."
            )

        st.markdown("---")
        st.subheader("Visualizations")
        st.pyplot(_plot_waveform(signal, sample_rate))
        st.pyplot(_plot_autocorrelation(result))
        with st.expander("Show frequency spectrum (supportive view)"):
            st.pyplot(_plot_spectrum(signal, sample_rate))

        with st.expander("⚠ Known limitation"):
            st.write(
                "Strongly periodic audio (drum loops, repeated musical "
                "phrases, sustained rhythmic patterns) can sometimes produce "
                "a correlation peak that looks like an echo, since periodic "
                "signals are inherently self-similar at a delay. Multi-frame "
                "consistency checking (enabled by default) reduces this risk "
                "but cannot eliminate it entirely -- treat results on music "
                "with extra caution. MP3 compression can also slightly shift "
                "the estimated strength (though delay estimates remain accurate)."
            )