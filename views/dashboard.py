import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io

from models.loader import get_signal_properties
from models.fourier import compute_fft
from models.transforms import design_analog_filter, design_digital_filter, check_stability
from models.filters import apply_bandpass_filter
from models.peak_detection import get_envelope, detect_peaks, classify_s1_s2, calculate_heart_rate


st.set_page_config(page_title="Heart Sound Analyzer", layout="wide")
st.title("🫀 Heart Sound (PCG) Signal Analyzer")
st.write("Upload a heart sound `.wav` file to analyze its waveform, spectrum, filter it, and detect heartbeats.")


# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload a heart sound (.wav) file", type=["wav"])

if uploaded_file is not None:
    # Load audio directly from the uploaded file
    signal, sample_rate = sf.read(uploaded_file)
    if signal.ndim > 1:
        signal = signal[:, 0]  # use one channel if stereo

    # ---------- Basic properties ----------
    st.header("1. Signal Properties")
    props = get_signal_properties(signal, sample_rate)
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Rate", f"{props['sample_rate']} Hz")
    col2.metric("Duration", f"{props['duration']} s")
    col3.metric("Samples", props['num_samples'])

    # ---------- Waveform ----------
    st.subheader("Raw Waveform")
    fig1, ax1 = plt.subplots(figsize=(12, 3))
    time = np.linspace(0, props['duration'], len(signal))
    ax1.plot(time, signal, linewidth=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # ---------- Audio playback: original ----------
    st.audio(uploaded_file, format="audio/wav")

    # ---------- FFT Spectrum ----------
    st.header("2. Frequency Spectrum (FFT)")
    freqs, magnitude, phase = compute_fft(signal, sample_rate)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(freqs, magnitude, linewidth=0.7)
    ax2.set_xlim(0, 200)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    st.pyplot(fig2)

    # ---------- Filter design (interactive sliders) ----------
    st.header("3. Bandpass Filter Design (Laplace + Z-Transform)")
    col1, col2 = st.columns(2)
    low_cutoff = col1.slider("Low cutoff (Hz)", 5, 100, 20)
    high_cutoff = col2.slider("High cutoff (Hz)", 100, 300, 150)

    b_d, a_d, zeros_d, poles_d, gain_d = design_digital_filter(4, low_cutoff, high_cutoff, sample_rate)
    is_stable, magnitudes = check_stability(poles_d)

    st.write(f"**Filter stability:** {'✅ Stable' if is_stable else '❌ Unstable'}")

    # Pole-zero plot
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(poles_d.real, poles_d.imag, marker='x', s=100, color='red', label='Poles')
    ax3.scatter(zeros_d.real, zeros_d.imag, marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.axvline(0, color='gray', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    ax3.add_patch(circle)
    ax3.set_title("Z-Plane Pole-Zero Plot")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imaginary")
    ax3.legend()
    ax3.axis('equal')
    st.pyplot(fig3)

    # ---------- Apply filter ----------
    filtered_signal = apply_bandpass_filter(signal, b_d, a_d)

    st.subheader("Before vs After Filtering")
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 5))
    ax4a.plot(time, signal, linewidth=0.5)
    ax4a.set_title("Original")
    ax4b.plot(time, filtered_signal, linewidth=0.5, color='green')
    ax4b.set_title(f"Filtered ({low_cutoff}-{high_cutoff} Hz)")
    ax4b.set_xlabel("Time (s)")
    st.pyplot(fig4)

    # ---------- Filtered audio playback ----------
    filtered_buffer = io.BytesIO()
    sf.write(filtered_buffer, filtered_signal, sample_rate, format='WAV')
    filtered_buffer.seek(0)
    st.write("**Filtered audio:**")
    st.audio(filtered_buffer, format="audio/wav")

    # ---------- Heartbeat detection ----------
    st.header("4. Heartbeat Detection (S1/S2)")
    envelope = get_envelope(filtered_signal, sample_rate)
    peaks, _ = detect_peaks(envelope, sample_rate)
    peak_times, labels = classify_s1_s2(peaks, sample_rate)
    heart_rate = calculate_heart_rate(peak_times, labels)

    col1, col2 = st.columns(2)
    col1.metric("Heart Rate", f"{heart_rate} BPM" if heart_rate else "N/A")
    col2.metric("Peaks Detected", len(peaks))

    fig5, ax5 = plt.subplots(figsize=(14, 3))
    ax5.plot(time, filtered_signal, linewidth=0.5)
    for t, label in zip(peak_times, labels):
        idx = int(t * sample_rate)
        if idx < len(filtered_signal):
            color = 'red' if label == "S1" else 'green'
            ax5.scatter(t, filtered_signal[idx], color=color, zorder=5)
            ax5.annotate(label, (t, filtered_signal[idx]), textcoords="offset points", xytext=(0, 8), ha='center')
    ax5.set_title("Annotated Heartbeat Graph")
    ax5.set_xlabel("Time (s)")
    st.pyplot(fig5)

    # ---------- Simple abnormality flag ----------
    st.header("5. Abnormality Check")
    if len(peak_times) > 2:
        intervals = np.diff(peak_times)
        irregularity = np.std(intervals) / np.mean(intervals)  # coefficient of variation
        if irregularity > 0.25:
            st.error(f"⚠️ Irregular rhythm detected (variability: {irregularity:.2f})")
        else:
            st.success(f"✅ Regular rhythm (variability: {irregularity:.2f})")
    else:
        st.warning("Not enough peaks detected to assess rhythm regularity.")

else:
    st.info("👆 Upload a .wav file to begin analysis.")