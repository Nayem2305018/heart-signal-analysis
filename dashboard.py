import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io
import os

from models.loader import get_signal_properties
from models.fourier import compute_fft
from models.transforms import design_analog_filter, design_digital_filter, check_stability
from models.filters import apply_bandpass_filter
from models.peak_detection import (
    get_envelope, detect_peaks, classify_s1_s2, calculate_heart_rate,
    calculate_interval_cv, get_s1_intervals,
)
from views.plots import create_animated_waveform_gif
from ui_theme import apply_theme, render_empty_state, render_hero, render_sidebar_brand


HEART_ANALYSIS_SECONDS = 20


st.set_page_config(page_title="Signal Studio", page_icon="🎛️", layout="wide")
apply_theme()
render_sidebar_brand()
page = st.sidebar.radio("Workspace", ["Heart sounds", "Audio analyzer"])
if page == "Audio analyzer":
    from audio_analyzer import render_audio_analyzer

    render_audio_analyzer()
    st.stop()
render_hero("heart")


# ---------- Heart-sound source ----------
uploaded_file = st.file_uploader("Upload a heart sound (.wav) file", type=["wav"])
source_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

if source_bytes is not None:
    try:
        signal, sample_rate = sf.read(io.BytesIO(source_bytes))
    except Exception as exc:
        st.error(f"Could not read heart-sound WAV: {exc}")
        st.stop()
    if len(signal) < max(2, int(0.1 * sample_rate)):
        st.error("Upload at least 0.1 seconds of heart sounds.")
        st.stop()
    
    if signal.ndim > 1:
        signal = signal[:, 0]  # use one channel if stereo

    # Use a 20-second segment for every heart analysis; keep the GIF shorter.
    signal = signal[:HEART_ANALYSIS_SECONDS * sample_rate]
    st.caption(f"Analyzing {len(signal) / sample_rate:.2f} s of audio "
               f"(up to the first {HEART_ANALYSIS_SECONDS} s).")

    # ---------- Basic properties ----------
    st.header("1. Signal Properties")
    props = get_signal_properties(signal, sample_rate)
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Rate", f"{props['sample_rate']} Hz")
    col2.metric("Analyzed duration", f"{props['duration']} s")
    col3.metric("Samples", props['num_samples'])

    # ---------- Waveform ----------
    st.subheader("Raw Waveform")
    fig1, ax1 = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
    time = np.arange(len(signal)) / sample_rate
    ax1.plot(time, signal, linewidth=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # ---------- Animated Waveform (GIF) ----------
    st.subheader("Animated Waveform")
    if st.button("Generate Animated GIF"):
        with st.spinner("Generating animation... (this takes a few seconds)"):
            os.makedirs("outputs", exist_ok=True)
            gif_path = "outputs/animated_waveform.gif"
            
            # Slice first 5 seconds to prevent massive file sizes and long render times
            gif_samples = min(len(signal), 5 * sample_rate)
            create_animated_waveform_gif(signal[:gif_samples], sample_rate, save_path=gif_path, fps=30)
            
            st.success("Animation created!")
            st.image(gif_path, use_container_width=True)

    # ---------- Audio playback: original ----------
    st.audio(source_bytes, format="audio/wav")

    # ---------- FFT Spectrum ----------
    st.header("2. Frequency Spectrum (FFT)")
    freqs, magnitude, phase = compute_fft(signal, sample_rate)

    fig2, ax2 = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
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
    # fig3, ax3 = plt.subplots(figsize=(5, 5))
    # ax3.scatter(poles_d.real, poles_d.imag, marker='x', s=100, color='red', label='Poles')
    # ax3.scatter(zeros_d.real, zeros_d.imag, marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    # ax3.axhline(0, color='gray', linewidth=0.5)
    # ax3.axvline(0, color='gray', linewidth=0.5)
    # circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray')
    # ax3.add_patch(circle)
    # ax3.set_title("Z-Plane Pole-Zero Plot")
    # ax3.set_xlabel("Real")
    # ax3.set_ylabel("Imaginary")
    # ax3.legend()
    # ax3.axis('equal')
    # st.pyplot(fig3)

    # ---------- Apply filter ----------
    filtered_signal = apply_bandpass_filter(signal, b_d, a_d)

    st.subheader("Before vs After Filtering")
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 6),
                                     sharex=True, constrained_layout=True)
    ax4a.plot(time, signal, linewidth=0.5)
    ax4a.set_title("Original")
    ax4a.set_ylabel("Amplitude")
    ax4b.plot(time, filtered_signal, linewidth=0.5, color='#53ddcb')
    ax4b.set_title(f"Filtered ({low_cutoff}-{high_cutoff} Hz)")
    ax4b.set_xlabel("Time (s)")
    ax4b.set_ylabel("Amplitude")
    st.pyplot(fig4)

    # ---------- Filtered audio playback ----------
    filtered_buffer = io.BytesIO()
    sf.write(filtered_buffer, filtered_signal, sample_rate, format='WAV')
    filtered_buffer.seek(0)
    st.write("**Filtered audio:**")
    st.audio(filtered_buffer, format="audio/wav")

    # ---------- 4. Beat timing ----------
    st.header("4. Beat Timing Analysis")
    
    envelope = get_envelope(filtered_signal, sample_rate)
    peaks, _ = detect_peaks(envelope, sample_rate)
    peak_times, labels = classify_s1_s2(peaks, sample_rate)
    heart_rate = calculate_heart_rate(peak_times, labels)

    s1_intervals = get_s1_intervals(peak_times, labels)
    interval_cv = calculate_interval_cv(peak_times, labels)

    if heart_rate is None:
        st.info(f"Found {len(peaks)} candidate sound peaks, but not enough complete "
                "S1-S2-S1 cycles for a timing estimate. Try another recording or adjust the filter.")
    else:
        bpm_col, count_col, cv_col = st.columns(3)
        bpm_col.metric("Estimated BPM", f"{heart_rate:.1f}")
        count_col.metric("Intervals used", len(s1_intervals))
        cv_col.metric("Timing CV",
                      f"{interval_cv:.2f}" if interval_cv is not None else "Not enough")

        if interval_cv is None:
            st.warning("At least three S1-to-S1 intervals are needed to describe timing variation.")
        else:
            st.info(f"Median S1-to-S1 interval: {np.median(s1_intervals) * 1000:.0f} ms (used for BPM). "
                    f"Range: {np.min(s1_intervals) * 1000:.0f}–"
                    f"{np.max(s1_intervals) * 1000:.0f} ms. "
                    f"Interval CV: {interval_cv:.2f} (sample standard deviation divided by the mean).")
        if any(label == "Unclassified" for label in labels):
            st.caption("Only the locally consistent cycle run was used; other detected sounds are shown as candidates below.")

    st.caption("Timing analysis uses up to the first 20 seconds of the recording. "
               "These are estimates from heart-sound timing, not a diagnosis of rhythm. "
               "S1/S2 labels and the shaded spans are approximate; "
               "an ECG is used to assess suspected arrhythmia.")

    # Feature: Synchronized Heart Animation
    if heart_rate:
        pulse_duration = 60.0 / heart_rate 
        heart_html = f"""
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            15% {{ transform: scale(1.25); }} /* S1 (Lub) */
            30% {{ transform: scale(1); }}
            45% {{ transform: scale(1.15); }} /* S2 (Dub) */
            60% {{ transform: scale(1); }}
            100% {{ transform: scale(1); }}
        }}
        .heart-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 120px;
            margin: 12px 0 20px;
        }}
        .pulsing-heart {{
            display: inline-block;
            line-height: 1;
            font-size: 70px;
            animation: pulse {pulse_duration}s infinite;
            transform-origin: center;
        }}
        </style>
        <div class="heart-container">
            <div class="pulsing-heart">🫀</div>
        </div>
        """
        st.markdown(heart_html, unsafe_allow_html=True)

    # Show timing across the whole clip without overlapping per-beat annotations.
    fig5, ax5 = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    ax5.plot(time, filtered_signal, linewidth=0.6, color='#53ddcb')
    for i in range(len(peak_times) - 1):
        if labels[i] == "S1" and labels[i + 1] == "S2":
            ax5.axvspan(peak_times[i], peak_times[i + 1],
                        color='#ff8398', alpha=0.12, label='Approx. S1–S2 span')
    label_array = np.asarray(labels)
    for label, color, size in (("S1", "#ff8398", 32),
                               ("S2", "#b9a1ff", 28),
                               ("Unclassified", "#9eb9c4", 14)):
        selected = peaks[label_array == label]
        if len(selected):
            ax5.scatter(selected / sample_rate, filtered_signal[selected],
                        s=size, color=color, zorder=5,
                        label="Candidate sound" if label == "Unclassified" else label)
    ax5.set_title("Estimated S1/S2 timing" if heart_rate is not None
                  else "Candidate sound peaks (S1/S2 uncertain)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Amplitude")
    ax5.set_xlim(0, len(signal) / sample_rate)
    handles, legend_labels = ax5.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        ax5.legend(by_label.values(), by_label.keys(), loc='upper right')

    st.pyplot(fig5)
    st.caption("Colored markers show estimated sounds. Gray markers were excluded from beat timing.")
    
    # Prevent Matplotlib memory leaks on the server
    plt.close('all')

else:
    render_empty_state("Start with a heart sound recording",
                       "Upload a WAV file above to see its waveform, spectrum, filtering, and beat markers.")
