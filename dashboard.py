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
from models.peak_detection import get_envelope, detect_peaks, classify_s1_s2, calculate_heart_rate
from views.plots import create_animated_waveform_gif
from ui_theme import apply_theme, render_empty_state, render_hero, render_sidebar_brand


st.set_page_config(page_title="Signal Studio", page_icon="🎛️", layout="wide")
apply_theme()
render_sidebar_brand()
page = st.sidebar.radio("Workspace", ["Heart sounds", "Audio analyzer"])
if page == "Audio analyzer":
    from audio_analyzer import render_audio_analyzer

    render_audio_analyzer()
    st.stop()
render_hero("heart")


# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload a heart sound (.wav) file", type=["wav"])

if uploaded_file is not None:
    # Load audio directly from the uploaded file
    signal, sample_rate = sf.read(uploaded_file)
    
    # Rewind the file pointer so st.audio can play it
    uploaded_file.seek(0)
    
    if signal.ndim > 1:
        signal = signal[:, 0]  # use one channel if stereo

    max_samples = 5 * sample_rate
    signal = signal[:max_samples]

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

    # ---------- Animated Waveform (GIF) ----------
    st.subheader("Animated Waveform")
    if st.button("Generate Animated GIF"):
        with st.spinner("Generating animation... (this takes a few seconds)"):
            os.makedirs("outputs", exist_ok=True)
            gif_path = "outputs/animated_waveform.gif"
            
            # Slice first 5 seconds to prevent massive file sizes and long render times
            max_samples = min(len(signal), 5 * sample_rate)
            create_animated_waveform_gif(signal[:max_samples], sample_rate, save_path=gif_path, fps=30)
            
            st.success("Animation created!")
            st.image(gif_path, use_container_width=True)

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
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 5))
    ax4a.plot(time, signal, linewidth=0.5)
    ax4a.set_title("Original")
    ax4b.plot(time, filtered_signal, linewidth=0.5, color='#53ddcb')
    ax4b.set_title(f"Filtered ({low_cutoff}-{high_cutoff} Hz)")
    ax4b.set_xlabel("Time (s)")
    st.pyplot(fig4)

    # ---------- Filtered audio playback ----------
    filtered_buffer = io.BytesIO()
    sf.write(filtered_buffer, filtered_signal, sample_rate, format='WAV')
    filtered_buffer.seek(0)
    st.write("**Filtered audio:**")
    st.audio(filtered_buffer, format="audio/wav")

    # ---------- 4. Heartbeat Detection & Diagnosis ----------
    st.header("4. Heartbeat Detection & Diagnosis")
    
    envelope = get_envelope(filtered_signal, sample_rate)
    peaks, _ = detect_peaks(envelope, sample_rate)
    peak_times, labels = classify_s1_s2(peaks, sample_rate)
    heart_rate = calculate_heart_rate(peak_times, labels)

    # Calculate Coefficient of Variation (CV)
    # Calculate Coefficient of Variation (CV) using ONLY S1-to-S1 intervals
    s1_times = [t for t, label in zip(peak_times, labels) if label == "S1"]
    
    if len(s1_times) > 2:
        intervals = np.diff(s1_times)
        irregularity = np.std(intervals) / np.mean(intervals)
    else:
        irregularity = 0.0

    # Feature: Plain-Language Verdict Banner
    if len(peak_times) > 2:
        if irregularity > 0.25:
            st.error(f"### ⚠️ {heart_rate} BPM — Irregular rhythm detected\n*(Variability CV = {irregularity:.2f})*")
        else:
            st.success(f"### ❤️ {heart_rate} BPM — Regular rhythm\n*(Variability CV = {irregularity:.2f})*")
    else:
        st.warning("Not enough peaks detected to assess rhythm regularity.")

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
            margin-top: -10px;
            margin-bottom: 20px;
        }}
        .pulsing-heart {{
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

    # Feature: Annotated Waveform "Story" Overlay
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    ax5.plot(time, filtered_signal, linewidth=0.6, color='#53ddcb')

    # Stable Y-limit calculations using absolute span
    y_min, y_max = ax5.get_ylim()
    y_span = y_max - y_min
    bracket_y = y_min - (y_span * 0.15) 

    for i in range(len(peak_times) - 1):
        t1 = peak_times[i]
        t2 = peak_times[i+1]
        l1 = labels[i]
        interval_ms = (t2 - t1) * 1000
        
        # Shade Systole and draw bracket
        if l1 == "S1":
            ax5.axvspan(t1, t2, color='red', alpha=0.1, label='Systole' if i==0 else "")
            ax5.annotate('', xy=(t1, bracket_y), xytext=(t2, bracket_y),
                         arrowprops=dict(arrowstyle='|-|', color='gray', lw=1.5))
            ax5.text((t1+t2)/2, bracket_y, f"{interval_ms:.0f} ms\n(Systole)", 
                     ha='center', va='top', fontsize=9, color='gray')
                     
        # Draw bracket for Diastole
        else:
            ax5.annotate('', xy=(t1, bracket_y), xytext=(t2, bracket_y),
                         arrowprops=dict(arrowstyle='|-|', color='gray', lw=1.5))
            ax5.text((t1+t2)/2, bracket_y, f"{interval_ms:.0f} ms\n(Diastole)", 
                     ha='center', va='top', fontsize=9, color='gray')

    # Draw the original S1/S2 dots
    for t, label in zip(peak_times, labels):
        idx = int(t * sample_rate)
        if idx < len(filtered_signal):
            color = '#ff8398' if label == "S1" else '#53ddcb'
            ax5.scatter(t, filtered_signal[idx], color=color, zorder=5)
            ax5.annotate(label, (t, filtered_signal[idx]), textcoords="offset points", 
                         xytext=(0, 8), ha='center', fontweight='bold')

    ax5.set_title("Annotated Heartbeat Graph (Systole vs Diastole)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Amplitude")
    
    # Apply the stable Y-limits
    ax5.set_ylim(bracket_y - (y_span * 0.15), y_max + (y_span * 0.1))
    
    # Clean up legend
    handles, legend_labels = ax5.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    if by_label:
        ax5.legend(by_label.values(), by_label.keys(), loc='upper right')

    st.pyplot(fig5)
    
    # Prevent Matplotlib memory leaks on the server
    plt.close('all')

else:
    render_empty_state("Start with a heart sound recording",
                       "Upload a WAV file above to explore its waveform, spectrum, filtering, and beat markers.")
