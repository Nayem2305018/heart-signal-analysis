import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import io

from models.steganography import hide_message, extract_message, get_capacity_info


def _wav_bytes(signal, sample_rate, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV", subtype=subtype)
    buf.seek(0)
    return buf.read()


def _plot_waveform_compare(original, stego, sample_rate, title_a="Original", title_b="Stego"):
    a = original if original.ndim == 1 else original[:, 0]
    b = stego if stego.ndim == 1 else stego[:, 0]
    t = np.arange(len(a)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, constrained_layout=True)
    ax1.plot(t, a, linewidth=0.5)
    ax1.set_title(title_a)
    ax1.set_ylabel("Amplitude")
    ax2.plot(t, b, linewidth=0.5, color="#53ddcb")
    ax2.set_title(title_b)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    return fig


def _plot_spectrum_compare(original, stego, sample_rate, title_a="Original Spectrum", title_b="Stego Spectrum"):
    a = original if original.ndim == 1 else original[:, 0]
    b = stego if stego.ndim == 1 else stego[:, 0]

    fa = np.abs(np.fft.rfft(a))
    fb = np.abs(np.fft.rfft(b))
    freqs = np.fft.rfftfreq(len(a), d=1 / sample_rate)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, constrained_layout=True)
    ax1.plot(freqs, fa, linewidth=0.6)
    ax1.set_title(title_a)
    ax1.set_ylabel("Magnitude")
    ax1.set_xlim(0, min(5000, sample_rate / 2))
    ax2.plot(freqs, fb, linewidth=0.6, color="#53ddcb")
    ax2.set_title(title_b)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    return fig


def render_steganography():
    st.title("🕵️ Audio Steganography (FFT-Based)")
    st.write(
        "Hide a secret text message inside a WAV file's **frequency-domain "
        "representation** using FFT, or extract a previously hidden message "
        "from a stego audio file. This is an educational demonstration of "
        "frequency-domain data embedding -- not cryptographically secure "
        "communication."
    )

    tab_hide, tab_extract = st.tabs(["🔐 Hide Message", "🔓 Extract Message"])

    # ==================== HIDE ====================
    with tab_hide:
        st.subheader("Hide a secret message in a WAV file")

        cover_file = st.file_uploader("Upload cover audio (.wav)", type=["wav"], key="stego_cover")
        secret_message = st.text_area(
            "Secret message", placeholder="Type your secret message here...", key="stego_msg"
        )

        if cover_file is not None:
            try:
                cover_signal, cover_sr = sf.read(cover_file, dtype="float32")
            except Exception as e:
                st.error(f"Could not read this WAV file: {e}")
                st.stop()

            info = get_capacity_info(cover_signal)
            st.caption(
                f"This file can hold up to **{info['capacity_chars']} characters** "
                f"({info['capacity_bits']} raw bits of capacity)."
            )

        if st.button("🔐 Hide Message", key="do_hide"):
            if cover_file is None:
                st.error("⚠ Please upload a WAV file first.")
            elif not secret_message:
                st.error("⚠ Please enter a secret message.")
            else:
                try:
                    stego_signal = hide_message(cover_signal, secret_message)
                except ValueError as e:
                    st.error(f"⚠ {e}")
                    st.stop()

                st.success("✓ Message successfully hidden in the audio.")

                stego_wav_bytes = _wav_bytes(stego_signal, cover_sr)

                st.write("**Stego audio:**")
                st.audio(stego_wav_bytes, format="audio/wav")

                base_name = cover_file.name.rsplit(".", 1)[0]
                st.download_button(
                    "⬇️ Download Stego Audio",
                    data=stego_wav_bytes,
                    file_name=f"{base_name}_stego.wav",
                    mime="audio/wav",
                )

                with st.expander("📊 Show visualizations (waveform & spectrum comparison)"):
                    st.pyplot(_plot_waveform_compare(cover_signal, stego_signal, cover_sr))
                    st.pyplot(_plot_spectrum_compare(cover_signal, stego_signal, cover_sr))
                    max_diff = float(np.max(np.abs(
                        (stego_signal if stego_signal.ndim == 1 else stego_signal[:, 0])
                        - (cover_signal if cover_signal.ndim == 1 else cover_signal[:, 0])
                    )))
                    st.caption(f"Maximum sample-level change introduced: {max_diff:.5f} "
                               f"(out of a -1.0 to 1.0 range) -- a small, mostly inaudible change.")

    # ==================== EXTRACT ====================
    with tab_extract:
        st.subheader("Extract a hidden message from a stego WAV file")

        stego_file = st.file_uploader("Upload stego audio (.wav)", type=["wav"], key="stego_extract_upload")

        if st.button("🔓 Extract Message", key="do_extract"):
            if stego_file is None:
                st.error("⚠ Please upload a WAV file first.")
            else:
                try:
                    stego_signal, stego_sr = sf.read(stego_file, dtype="float32")
                except Exception as e:
                    st.error(f"⚠ Unsupported audio format: {e}")
                    st.stop()

                try:
                    recovered_message = extract_message(stego_signal)
                except ValueError as e:
                    st.error(f"⚠ {e}")
                    st.stop()

                st.success("✓ Secret message successfully extracted.")
                st.text_area("Recovered Secret Message", value=recovered_message, height=100, disabled=True)