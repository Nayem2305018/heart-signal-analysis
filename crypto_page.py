import streamlit as st
import numpy as np
import soundfile as sf
import io

from models.audio_crypto import encrypt_audio_file_bytes, decrypt_audio_file_bytes


SUPPORTED_INPUT_TYPES = ["wav", "mp3", "flac", "ogg"]
CONTAINER_SAMPLE_RATE = 44100  # arbitrary -- the WAV here is just a byte container


def _wav_bytes_from_signal(signal, sample_rate=CONTAINER_SAMPLE_RATE, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV", subtype=subtype)
    buf.seek(0)
    return buf.read()


def _guess_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


def render_audio_crypto():
    st.title("Password-protect an audio file")
    st.write(
        "Choose an audio file and a password to make an encrypted WAV. "
        "To get the original file back, upload that WAV in the Decrypt tab "
        "and enter the same password. Its format and contents are restored exactly."
    )

    tab_encrypt, tab_decrypt = st.tabs(["🔒 Encrypt", "🔓 Decrypt"])

    # ---------------- ENCRYPT ----------------
    with tab_encrypt:
        st.subheader("Encrypt an audio file")
        st.caption("Accepted formats: WAV, MP3, FLAC, OGG")

        plain_file = st.file_uploader(
            "Upload audio", type=SUPPORTED_INPUT_TYPES, key="enc_upload"
        )
        enc_password = st.text_input("Password", type="password", key="enc_pw")
        enc_password_confirm = st.text_input("Confirm password", type="password", key="enc_pw2")

        if st.button("Encrypt", key="do_encrypt"):
            if plain_file is None:
                st.error("Please upload an audio file first.")
            elif not enc_password:
                st.error("Please enter a password.")
            elif enc_password != enc_password_confirm:
                st.error("Passwords do not match.")
            else:
                raw_bytes = plain_file.getvalue()

                encrypted_signal = encrypt_audio_file_bytes(
                    raw_bytes, plain_file.name, enc_password
                )

                st.success("File encrypted. Download the WAV to keep it.")

                encrypted_wav_bytes = _wav_bytes_from_signal(encrypted_signal)

                st.caption("The encrypted recording will sound like noise.")
                st.audio(encrypted_wav_bytes, format="audio/wav")

                st.download_button(
                    "⬇️ Download Encrypted File (.wav)",
                    data=encrypted_wav_bytes,
                    file_name=f"{plain_file.name.rsplit('.', 1)[0]}_encrypted.wav",
                    mime="audio/wav",
                )

    # ---------------- DECRYPT ----------------
    with tab_decrypt:
        st.subheader("Get your original file back")
        st.caption(
            "Use the WAV downloaded from the Encrypt tab. Keep it as a WAV; "
            "converting it to MP3 or OGG will prevent decryption."
        )

        enc_file = st.file_uploader(
            "Upload encrypted file", type=["wav"], key="dec_upload"
        )
        dec_password = st.text_input("Password", type="password", key="dec_pw")

        if st.button("Decrypt", key="do_decrypt"):
            if enc_file is None:
                st.error("Please upload an encrypted .wav file first.")
            elif not dec_password:
                st.error("Please enter the password.")
            else:
                try:
                    encrypted_signal, _ = sf.read(
                        io.BytesIO(enc_file.getvalue()), dtype="int16"
                    )
                    if encrypted_signal.ndim > 1:
                        encrypted_signal = encrypted_signal[:, 0]
                except Exception as e:
                    st.error(f"Could not read this file as encrypted data: {e}")
                    st.stop()

                try:
                    original_bytes, original_filename = decrypt_audio_file_bytes(
                        encrypted_signal, dec_password
                    )
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.stop()

                st.success(f"Decrypted. Your original file, '{original_filename}', is ready to download.")

                mime = _guess_mime(original_filename)
                st.audio(original_bytes, format=mime)

                st.download_button(
                    f"⬇️ Download Decrypted File ({original_filename})",
                    data=original_bytes,
                    file_name=original_filename,
                    mime=mime,
                )
