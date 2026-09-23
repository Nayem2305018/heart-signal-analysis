"""
AES-GCM authenticated encryption applied to the raw AUDIO SIGNAL (samples),
not the file bytes. The encrypted output is itself packed back into a .wav
container, so it stays a playable (noise-sounding) audio file rather than
an opaque binary blob.

Structure of the encrypted WAV's raw PCM data (int16 samples, reinterpreted
as bytes):

    [ 4 bytes  ] blob_length        (uint32, unencrypted -- just a length marker)
    [ 16 bytes ] salt               (unencrypted, needed to derive the key)
    [ 12 bytes ] nonce              (unencrypted, required by AES-GCM)
    [ N bytes  ] ciphertext + tag   (the actual encrypted signal + metadata)

Inside the ciphertext (only visible after successful decryption):
    [ 4 bytes ] original sample_rate   (uint32)
    [ 4 bXytes ] number of samples      (uint32)
    [ 1 byte  ] dtype code (0 = float32)
    [ ...     ] the original signal samples, as float32 bytes
"""

import os
import struct
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

SALT_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32
PBKDF2_ITERATIONS = 390_000

DTYPE_FLOAT32 = 0


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8"))


# ---------------------------------------------------------------------
# NEW: signal-level encryption (encrypts the samples, not the file)
# ---------------------------------------------------------------------

def encrypt_audio_signal(signal: np.ndarray, sample_rate: int, password: str):
    """
    Encrypt a raw audio signal (numpy array) with AES-256-GCM.

    Returns:
        encrypted_signal (np.ndarray, int16) -- noise-like samples, ready
            to be written straight to a .wav file with soundfile.
        container_sample_rate (int) -- sample rate to use when saving the
            encrypted .wav (kept equal to the original for a similar
            playback duration).
    """
    signal = np.asarray(signal, dtype=np.float32).flatten()

    # ---- Build the plaintext payload: metadata + raw samples ----
    header = struct.pack("<I I B", sample_rate, len(signal), DTYPE_FLOAT32)
    plaintext = header + signal.tobytes()

    # ---- Encrypt ----
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    blob = salt + nonce + ciphertext
    length_prefix = struct.pack("<I", len(blob))
    packed = length_prefix + blob

    # ---- Pad to an even number of bytes (int16 needs pairs of bytes) ----
    if len(packed) % 2 != 0:
        packed += b"\x00"

    # ---- Reinterpret as int16 samples so it can be saved/played as audio ----
    encrypted_signal = np.frombuffer(packed, dtype=np.int16).copy()

    return encrypted_signal, sample_rate


def decrypt_audio_signal(encrypted_signal: np.ndarray, password: str):
    """
    Decrypt samples produced by encrypt_audio_signal().

    Returns:
        original_signal (np.ndarray, float32)
        original_sample_rate (int)

    Raises ValueError if the password is wrong or the data was altered.
    """
    encrypted_signal = np.asarray(encrypted_signal, dtype=np.int16)
    packed = encrypted_signal.tobytes()

    if len(packed) < 4:
        raise ValueError("This does not look like an encrypted audio signal.")

    (blob_length,) = struct.unpack("<I", packed[:4])
    blob = packed[4:4 + blob_length]

    if len(blob) < SALT_SIZE + NONCE_SIZE:
        raise ValueError("Encrypted data is corrupted or incomplete.")

    salt = blob[:SALT_SIZE]
    nonce = blob[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = blob[SALT_SIZE + NONCE_SIZE:]

    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag:
        raise ValueError("Decryption failed: wrong password, or the audio was altered.")

    sample_rate, num_samples, dtype_code = struct.unpack("<I I B", plaintext[:9])
    signal_bytes = plaintext[9:]

    if dtype_code != DTYPE_FLOAT32:
        raise ValueError("Unknown or unsupported signal encoding.")

    original_signal = np.frombuffer(signal_bytes, dtype=np.float32)[:num_samples]

    return original_signal.copy(), sample_rate


# ---------------------------------------------------------------------
# Kept from before, unchanged: whole-file encryption (still usable if
# you ever want to encrypt an arbitrary file rather than a signal).
# ---------------------------------------------------------------------

def encrypt_audio_bytes(plaintext: bytes, password: str) -> bytes:
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return salt + nonce + ciphertext


def decrypt_audio_bytes(encrypted_data: bytes, password: str) -> bytes:
    if len(encrypted_data) < SALT_SIZE + NONCE_SIZE:
        raise ValueError("File is too short to be a valid encrypted audio file.")
    salt = encrypted_data[:SALT_SIZE]
    nonce = encrypted_data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = encrypted_data[SALT_SIZE + NONCE_SIZE:]
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag:
        raise ValueError("Decryption failed: wrong password, or the file has been altered.")
    return plaintext

def encrypt_audio_file_bytes(file_bytes: bytes, filename: str, password: str) -> np.ndarray:
    """
    Encrypt the RAW BYTES of any audio file (mp3/wav/flac/ogg/etc.) --
    no decoding happens, so the original file is byte-for-byte preserved
    and recoverable exactly on decryption.
    """
    filename_bytes = filename.encode("utf-8")[:255]
    header = struct.pack("<B", len(filename_bytes)) + filename_bytes
    plaintext = header + file_bytes

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    blob = salt + nonce + ciphertext
    packed = struct.pack("<I", len(blob)) + blob

    if len(packed) % 2 != 0:
        packed += b"\x00"

    return np.frombuffer(packed, dtype=np.int16).copy()


def decrypt_audio_file_bytes(encrypted_signal: np.ndarray, password: str):
    """
    Reverse of encrypt_audio_file_bytes(). Returns the exact original
    file bytes and its original filename.
    """
    packed = np.asarray(encrypted_signal, dtype=np.int16).tobytes()
    if len(packed) < 4:
        raise ValueError("This does not look like an encrypted audio file.")

    (blob_length,) = struct.unpack("<I", packed[:4])
    blob = packed[4:4 + blob_length]
    if len(blob) < SALT_SIZE + NONCE_SIZE:
        raise ValueError("Encrypted data is corrupted or incomplete.")

    salt = blob[:SALT_SIZE]
    nonce = blob[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = blob[SALT_SIZE + NONCE_SIZE:]

    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag:
        raise ValueError("Decryption failed: wrong password, or the file has been altered.")

    (name_len,) = struct.unpack("<B", plaintext[:1])
    filename = plaintext[1:1 + name_len].decode("utf-8")
    original_bytes = plaintext[1 + name_len:]

    return original_bytes, filename