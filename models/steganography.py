"""
FFT-based audio steganography: hide a UTF-8 text message inside the
frequency-domain representation of a WAV audio signal, and recover it later.

Pipeline (embedding):
    x[n] -> split into blocks -> FFT per block -> X[k]
    -> quantize the magnitude of selected mid-frequency bins to encode bits
       (each bit is repeated 3x with majority-vote decoding, so a single
       flipped bit -- e.g. from 16-bit PCM quantization noise -- cannot
       corrupt the recovered message)
    -> IFFT per block -> x'[n] (stego signal)

Pipeline (extraction):
    x'[n] -> split into blocks -> FFT per block -> X'[k]
    -> read back the quantization parity of the same selected bins,
       majority-vote every group of 3 -> bits
    -> bits -> bytes -> UTF-8 text
"""

import struct
import numpy as np

# ---------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------
BLOCK_SIZE = 4096       # samples per FFT block
LOW_SKIP = 20           # never touch the lowest bins (main audible structure)
HIGH_SKIP = 20          # never touch the highest bins (near Nyquist)
EMBED_STRENGTH = 0.25   # quantization step, as a fraction of the block's own
                        # low-frequency reference magnitude. Large enough to
                        # comfortably survive 16-bit PCM quantization noise.
MIN_DELTA = 0.01        # floor for the quantization step

REPEAT = 3              # each payload bit is embedded 3x; majority vote on
                        # extraction protects against an occasional flipped
                        # bit (e.g. from PCM quantization when saving/reloading)

MAGIC_BYTES = b"STG1"   # 4-byte signature identifying a valid stego file
MAGIC_BITS = len(MAGIC_BYTES) * 8
LENGTH_BITS = 32        # message length stored as a 4-byte big-endian uint
HEADER_BITS = MAGIC_BITS + LENGTH_BITS


# ---------------------------------------------------------------------
# Bit <-> byte helpers
# ---------------------------------------------------------------------
def _bytes_to_bits(data: bytes):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def _bits_to_int(bits):
    value = 0
    for b in bits:
        value = (value << 1) | b
    return value


def _build_payload_bits(message: str):
    message_bytes = message.encode("utf-8")
    header = MAGIC_BYTES + struct.pack(">I", len(message_bytes))
    raw_bits = _bytes_to_bits(header + message_bytes)
    encoded = []
    for bit in raw_bits:
        encoded.extend([bit] * REPEAT)
    return encoded


# ---------------------------------------------------------------------
# Magnitude quantization (QIM-style) -- one bit per frequency coefficient
# ---------------------------------------------------------------------
def _quantize_magnitude(magnitude: float, bit: int, delta: float) -> float:
    k = int(np.round(magnitude / delta))
    if bit == 1 and k % 2 == 0:
        k += 1
    elif bit == 0 and k % 2 == 1:
        k += 1
    return k * delta


def _read_bit_from_magnitude(magnitude: float, delta: float) -> int:
    k = int(np.round(magnitude / delta))
    return k % 2


def _usable_bin_indices(block_size: int):
    n_bins = block_size // 2 + 1
    start, end = LOW_SKIP, n_bins - HIGH_SKIP
    return list(range(start, end)) if end > start else []


def calculate_capacity_bits(num_samples: int, block_size: int = BLOCK_SIZE) -> int:
    """Usable RAW (post repetition-decode) message-bit capacity."""
    num_blocks = int(np.ceil(num_samples / block_size))
    usable = len(_usable_bin_indices(block_size))
    total_code_bits = num_blocks * usable
    return total_code_bits // REPEAT


def calculate_capacity_chars(num_samples: int, block_size: int = BLOCK_SIZE) -> int:
    capacity_bits = calculate_capacity_bits(num_samples, block_size)
    return max(0, (capacity_bits - HEADER_BITS) // 8)


# ---------------------------------------------------------------------
# Core: hide / extract on a single-channel float32 array
# ---------------------------------------------------------------------
def _hide_in_channel(channel: np.ndarray, encoded_bits, block_size: int = BLOCK_SIZE):
    original_length = len(channel)
    padded_length = int(np.ceil(original_length / block_size) * block_size)
    padded = np.zeros(padded_length, dtype=np.float64)
    padded[:original_length] = channel

    usable_bins = _usable_bin_indices(block_size)
    num_blocks = padded_length // block_size
    capacity_code_bits = num_blocks * len(usable_bins)
    if len(encoded_bits) > capacity_code_bits:
        raw_capacity = capacity_code_bits // REPEAT
        raw_needed = len(encoded_bits) // REPEAT
        raise ValueError(
            f"Message is too large for this audio file. "
            f"Capacity: {raw_capacity} bits, needed: {raw_needed} bits. "
            f"Please use a shorter message or a longer audio file."
        )

    stego = np.zeros_like(padded)
    bit_idx = 0
    for b in range(num_blocks):
        block = padded[b * block_size:(b + 1) * block_size]
        spectrum = np.fft.rfft(block)

        if bit_idx < len(encoded_bits):
            ref_bins = spectrum[:LOW_SKIP]
            ref_magnitude = float(np.mean(np.abs(ref_bins))) if LOW_SKIP > 0 else 1e-3
            delta = max(MIN_DELTA, ref_magnitude * EMBED_STRENGTH)

            for k in usable_bins:
                if bit_idx >= len(encoded_bits):
                    break
                mag = np.abs(spectrum[k])
                phase = np.angle(spectrum[k])
                new_mag = _quantize_magnitude(mag, encoded_bits[bit_idx], delta)
                spectrum[k] = new_mag * np.exp(1j * phase)
                bit_idx += 1

        stego[b * block_size:(b + 1) * block_size] = np.fft.irfft(spectrum, n=block_size)

    return stego[:original_length].astype(np.float32)


def _extract_from_channel(channel: np.ndarray, block_size: int = BLOCK_SIZE) -> str:
    original_length = len(channel)
    padded_length = int(np.ceil(original_length / block_size) * block_size)
    padded = np.zeros(padded_length, dtype=np.float64)
    padded[:original_length] = channel

    usable_bins = _usable_bin_indices(block_size)
    num_blocks = padded_length // block_size

    raw_bits = []
    decoded_bits = []
    message_total_raw_bits = None

    for b in range(num_blocks):
        block = padded[b * block_size:(b + 1) * block_size]
        spectrum = np.fft.rfft(block)
        ref_bins = spectrum[:LOW_SKIP]
        ref_magnitude = float(np.mean(np.abs(ref_bins))) if LOW_SKIP > 0 else 1e-3
        delta = max(MIN_DELTA, ref_magnitude * EMBED_STRENGTH)

        for k in usable_bins:
            mag = np.abs(spectrum[k])
            raw_bits.append(_read_bit_from_magnitude(mag, delta))

            if len(raw_bits) % REPEAT == 0:
                group = raw_bits[-REPEAT:]
                decoded_bits.append(1 if sum(group) * 2 > REPEAT else 0)

                if message_total_raw_bits is None and len(decoded_bits) >= HEADER_BITS:
                    magic_bits = decoded_bits[:MAGIC_BITS]
                    if _bits_to_bytes(magic_bits) != MAGIC_BYTES:
                        raise ValueError("No hidden message was detected in this audio file.")
                    length_bits = decoded_bits[MAGIC_BITS:HEADER_BITS]
                    msg_len_bytes = _bits_to_int(length_bits)
                    message_total_raw_bits = HEADER_BITS + msg_len_bytes * 8

                if (message_total_raw_bits is not None
                        and len(decoded_bits) >= message_total_raw_bits):
                    message_bits = decoded_bits[HEADER_BITS:message_total_raw_bits]
                    return _bits_to_bytes(message_bits).decode("utf-8", errors="replace")

    raise ValueError("No hidden message was detected in this audio file.")


# ---------------------------------------------------------------------
# Public API: works on mono (1-D) or multi-channel (2-D: samples x channels)
# ---------------------------------------------------------------------
def hide_message(signal: np.ndarray, message: str, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Embed `message` into `signal` (channel 0 only, if multi-channel)."""
    encoded_bits = _build_payload_bits(message)
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        stego_channel = _hide_in_channel(signal, encoded_bits, block_size)
        return np.clip(stego_channel, -1.0, 1.0).astype(np.float32)

    result = signal.copy()
    stego_channel = _hide_in_channel(signal[:, 0], encoded_bits, block_size)
    result[:, 0] = np.clip(stego_channel, -1.0, 1.0)
    return result.astype(np.float32)


def extract_message(signal: np.ndarray, block_size: int = BLOCK_SIZE) -> str:
    """Recover a message previously hidden with hide_message()."""
    signal = np.asarray(signal, dtype=np.float32)
    channel = signal if signal.ndim == 1 else signal[:, 0]
    return _extract_from_channel(channel, block_size)


def get_capacity_info(signal: np.ndarray, block_size: int = BLOCK_SIZE):
    signal = np.asarray(signal)
    num_samples = signal.shape[0]
    return {
        "capacity_bits": calculate_capacity_bits(num_samples, block_size),
        "capacity_chars": calculate_capacity_chars(num_samples, block_size),
    }