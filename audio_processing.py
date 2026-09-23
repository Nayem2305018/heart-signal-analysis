"""Signal processing helpers for the general-purpose audio analyzer."""

from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf
from scipy import signal


MAX_SECONDS = 5 * 60
REDUCED_OVERLAP_AFTER_SECONDS = 120
MIN_PLAYBACK_RATE = 16000


def _too_long_message() -> str:
    return f"Choose a recording no longer than {MAX_SECONDS // 60} minutes."


def read_audio(data: bytes) -> tuple[np.ndarray, int]:
    info = sf.info(io.BytesIO(data))
    if info.samplerate <= 0 or info.frames / info.samplerate > MAX_SECONDS:
        raise ValueError(_too_long_message())
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if sample_rate <= 0 or len(audio) < max(2, math.ceil(sample_rate * 0.1)):
        raise ValueError("Choose a recording at least 0.1 seconds long.")
    if len(audio) / sample_rate > MAX_SECONDS:
        raise ValueError(_too_long_message())
    if not np.isfinite(audio).all():
        raise ValueError("The recording contains invalid sample values.")
    return np.mean(audio, axis=1).astype(np.float32), sample_rate


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1, 1), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def playback_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Make a PCM WAV at a rate that browser audio decoders can play."""
    if sample_rate < MIN_PLAYBACK_RATE:
        divisor = math.gcd(sample_rate, MIN_PLAYBACK_RATE)
        audio = signal.resample_poly(audio, MIN_PLAYBACK_RATE // divisor,
                                     sample_rate // divisor)
        sample_rate = MIN_PLAYBACK_RATE
    return wav_bytes(audio, sample_rate)


def stft(audio: np.ndarray, sample_rate: int):
    size = min(2048, len(audio))
    size -= size % 2
    overlap = (size // 2 if len(audio) > REDUCED_OVERLAP_AFTER_SECONDS * sample_rate
               else size * 3 // 4)
    return signal.stft(audio, fs=sample_rate, nperseg=size, noverlap=overlap)


def istft(spectrum: np.ndarray, sample_rate: int, length: int) -> np.ndarray:
    size = (spectrum.shape[0] - 1) * 2
    overlap = (size // 2 if length > REDUCED_OVERLAP_AFTER_SECONDS * sample_rate
               else size * 3 // 4)
    _, restored = signal.istft(spectrum, fs=sample_rate, nperseg=size,
                               noverlap=overlap)
    return restored[:length].astype(np.float32)


def spectrogram_preview(audio: np.ndarray, sample_rate: int):
    """Sample at most 900 time windows for long-recording visualization."""
    size = min(2048, len(audio))
    count = min(900, max(2, int(len(audio) / sample_rate * 40)))
    starts = np.linspace(0, len(audio) - size, count, dtype=np.int64)
    window = np.hanning(size).astype(np.float32)
    magnitude = np.empty((size // 2 + 1, count), dtype=np.float32)
    for start in range(0, count, 128):
        positions = starts[start:start + 128, None] + np.arange(size)
        frames = audio[positions] * window
        magnitude[:, start:start + len(frames)] = np.abs(
            np.fft.rfft(frames, axis=1)).T
    times = (starts + size / 2) / sample_rate
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    return frequencies, times, magnitude


def denoise(audio: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    _, _, spectrum = stft(audio, sample_rate)
    # Estimate the floor across the recording without retaining a full-size
    # magnitude array. Process columns in blocks to bound peak memory use.
    stride = max(1, spectrum.shape[1] // 1500)
    noise_floor = np.percentile(np.abs(spectrum[:, ::stride]), 15,
                                axis=1, keepdims=True)
    for start in range(0, spectrum.shape[1], 128):
        block = spectrum[:, start:start + 128]
        magnitude = np.abs(block)
        mask = np.clip(1 - strength * noise_floor / (magnitude + 1e-10), 0.08, 1)
        block *= mask
    return istft(spectrum, sample_rate, len(audio))


def equalize(audio: np.ndarray, sample_rate: int, bass: float, mids: float,
             treble: float) -> np.ndarray:
    """Gain three complementary Butterworth bands; sliders are in dB."""
    nyquist = sample_rate / 2
    low_cut = min(250, nyquist * 0.25)
    high_cut = min(4000, nyquist * 0.8)
    low_sos = signal.butter(2, low_cut, btype="lowpass", fs=sample_rate, output="sos")
    high_sos = signal.butter(2, high_cut, btype="highpass", fs=sample_rate, output="sos")
    low = signal.sosfilt(low_sos, audio).astype(np.float32)
    high = signal.sosfilt(high_sos, audio).astype(np.float32)
    result = audio.copy()
    result -= low
    result -= high
    result *= 10 ** (mids / 20)
    result += low * 10 ** (bass / 20)
    result += high * 10 ** (treble / 20)
    return result


def frequency_filter(audio: np.ndarray, sample_rate: int, kind: str,
                     low_hz: float, high_hz: float | None = None) -> np.ndarray:
    """Apply a stable fourth-order low, high, band-pass, or band-stop filter."""
    types = {"Low pass": "lowpass", "High pass": "highpass",
             "Band pass": "bandpass", "Band stop": "bandstop"}
    if kind not in types:
        raise ValueError("Choose a valid frequency filter type.")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if len(audio) < 2:
        raise ValueError("Choose a longer recording for frequency filtering.")
    nyquist = sample_rate / 2
    if not np.isfinite(low_hz) or not 0 < low_hz < nyquist:
        raise ValueError("Cutoff must be between 0 Hz and the Nyquist frequency.")
    cutoff: float | tuple[float, float] = low_hz
    if kind in ("Band pass", "Band stop"):
        if high_hz is None or not np.isfinite(high_hz) or not low_hz < high_hz < nyquist:
            raise ValueError("The upper cutoff must be above the lower cutoff and below Nyquist.")
        cutoff = (low_hz, high_hz)
    sos = signal.butter(4, cutoff, btype=types[kind], fs=sample_rate, output="sos")
    # Zero-phase filtering keeps events aligned with the original waveform.
    padlen = min(len(audio) - 2, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, audio, padlen=padlen).astype(np.float32)


def room_simulate(audio: np.ndarray, sample_rate: int, room: str,
                  wet: float) -> tuple[np.ndarray, np.ndarray]:
    settings = {"Small room": (0.35, 0.045), "Hall": (1.8, 0.10),
                "Tunnel": (2.3, 0.19)}
    duration, spacing = settings[room]
    impulse = np.zeros(max(2, int(duration * sample_rate)), dtype=np.float32)
    impulse[0] = 1
    for echo_time in np.arange(spacing, duration, spacing):
        index = int(echo_time * sample_rate)
        impulse[index] += np.exp(-3 * echo_time / duration) / 2
    reverberant = signal.fftconvolve(audio, impulse).astype(np.float32)
    dry = np.pad(audio, (0, len(reverberant) - len(audio)))
    return ((1 - wet) * dry + wet * reverberant).astype(np.float32), impulse


def effects(audio: np.ndarray, sample_rate: int, kind: str, amount: float,
            delay_seconds: float) -> np.ndarray:
    if kind in ("Echo", "Delay"):
        offset = max(1, int(delay_seconds * sample_rate))
        repeats = 4 if kind == "Echo" else 1
        output = np.pad(audio.astype(np.float32), (0, offset * repeats))
        for repeat in range(1, repeats + 1):
            start = offset * repeat
            output[start:start + len(audio)] += audio * amount ** repeat
        return output
    if kind == "Tremolo":
        time = np.arange(len(audio), dtype=np.float32) / sample_rate
        return (audio * (1 - amount + amount * (1 + np.sin(2 * np.pi *
                max(0.2, delay_seconds * 10) * time)) / 2)).astype(np.float32)
    drive = 1 + 24 * amount
    return (np.tanh(drive * audio) / np.tanh(drive)).astype(np.float32)


def paint_spectrogram(audio: np.ndarray, sample_rate: int,
                      start: float, end: float, low: float, high: float,
                      reduction: float) -> np.ndarray:
    frequencies, times, spectrum = stft(audio, sample_rate)
    freq_start = np.searchsorted(frequencies, low, side="left")
    freq_end = np.searchsorted(frequencies, high, side="right")
    time_start = np.searchsorted(times, start, side="left")
    time_end = np.searchsorted(times, end, side="right")
    spectrum[freq_start:freq_end, time_start:time_end] *= 1 - reduction
    return istft(spectrum, sample_rate, len(audio))


def loudness(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, int(sample_rate * 0.05))
    chunks = [audio[i:i + width] for i in range(0, len(audio), width)]
    rms = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in chunks])
    return np.arange(len(rms)) * width / sample_rate, 20 * np.log10(rms + 1e-8)


def spectrum_db(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frequencies, power = signal.welch(audio, fs=sample_rate,
                                       nperseg=min(4096, len(audio)))
    return frequencies, 10 * np.log10(power + 1e-12)


def estimate_alignment(first: np.ndarray, second: np.ndarray, sample_rate: int,
                       max_shift_seconds: float = 10.0) -> tuple[float, float] | None:
    """Estimate how far to shift the second clip using 100 Hz loudness envelopes."""
    if (not np.isfinite(sample_rate) or sample_rate <= 0
            or not np.isfinite(max_shift_seconds) or max_shift_seconds < 0):
        raise ValueError("Sample rate and shift range must be valid.")
    hop = max(1, round(sample_rate / 100))

    def envelope(audio: np.ndarray) -> np.ndarray:
        blocks = audio[:len(audio) // hop * hop].reshape(-1, hop)
        return np.mean(np.abs(blocks), axis=1, dtype=np.float64)

    a, b = envelope(first), envelope(second)
    if min(len(a), len(b)) < 100 or np.std(a) < 1e-5 or np.std(b) < 1e-5:
        return None
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    permitted = np.abs(lags) <= round(max_shift_seconds * sample_rate / hop)
    lags = lags[permitted]
    starts = np.maximum(0, lags)
    ends = np.minimum(len(a), len(b) + lags)
    counts = ends - starts
    valid = counts >= max(100, min(len(a), len(b)) // 2)
    if not np.any(valid):
        return None
    lags, starts, ends, counts = (item[valid] for item in (lags, starts, ends, counts))
    second_starts, second_ends = starts - lags, ends - lags

    def sums(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = np.r_[0.0, np.cumsum(values)]
        squares = np.r_[0.0, np.cumsum(values * values)]
        return total, squares

    a_sum, a_sq = sums(a)
    b_sum, b_sq = sums(b)
    sum_a = a_sum[ends] - a_sum[starts]
    sum_b = b_sum[second_ends] - b_sum[second_starts]
    variance_a = np.maximum(0, a_sq[ends] - a_sq[starts] - sum_a ** 2 / counts)
    variance_b = np.maximum(0, b_sq[second_ends] - b_sq[second_starts] - sum_b ** 2 / counts)
    cross = signal.correlate(a, b, mode="full", method="fft")[permitted][valid]
    denominator = np.sqrt(variance_a * variance_b)
    scores = np.full(len(lags), -np.inf)
    usable = denominator > 1e-10
    scores[usable] = ((cross[usable] - sum_a[usable] * sum_b[usable] / counts[usable])
                      / denominator[usable])
    best = int(np.argmax(scores))
    if not np.isfinite(scores[best]) or scores[best] < 0.35:
        return None
    return float(lags[best] * hop / sample_rate), float(np.clip(scores[best], -1, 1))


def comparison_overlap(first: np.ndarray, second: np.ndarray,
                       shift_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return matching samples after shifting the second clip on the first clip's timeline."""
    start = max(0, shift_samples)
    end = min(len(first), len(second) + shift_samples)
    if end <= start:
        return first[:0], second[:0], start
    return first[start:end], second[start - shift_samples:end - shift_samples], start


def comparison_stats(first: np.ndarray, second: np.ndarray
                     ) -> tuple[float, float | None, float | None]:
    """Return RMS difference, waveform correlation, and second-to-first level in dB."""
    if len(first) == 0 or len(first) != len(second):
        raise ValueError("Comparison requires equal, nonempty overlapping segments.")
    sums = np.zeros(6, dtype=np.float64)
    for start in range(0, len(first), 65536):
        a = first[start:start + 65536].astype(np.float64)
        b = second[start:start + 65536].astype(np.float64)
        sums += (a.sum(), b.sum(), np.dot(a, a), np.dot(b, b),
                 np.dot(a, b), np.dot(a - b, a - b))
    count = len(first)
    mean_a, mean_b = sums[0] / count, sums[1] / count
    variance_a = max(0.0, sums[2] / count - mean_a ** 2)
    variance_b = max(0.0, sums[3] / count - mean_b ** 2)
    correlation = ((sums[4] / count - mean_a * mean_b) / np.sqrt(variance_a * variance_b)
                   if variance_a > 1e-12 and variance_b > 1e-12 else None)
    if correlation is not None:
        correlation = float(np.clip(correlation, -1, 1))
    rms_a, rms_b = np.sqrt(sums[2] / count), np.sqrt(sums[3] / count)
    level_difference = (20 * np.log10(rms_b / rms_a)
                        if rms_a > 1e-8 and rms_b > 1e-8 else None)
    if level_difference is not None:
        level_difference = float(level_difference)
    return float(np.sqrt(sums[5] / count)), correlation, level_difference


def detect_beats(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate > 11025:
        divisor = math.gcd(sample_rate, 11025)
        audio = signal.resample_poly(audio, 11025 // divisor,
                                     sample_rate // divisor).astype(np.float32)
        sample_rate = 11025
    frequencies, times, spectrum = stft(audio, sample_rate)
    del frequencies
    magnitude = np.abs(spectrum)
    flux = np.maximum(0, np.diff(magnitude, axis=1)).sum(axis=0)
    if not len(flux) or np.max(flux) < 1e-7:
        return np.array([])
    distance = max(1, int(0.25 / (times[1] - times[0])))
    peaks, _ = signal.find_peaks(flux, height=np.median(flux) + np.std(flux),
                                 prominence=np.std(flux) * 0.5, distance=distance)
    return times[1:][peaks]


def detect_pitch(audio: np.ndarray, sample_rate: int,
                 at_seconds: float) -> tuple[float, str, float] | None:
    """Autocorrelation tuner for one 100 ms window around the chosen time."""
    width = min(len(audio), max(256, int(0.1 * sample_rate)))
    start = int(np.clip(at_seconds * sample_rate - width // 2, 0, len(audio) - width))
    frame = audio[start:start + width].astype(np.float64)
    frame -= np.mean(frame)
    if np.sqrt(np.mean(frame ** 2)) < 0.003:
        return None
    frame *= np.hanning(len(frame))
    correlation = signal.fftconvolve(frame, frame[::-1], mode="full")[width - 1:]
    minimum = max(1, int(sample_rate / 1200))
    maximum = min(len(correlation) - 1, int(sample_rate / 50))
    if minimum >= maximum:
        return None
    peaks, _ = signal.find_peaks(correlation[minimum:maximum])
    if not len(peaks):
        return None
    lag = int(peaks[np.argmax(correlation[minimum + peaks])] + minimum)
    if correlation[lag] < 0.15 * correlation[0]:
        return None
    frequency = sample_rate / lag
    midi = 69 + 12 * np.log2(frequency / 440)
    nearest = int(round(midi))
    note = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][nearest % 12]
    return frequency, f"{note}{nearest // 12 - 1}", (midi - nearest) * 100
