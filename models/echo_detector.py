"""
FFT-based echo detection via autocorrelation.

Model: an echoed signal is y[n] = x[n] + alpha * x[n-D], where D is the echo
delay (in samples) and alpha is the relative echo strength. Such a signal's
autocorrelation R_yy[m] shows a strong peak at lag 0 (always) and a
secondary peak at lag = D (the echo), roughly proportional to alpha.

We compute the autocorrelation efficiently via FFT (Wiener-Khinchin theorem):

    X[k]    = FFT(x[n])
    P[k]    = X[k] * conj(X[k])      (power spectrum)
    Rxx[m]  = IFFT(P[k])             (autocorrelation)

The signal is zero-padded to at least 2N before the FFT so the result is a
true LINEAR autocorrelation, not a circular one (which would wrap echoes
from the end of the signal back to the start and corrupt the estimate).
"""

import numpy as np
from scipy.signal import find_peaks

MIN_DELAY_MS_DEFAULT = 50
MAX_DELAY_MS_DEFAULT = 500
CORR_THRESHOLD_DEFAULT = 0.15
PROMINENCE_THRESHOLD_DEFAULT = 0.05
CONFIDENT_MULTIPLIER = 1.6     # strength >= threshold * this -> "high confidence"

MAX_ANALYSIS_SECONDS = 6.0     # cap a single frame's length for speed
MIN_ENERGY_RMS = 1e-4          # below this, treat audio as silent/too quiet


class EchoDetectionError(ValueError):
    """Raised for audio that cannot be analyzed (too short, silent, etc.)."""
    pass


def _to_mono(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal


def _validate_signal(signal: np.ndarray, sample_rate: int, max_delay_ms: float):
    if sample_rate is None or sample_rate <= 0:
        raise EchoDetectionError("Invalid sample rate.")
    if signal.size == 0:
        raise EchoDetectionError("Audio is empty.")

    max_delay_samples = int(max_delay_ms / 1000 * sample_rate)
    min_required = int(max_delay_samples * 1.5)
    if len(signal) < min_required:
        raise EchoDetectionError(
            "Audio is too short for reliable echo detection. "
            f"Need at least {min_required / sample_rate:.2f} s of audio to "
            f"search delays up to {max_delay_ms:.0f} ms."
        )

    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))
    if rms < MIN_ENERGY_RMS:
        raise EchoDetectionError(
            "The audio contains insufficient signal energy for reliable "
            "echo analysis (it may be silent or extremely quiet)."
        )


def compute_autocorrelation_fft(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Linear autocorrelation via FFT, normalized so Rxx[0] == 1.
    Returns Rxx[0 .. max_lag].
    """
    signal = signal - np.mean(signal)  # remove DC offset

    n = len(signal)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2

    spectrum = np.fft.fft(signal, n=nfft)
    power = spectrum * np.conj(spectrum)          # |X[k]|^2
    r_full = np.fft.ifft(power).real

    zero_lag = r_full[0]
    if zero_lag <= 0:
        raise EchoDetectionError(
            "The audio contains insufficient signal energy for reliable echo analysis."
        )

    return r_full[:max_lag + 1] / zero_lag


def _find_best_peak(autocorr, min_lag, max_lag, corr_threshold, prominence_threshold):
    max_lag = min(max_lag, len(autocorr) - 1)
    if min_lag > max_lag:
        return None

    segment = autocorr[min_lag:max_lag + 1]
    peaks, props = find_peaks(segment, height=corr_threshold, prominence=prominence_threshold)

    if len(peaks) == 0:
        return None

    best_local = peaks[np.argmax(props["peak_heights"])]
    best_idx_in_all = np.where(peaks == best_local)[0][0]

    return {
        "lag": min_lag + int(best_local),
        "strength": float(segment[best_local]),
        "prominence": float(props["prominences"][best_idx_in_all]),
    }


def detect_echo_single(signal, sample_rate,
                        min_delay_ms=MIN_DELAY_MS_DEFAULT,
                        max_delay_ms=MAX_DELAY_MS_DEFAULT,
                        corr_threshold=CORR_THRESHOLD_DEFAULT,
                        prominence_threshold=PROMINENCE_THRESHOLD_DEFAULT,
                        _signal_already_mono=False):
    """Run FFT-based autocorrelation echo detection on one block of audio."""
    mono = signal if _signal_already_mono else _to_mono(signal)
    _validate_signal(mono, sample_rate, max_delay_ms)

    max_samples = int(MAX_ANALYSIS_SECONDS * sample_rate)
    analysis_signal = mono[:max_samples] if len(mono) > max_samples else mono

    min_lag = max(1, int(min_delay_ms / 1000 * sample_rate))
    max_lag = int(max_delay_ms / 1000 * sample_rate)
    max_lag = min(max_lag, len(analysis_signal) - 1)

    autocorr = compute_autocorrelation_fft(analysis_signal, max_lag)
    peak = _find_best_peak(autocorr, min_lag, max_lag, corr_threshold, prominence_threshold)

    result = {
        "autocorrelation": autocorr, "min_lag": min_lag, "max_lag": max_lag,
        "sample_rate": sample_rate, "corr_threshold": corr_threshold,
        "prominence_threshold": prominence_threshold,
    }

    if peak is None:
        result.update({
            "status": "no_echo", "detected": False, "delay_ms": None,
            "strength": None, "prominence": None,
            "message": "No clear echo was found in the delay range you chose.",
        })
        return result

    delay_ms = peak["lag"] / sample_rate * 1000
    confident = peak["strength"] >= corr_threshold * CONFIDENT_MULTIPLIER

    result.update({
        "status": "echo_detected" if confident else "possible_echo",
        "detected": True, "delay_ms": delay_ms, "delay_samples": peak["lag"],
        "strength": peak["strength"], "prominence": peak["prominence"],
        "message": (
            "A clear echo appears in this recording."
            if confident else
            "There may be an echo, but the match is weak."
        ),
    })
    return result


def detect_echo_multiframe(signal, sample_rate,
                            min_delay_ms=MIN_DELAY_MS_DEFAULT,
                            max_delay_ms=MAX_DELAY_MS_DEFAULT,
                            corr_threshold=CORR_THRESHOLD_DEFAULT,
                            prominence_threshold=PROMINENCE_THRESHOLD_DEFAULT,
                            frame_seconds=3.0, hop_seconds=1.5,
                            max_frames=8, tolerance_ms=15.0):
    """
    Split the audio into overlapping frames, run detect_echo_single on each,
    and check whether a similar delay shows up consistently -- stronger
    evidence of a real echo than a single frame's peak alone.
    """
    mono = _to_mono(signal)
    _validate_signal(mono, sample_rate, max_delay_ms)

    frame_len = int(frame_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)

    starts = list(range(0, max(1, len(mono) - frame_len + 1), hop_len))
    if not starts:
        starts = [0]
    starts = starts[:max_frames]

    frame_results = []
    for start in starts:
        frame = mono[start:start + frame_len]
        if len(frame) < frame_len:
            continue
        try:
            r = detect_echo_single(frame, sample_rate, min_delay_ms, max_delay_ms,
                                   corr_threshold, prominence_threshold,
                                   _signal_already_mono=True)
        except EchoDetectionError:
            continue
        frame_results.append(r)

    if not frame_results:
        single = detect_echo_single(mono, sample_rate, min_delay_ms, max_delay_ms,
                                    corr_threshold, prominence_threshold)
        single["consistency"] = None
        single["frames_analyzed"] = 1
        single["frames_with_peak"] = 1 if single["detected"] else 0
        return single

    detected_delays = [r["delay_ms"] for r in frame_results if r["detected"]]

    if not detected_delays:
        return {
            "status": "no_echo", "detected": False, "delay_ms": None,
            "strength": None, "prominence": None, "consistency": 0.0,
            "frames_analyzed": len(frame_results), "frames_with_peak": 0,
            "message": "No clear echo was found in the parts of the recording we checked.",
            "autocorrelation": frame_results[0]["autocorrelation"],
            "min_lag": frame_results[0]["min_lag"], "max_lag": frame_results[0]["max_lag"],
            "sample_rate": sample_rate, "corr_threshold": corr_threshold,
            "prominence_threshold": prominence_threshold,
        }

    median_delay = float(np.median(detected_delays))
    consistent = [d for d in detected_delays if abs(d - median_delay) <= tolerance_ms]
    consistency = len(consistent) / len(frame_results)

    matching_frames = [r for r in frame_results
                       if r["detected"] and abs(r["delay_ms"] - median_delay) <= tolerance_ms]
    avg_strength = float(np.mean([r["strength"] for r in matching_frames]))
    avg_prominence = float(np.mean([r["prominence"] for r in matching_frames]))

    confident = (avg_strength >= corr_threshold * CONFIDENT_MULTIPLIER) and (consistency >= 0.5)
    representative = matching_frames[0]

    return {
        "status": "echo_detected" if confident else "possible_echo",
        "detected": True, "delay_ms": median_delay,
        "delay_samples": int(median_delay / 1000 * sample_rate),
        "strength": avg_strength, "prominence": avg_prominence,
        "consistency": consistency, "frames_analyzed": len(frame_results),
        "frames_with_peak": len(consistent),
        "message": (
            f"An echo about {median_delay:.0f} ms later appeared in "
            f"{len(consistent)} of {len(frame_results)} sections checked."
            if confident else
            f"A possible echo about {median_delay:.0f} ms later appeared in some "
            "sections, but the match was too weak or inconsistent to be sure."
        ),
        "autocorrelation": representative["autocorrelation"],
        "min_lag": representative["min_lag"], "max_lag": representative["max_lag"],
        "sample_rate": sample_rate, "corr_threshold": corr_threshold,
        "prominence_threshold": prominence_threshold,
    }
