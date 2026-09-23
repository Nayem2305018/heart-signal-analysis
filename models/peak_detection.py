import numpy as np
from scipy.signal import find_peaks

def get_envelope(filtered_signal, sample_rate, lowpass_cutoff=10):
    """Extract amplitude envelope via rectification + low-pass filtering (uses filtering topic again)."""
    from scipy import signal as sig
    rectified = np.abs(filtered_signal)
    b, a = sig.butter(2, lowpass_cutoff, btype='low', fs=sample_rate)
    envelope = sig.filtfilt(b, a, rectified)
    return envelope

def detect_peaks(envelope, sample_rate, min_distance_sec=0.18, prominence=None):
    """Find prominent sound peaks using the recording's own envelope scale."""
    envelope = np.asarray(envelope, dtype=float)
    distance_samples = max(1, int(min_distance_sec * sample_rate))
    candidates, _ = find_peaks(envelope, distance=distance_samples)
    if len(candidates) == 0:
        return candidates, {}
    reference = float(np.mean(np.sort(envelope[candidates])[-5:]))
    if not np.isfinite(reference) or reference <= 0:
        return np.array([], dtype=int), {}
    minimum_prominence = 0.08 * reference if prominence is None else prominence
    peaks, properties = find_peaks(envelope, distance=distance_samples,
                                   height=0.25 * reference,
                                   prominence=minimum_prominence)
    return peaks, properties

def classify_s1_s2(peaks, sample_rate):
    """Label the longest plausible alternating S1/S2 run; leave other peaks unknown."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    peak_times = np.asarray(peaks, dtype=float) / sample_rate
    unknown = ["Unclassified"] * len(peak_times)
    if len(peak_times) < 5:
        return peak_times, unknown

    intervals = np.diff(peak_times)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        return peak_times, unknown

    best_start, best_cycles = 0, 0
    for start in range(len(intervals) - 1):
        cycles = 0
        for i in range(start, len(intervals) - 1, 2):
            short, long = intervals[i], intervals[i + 1]
            if not (0.11 <= short <= 0.55 and 0.22 <= long <= 1.5
                    and long >= 1.15 * short and short + long <= 1.8):
                break
            cycles += 1
        if cycles > best_cycles:
            best_start, best_cycles = start, cycles
    if best_cycles < 2:
        return peak_times, unknown

    labels = unknown
    for i in range(best_start, best_start + 2 * best_cycles + 1):
        labels[i] = "S1" if (i - best_start) % 2 == 0 else "S2"
    if (best_start > 0 and 0.22 <= intervals[best_start - 1] <= 1.5
            and intervals[best_start - 1] >= 1.15 * intervals[best_start]):
        labels[best_start - 1] = "S2"
    return peak_times, labels


def get_s1_intervals(peak_times, labels):
    """Return S1-to-S1 intervals from complete, locally labeled S1/S2/S1 cycles."""
    if len(peak_times) != len(labels):
        return np.array([])
    times = np.asarray(peak_times, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        return np.array([])
    return np.asarray([times[i + 2] - times[i]
                       for i in range(len(times) - 2)
                       if labels[i] == "S1" and labels[i + 1] == "S2"
                       and labels[i + 2] == "S1"])

def calculate_heart_rate(peak_times, labels):
    """Estimate typical BPM from the median S1-to-S1 interval."""
    intervals = get_s1_intervals(peak_times, labels)
    if len(intervals) < 2:
        return None
    return round(60 / float(np.median(intervals)), 1)


def calculate_interval_cv(peak_times, labels):
    """Sample SD / mean of S1 intervals; require at least three intervals."""
    intervals = get_s1_intervals(peak_times, labels)
    if len(intervals) < 3:
        return None
    return float(np.std(intervals, ddof=1) / np.mean(intervals))
