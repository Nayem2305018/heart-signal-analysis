import numpy as np
from scipy.signal import find_peaks

def get_envelope(filtered_signal, sample_rate, lowpass_cutoff=10):
    """Extract amplitude envelope via rectification + low-pass filtering (uses filtering topic again)."""
    from scipy import signal as sig
    rectified = np.abs(filtered_signal)
    b, a = sig.butter(2, lowpass_cutoff, btype='low', fs=sample_rate)
    envelope = sig.filtfilt(b, a, rectified)
    return envelope

def detect_peaks(envelope, sample_rate, min_distance_sec=0.3, prominence=0.01):
    """Detect peaks in the envelope (candidate S1/S2 locations)."""
    distance_samples = int(min_distance_sec * sample_rate)
    peaks, properties = find_peaks(envelope, distance=distance_samples, prominence=prominence)
    return peaks, properties

def classify_s1_s2(peaks, sample_rate):
    """Classify each peak as S1 or S2 based on interval pattern (systole < diastole)."""
    peak_times = peaks / sample_rate
    intervals = np.diff(peak_times)

    if len(intervals) == 0:
        return peak_times, []

    median_interval = np.median(intervals)
    labels = ["S1"]  # assume first detected peak is S1

    for i in range(len(intervals)):
        if intervals[i] < median_interval:
            labels.append("S2" if labels[-1] == "S1" else "S1")
        else:
            labels.append("S1" if labels[-1] == "S2" else "S2")

    return peak_times, labels

def calculate_heart_rate(peak_times, labels):
    """Calculate heart rate (BPM) using only S1-to-S1 intervals."""
    s1_times = [t for t, l in zip(peak_times, labels) if l == "S1"]
    if len(s1_times) < 2:
        return None
    s1_intervals = np.diff(s1_times)
    heart_rate_bpm = 60 / np.mean(s1_intervals)
    return round(heart_rate_bpm, 1)