from scipy import signal as sig

def apply_bandpass_filter(signal_data, b, a):
    """Apply a designed filter to the signal using zero-phase filtering."""
    filtered = sig.filtfilt(b, a, signal_data)
    return filtered