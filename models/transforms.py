from scipy import signal as sig

def design_analog_filter(order, low, high):
    """Design an analog (Laplace-domain) Butterworth bandpass filter."""
    b, a = sig.butter(order, [low, high], btype='bandpass', analog=True)
    zeros, poles, gain = sig.tf2zpk(b, a)
    return b, a, zeros, poles, gain

def design_digital_filter(order, low, high, sample_rate):
    """Design a digital (Z-domain) Butterworth bandpass filter."""
    b, a = sig.butter(order, [low, high], btype='bandpass', fs=sample_rate)
    zeros, poles, gain = sig.tf2zpk(b, a)
    return b, a, zeros, poles, gain

def check_stability(poles):
    """A digital filter is stable if all poles lie inside the unit circle."""
    magnitudes = [abs(p) for p in poles]
    is_stable = all(m < 1 for m in magnitudes)
    return is_stable, magnitudes