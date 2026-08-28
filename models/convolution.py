import numpy as np

def moving_average_kernel(window_size=5):
    """Create a simple moving-average convolution kernel."""
    return np.ones(window_size) / window_size

def apply_convolution(signal, kernel):
    """Apply a kernel to a signal via convolution."""
    return np.convolve(signal, kernel, mode='same')