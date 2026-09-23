import numpy as np

def moving_average_kernel(window_size=5):
    return np.ones(window_size) / window_size

def apply_convolution(signal, kernel):
    return np.convolve(signal, kernel, mode='same')