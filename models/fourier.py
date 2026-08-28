import numpy as np

def compute_fft(signal, sample_rate):
    """Compute FFT and return positive-frequency magnitude + phase spectrum."""
    n = len(signal)
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)

    positive_freqs = freqs[:n // 2]
    magnitude = np.abs(fft_values[:n // 2])
    phase = np.angle(fft_values[:n // 2])

    return positive_freqs, magnitude, phase

def fourier_series_reconstruct(segment, num_harmonics):
    """Reconstruct a periodic segment using a limited number of Fourier harmonics."""
    N = len(segment)
    coeffs = np.fft.fft(segment) / N

    recon = np.zeros(N, dtype=complex)
    for k in range(-num_harmonics, num_harmonics + 1):
        recon += coeffs[k % N] * np.exp(2j * np.pi * k * np.arange(N) / N)

    return recon.real

def generate_cft_illustration(freq_demo=50, duration=0.05, sample_rate=2000):
    """Generate a continuous vs sampled signal pair to illustrate the CFT -> DTFT concept."""
    t_continuous = np.linspace(0, duration, 5000)
    continuous_signal = np.sin(2 * np.pi * freq_demo * t_continuous)

    t_sampled = np.arange(0, duration, 1/sample_rate)
    sampled_signal = np.sin(2 * np.pi * freq_demo * t_sampled)

    return t_continuous, continuous_signal, t_sampled, sampled_signal