import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(signal, sample_rate, save_path="outputs/waveform.png"):
    duration = len(signal) / sample_rate
    time = np.linspace(0, duration, len(signal))

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, linewidth=0.7)
    plt.title("Heart Sound Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_spectrum(freqs, magnitude, phase, save_path_prefix="outputs/spectrum"):
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, magnitude, linewidth=0.8)
    plt.title("Magnitude Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 200)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_magnitude.png")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, phase, linewidth=0.8, color='orange')
    plt.title("Phase Spectrum (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.xlim(0, 200)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_phase.png")
    plt.show()

def plot_fourier_series(t, original, reconstructions, harmonics_list, save_path="outputs/fourier_series.png"):
    plt.figure(figsize=(12, 8))
    for i, (recon, num_h) in enumerate(zip(reconstructions, harmonics_list)):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, original, alpha=0.4, label="Original")
        plt.plot(t, recon, label=f"{num_h} harmonics")
        plt.legend()
        plt.title(f"Reconstruction with {num_h} harmonics")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_cft_illustration(t_c, continuous, t_s, sampled, save_path="outputs/cft_illustration.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(t_c, continuous, label="Continuous signal (CFT domain)", alpha=0.6)
    plt.stem(t_s, sampled, linefmt='r-', markerfmt='ro', basefmt=" ", label="Sampled signal (DTFT domain)")
    plt.legend()
    plt.title("Continuous vs Sampled Signal")
    plt.xlabel("Time (s)")
    plt.savefig(save_path)
    plt.show()

def plot_convolution_result(original, smoothed, save_path="outputs/convolution.png"):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Original", alpha=0.5)
    plt.plot(smoothed, label="Smoothed (Convolution)", linewidth=1)
    plt.legend()
    plt.title("Convolution-Based Smoothing")
    plt.savefig(save_path)
    plt.show()

def plot_pole_zero(zeros, poles, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(poles.real, poles.imag, marker='x', s=100, color='red', label='Poles')
    plt.scatter(zeros.real, zeros.imag, marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.title(title)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_filtered_comparison(original, filtered, sample_rate, save_path="outputs/filtered_comparison.png"):
    duration = len(original) / sample_rate
    time = np.linspace(0, duration, len(original))

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, original, linewidth=0.6)
    plt.title("Before Filtering")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(time, filtered, linewidth=0.6, color='green')
    plt.title("After Bandpass Filtering (20-150 Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_heartbeat_graph(filtered_signal, sample_rate, peak_times, labels, save_path="outputs/heartbeat_graph.png"):
    duration = len(filtered_signal) / sample_rate
    time = np.linspace(0, duration, len(filtered_signal))

    plt.figure(figsize=(14, 4))
    plt.plot(time, filtered_signal, linewidth=0.6)

    for t, label in zip(peak_times, labels):
        idx = int(t * sample_rate)
        if idx < len(filtered_signal):
            color = 'red' if label == "S1" else 'green'
            plt.scatter(t, filtered_signal[idx], color=color, zorder=5)
            plt.annotate(label, (t, filtered_signal[idx]), textcoords="offset points", xytext=(0, 8), ha='center')

    plt.title("Annotated Heartbeat Graph")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()