from models.loader import load_audio, get_signal_properties
from models.fourier import compute_fft, fourier_series_reconstruct, generate_cft_illustration
from models.convolution import moving_average_kernel, apply_convolution
from models.transforms import design_analog_filter, design_digital_filter, check_stability
from models.filters import apply_bandpass_filter
from models.peak_detection import get_envelope, detect_peaks, classify_s1_s2, calculate_heart_rate

from views.plots import (
    plot_waveform, plot_spectrum, plot_fourier_series, plot_cft_illustration,
    plot_convolution_result, plot_pole_zero, plot_filtered_comparison, plot_heartbeat_graph
)


def run_basic_analysis(filepath):
    """Step 1: Load audio and show basic properties + waveform."""
    signal, sample_rate = load_audio(filepath)
    props = get_signal_properties(signal, sample_rate)

    print("----- Signal Properties -----")
    for key, value in props.items():
        print(f"{key}: {value}")

    plot_waveform(signal, sample_rate)
    return signal, sample_rate


def run_frequency_analysis(signal, sample_rate):
    """Step 2: FFT magnitude + phase spectrum."""
    freqs, magnitude, phase = compute_fft(signal, sample_rate)
    plot_spectrum(freqs, magnitude, phase)
    return freqs, magnitude, phase


def run_fourier_series_demo(signal, sample_rate):
    """Step 3: Fourier Series reconstruction on one segment."""
    segment = signal[0:sample_rate]  # ~1 second segment
    t = [i / sample_rate for i in range(len(segment))]

    harmonics_list = [2, 5, 15, 50]
    reconstructions = [fourier_series_reconstruct(segment, h) for h in harmonics_list]

    plot_fourier_series(t, segment, reconstructions, harmonics_list)


def run_cft_illustration():
    """Step 4: Conceptual CFT vs DTFT illustration."""
    t_c, cont, t_s, samp = generate_cft_illustration()
    plot_cft_illustration(t_c, cont, t_s, samp)


def run_convolution_demo(signal):
    """Step 5: Convolution-based smoothing demo."""
    kernel = moving_average_kernel(window_size=5)
    smoothed = apply_convolution(signal, kernel)
    plot_convolution_result(signal, smoothed)
    return smoothed


def run_filter_design(sample_rate, low=20, high=150, order=4):
    """Step 6: Laplace (analog) + Z-transform (digital) filter design."""
    # Analog (Laplace domain)
    b_a, a_a, zeros_a, poles_a, gain_a = design_analog_filter(order, low, high)
    plot_pole_zero(zeros_a, poles_a, "Analog Filter Pole-Zero Plot (s-plane)", "outputs/s_plane.png")

    # Digital (Z-transform domain)
    b_d, a_d, zeros_d, poles_d, gain_d = design_digital_filter(order, low, high, sample_rate)
    plot_pole_zero(zeros_d, poles_d, "Digital Filter Pole-Zero Plot (z-plane)", "outputs/z_plane.png")

    is_stable, magnitudes = check_stability(poles_d)
    print(f"Digital filter stable: {is_stable}")
    print(f"Pole magnitudes: {magnitudes}")

    return b_d, a_d


def run_filtering(signal, sample_rate, b, a):
    """Step 7: Apply the digital filter to the real signal."""
    filtered = apply_bandpass_filter(signal, b, a)
    plot_filtered_comparison(signal, filtered, sample_rate)
    return filtered


def run_heartbeat_detection(filtered_signal, sample_rate):
    """Step 8: Envelope -> peak detection -> S1/S2 classification -> heart rate."""
    envelope = get_envelope(filtered_signal, sample_rate)
    peaks, _ = detect_peaks(envelope, sample_rate)
    peak_times, labels = classify_s1_s2(peaks, sample_rate)
    heart_rate = calculate_heart_rate(peak_times, labels)

    print(f"Detected {len(peaks)} peaks")
    print(f"Heart Rate: {heart_rate} BPM")

    plot_heartbeat_graph(filtered_signal, sample_rate, peak_times, labels)
    return heart_rate, peak_times, labels