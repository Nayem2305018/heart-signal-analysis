from controllers.pipeline import (
    run_basic_analysis, run_frequency_analysis, run_fourier_series_demo,
    run_cft_illustration, run_convolution_demo, run_filter_design,
    run_filtering, run_heartbeat_detection
)

if __name__ == "__main__":
    filepath = "data/training/training-a/a0001.wav"

    # Step 1: Basic properties
    signal, sample_rate = run_basic_analysis(filepath)

    # Step 2: FFT spectrum
    run_frequency_analysis(signal, sample_rate)

    # Step 3: Fourier Series demo
    run_fourier_series_demo(signal, sample_rate)

    # Step 4: CFT illustration
    run_cft_illustration()

    # Step 5: Convolution demo
    run_convolution_demo(signal)

    # Step 6: Filter design (Laplace + Z-transform)
    b, a = run_filter_design(sample_rate)

    # Step 7: Apply filtering
    filtered_signal = run_filtering(signal, sample_rate, b, a)

    # Step 8: Heartbeat detection + heart rate
    run_heartbeat_detection(filtered_signal, sample_rate)