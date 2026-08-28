import librosa

def load_audio(filepath):
    """Load a .wav file and return signal + sample rate."""
    signal, sample_rate = librosa.load(filepath, sr=None)
    return signal, sample_rate

def get_signal_properties(signal, sample_rate):
    """Return basic signal properties as a dictionary."""
    duration = len(signal) / sample_rate
    return {
        "sample_rate": sample_rate,
        "duration": round(duration, 2),
        "num_samples": len(signal),
        "max_amplitude": float(signal.max()),
        "min_amplitude": float(signal.min()),
        "mean_amplitude": float(signal.mean()),
    }