import librosa
import matplotlib.pyplot as plt
import numpy as np

# ---- Step 1: Load the audio file ----
filepath = "audio/training/training-a/a0001.wav"   # change this to your actual filename
signal, sample_rate = librosa.load(filepath, sr=None)  # sr=None keeps the original sampling rate

# ---- Step 2: Print basic signal properties ----
duration = len(signal) / sample_rate

print("----- Signal Properties -----")
print(f"Sample rate      : {sample_rate} Hz")
print(f"Duration         : {duration:.2f} seconds")
print(f"Number of samples: {len(signal)}")
print(f"Max amplitude    : {np.max(signal):.4f}")
print(f"Min amplitude    : {np.min(signal):.4f}")
print(f"Mean amplitude   : {np.mean(signal):.4f}")

# ---- Step 3: Create time axis for plotting ----
time = np.linspace(0, duration, len(signal))

# ---- Step 4: Plot the waveform ----
plt.figure(figsize=(12, 4))
plt.plot(time, signal, linewidth=0.7)
plt.title("Raw Heart Sound Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ---- Step 5: Save and show the plot ----
plt.savefig("raw_waveform.png")
print("\nPlot saved to outputs/raw_waveform.png")

plt.show()