import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sys

wav_path = sys.argv[1]
audio, sr = sf.read(wav_path)
if audio.ndim > 1:
    audio = audio[:, 0]

# Limit to first 30 s to keep the plot lightweight
min_sec = 22
max_sec = 30
print(f"Original audio: {audio.shape}, sr={sr}")
audio = audio[int(sr*min_sec): int(sr*max_sec)]
print(f"Sliced audio: {audio.shape}")

f, t, Sxx = spectrogram(audio, fs=sr, window="hann",
                        nperseg=2048, noverlap=1536,
                        scaling="density", mode="magnitude")

print(f"Spectrogram shapes: f={f.shape}, t={t.shape}, Sxx={Sxx.shape}")


plt.figure(figsize=(9, 4), dpi=150)
plt.pcolormesh(t, f, 20 * np.log10(Sxx + 1e-12))
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("W6EFI KN6UBF (RX by W6EFI)")
plt.ylim(0, 4000)
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.show()

