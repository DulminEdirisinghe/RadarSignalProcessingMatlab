import numpy as np
import matplotlib.pyplot as plt

# ssqueezepy
# pip install ssqueezepy
from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq

# -----------------------------
# 1) Generate IQ samples (1 MHz tone)
# -----------------------------
fs = 10_000_000          # 10 MHz sampling rate
f0 = 4_000_000           # 1 MHz signal
duration = 1e-3          # 1 ms
N = int(fs * duration)

t = np.arange(N) / fs

f_start = 1_000_000      # start frequency = 1 MHz
f_end   = 4_000_000      # end frequency = 4 MHz
k = (f_end - f_start) / duration
phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)

# Complex IQ tone: I = cos(...), Q = sin(...)
#iq = np.exp(1j * 2 * np.pi * f0 * t).astype(np.complex64)
iq = np.exp(1j * phase).astype(np.complex64)
# Print basic info
print("IQ dtype:", iq.dtype)
print("IQ shape:", iq.shape)
print("First 10 IQ samples:\n", iq[:10])

# Separate I and Q
I = iq.real
Q = iq.imag

# -----------------------------
# 2) CWT with ssqueezepy
# -----------------------------
# NOTE:
# ssqueezepy's cwt() may cast complex input to real in your setup,
# so use I (real part) explicitly to avoid the warning and confusion.
wavelet = Wavelet(('GMW', {'beta': 60}))

# CWT on real part (cosine at 1 MHz)
WI, scales = cwt(I, wavelet=wavelet)
WQ, scales = cwt(Q, wavelet=wavelet)
Wx =np.abs(WI)**2 + np.abs(WQ)**2  # CWT magnitude of I and Q combined
# Convert scales -> frequencies (Hz)
# IMPORTANT: use the SAME N as the signal length
freqs_hz = scale_to_freq(scales, wavelet, N=N, fs=fs)

# -----------------------------
# 3) Plot time-domain IQ
# -----------------------------
plt.figure(figsize=(10, 3))
plt.plot(t[:300] * 1e6, I[:300], label='I')
plt.plot(t[:300] * 1e6, Q[:300], label='Q', alpha=0.8)
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("1 MHz IQ Tone (first 300 samples)")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 4) Plot CWT magnitude properly (nonuniform frequency axis)
# -----------------------------
# Frequencies from wavelet scales may not be uniformly spaced.
# Sort frequencies and use pcolormesh instead of imshow.
sort_idx = np.argsort(freqs_hz)
freqs_plot = freqs_hz[sort_idx]
Wx_plot = np.abs(Wx)[sort_idx, :]

plt.figure(figsize=(10, 5))
plt.pcolormesh(t * 1e3, freqs_plot / 1e6, Wx_plot, shading='auto')
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (MHz)")
plt.title("CWT Magnitude (ssqueezepy)")
plt.colorbar(label="|CWT|")
plt.ylim(0, 5)
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Dominant frequency estimate from average CWT energy
# -----------------------------
energy_per_scale = np.mean(np.abs(Wx), axis=1)
peak_idx = np.argmax(energy_per_scale)
print(f"Estimated dominant frequency (CWT): {freqs_hz[peak_idx]/1e6:.3f} MHz")

# -----------------------------
# 6) Optional: Frequency estimate directly from IQ phase (very accurate for clean tone)
# -----------------------------
# Phase difference method
phase = np.unwrap(np.angle(iq))
inst_freq = np.diff(phase) * fs / (2 * np.pi)   # instantaneous frequency in Hz
f_est_phase = np.mean(inst_freq)
print(f"Estimated dominant frequency (IQ phase): {f_est_phase/1e6:.6f} MHz")