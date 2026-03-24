
import torch
import os
os.environ["SSQ_GPU"] = "0"  # force CPU backend for ssqueezepy during testing

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """
    Convert 1D bin centers to bin edges (len = N+1), suitable for pcolormesh/PColorMeshItem.
    Works for nonuniform spacing.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size < 2:
        # arbitrary
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)

    d = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + 0.5 * d
    edges[0] = centers[0] - 0.5 * d[0]
    edges[-1] = centers[-1] + 0.5 * d[-1]
    return edges


# -----------------------------
# 1) Generate IQ samples (chirp)
# -----------------------------
fs = 10_000_000          # 10 MHz sampling rate
duration = 1e-3          # 1 ms
N = int(fs * duration)
t = np.arange(N, dtype=np.float64) / fs

f_start = 1_000_000      # 1 MHz
f_end   = 4_000_000      # 4 MHz
k = (f_end - f_start) / duration
phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)

iq = np.exp(1j * phase).astype(np.complex64)
I = iq.real.astype(np.float32)
Q = iq.imag.astype(np.float32)

print("IQ dtype:", iq.dtype)
print("IQ shape:", iq.shape)
print("First 10 IQ samples:\n", iq[:10])

# -----------------------------
# 2) CWT with ssqueezepy
# -----------------------------
wavelet = Wavelet(('GMW', {'beta': 20, 'gamma': 10}))  # very narrowband for better freq resolution

WI, scales = cwt(I, wavelet=wavelet)
WQ, _      = cwt(Q, wavelet=wavelet)
Wx = (np.abs(WI) ** 2) + (np.abs(WQ) ** 2)  # energy-like

freqs_hz = scale_to_freq(scales, wavelet, N=N, fs=fs).astype(np.float64)

# Sort by frequency (needed for correct y-axis)
sort_idx = np.argsort(freqs_hz)
freqs_sorted = freqs_hz[sort_idx]
Wx_sorted = Wx[sort_idx, :]

# Convert to dB for display
P_db = 10.0 * np.log10(Wx_sorted + 1e-12)

# Robust display range
vmin = float(np.nanpercentile(P_db, 5))
vmax = float(np.nanpercentile(P_db, 99.5))
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    vmin = float(np.nanmin(P_db))
    vmax = float(np.nanmax(P_db))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

# -----------------------------
# 3) Dominant frequency estimate (from mean energy)
# -----------------------------
energy_per_scale = np.mean(Wx_sorted, axis=1)
peak_idx = int(np.argmax(energy_per_scale))
f_peak = freqs_sorted[peak_idx]
print(f"Estimated dominant frequency (CWT avg energy peak): {f_peak/1e6:.3f} MHz")

# Phase-based estimate (sanity check)
phase_u = np.unwrap(np.angle(iq.astype(np.complex128)))
inst_freq = np.diff(phase_u) * fs / (2 * np.pi)
f_est_phase = float(np.mean(inst_freq))
print(f"Estimated dominant frequency (IQ phase mean): {f_est_phase/1e6:.6f} MHz")

# -----------------------------
# 4) PyQtGraph plotting
# -----------------------------
app = pg.mkQApp("CWT Frequency Scale Test (PyQtGraph)")
pg.setConfigOptions(antialias=True)

win = pg.GraphicsLayoutWidget(title="Wavelet Frequency Scale Check (PyQtGraph)")
win.resize(1200, 850)

# ---- (A) Time-domain IQ (first 300 samples) ----
p_iq = win.addPlot(row=0, col=0, title="IQ (first 300 samples)")
p_iq.setLabel("bottom", "Time", units="µs")
p_iq.setLabel("left", "Amplitude")
p_iq.showGrid(x=True, y=True, alpha=0.25)

nshow = 300
t_us = t[:nshow] * 1e6
p_iq.plot(t_us, I[:nshow], name="I")
p_iq.plot(t_us, Q[:nshow], name="Q")

# ---- (B) CWT Magnitude (dB) with correct nonuniform frequency axis ----
p_cwt = win.addPlot(row=1, col=0, title="CWT Magnitude (dB) — Correct nonuniform freq axis (PColorMeshItem)")
p_cwt.setLabel("bottom", "Time", units="ms")
p_cwt.setLabel("left", "Frequency", units="MHz")
p_cwt.showGrid(x=True, y=True, alpha=0.25)

# Build bin edges for pcolormesh-style plotting
t_ms = t * 1e3
t_edges = edges_from_centers(t_ms)
f_mhz = freqs_sorted / 1e6
f_edges = edges_from_centers(f_mhz)

# PColorMeshItem expects X and Y as 2D grids of edges
# X shape: (ny+1, nx+1), Y shape: (ny+1, nx+1), Z shape: (ny, nx)
X = np.tile(t_edges[None, :], (f_edges.size, 1))
Y = np.tile(f_edges[:, None], (1, t_edges.size))

mesh = pg.PColorMeshItem(X, Y, P_db.astype(np.float32))
p_cwt.addItem(mesh)

# Apply a colormap + levels (like your ImageItem mechanism but for mesh)
import matplotlib.pyplot as plt
mpl_cmap = plt.get_cmap("jet", 256)
lut = (mpl_cmap(np.linspace(0, 1, 256)) * 255).astype(np.ubyte)  # RGBA uint8
cmap = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)
#cmap = pg.colormap.get("jet")  # built-in
mesh.setColorMap(cmap)
mesh.setLevels((vmin, vmax))

# Optional: clamp view ranges
p_cwt.setXRange(t_ms[0], t_ms[-1], padding=0.0)
p_cwt.setYRange(f_mhz[0], f_mhz[-1], padding=0.0)

# Add a ColorBar that controls the mesh
try:
    cbar = pg.ColorBarItem(values=(vmin, vmax), colorMap=cmap, label="|CWT| (dB)")
    cbar.setImageItem(mesh, insert_in=p_cwt)
except Exception:
    cbar = None

# ---- (C) Energy vs Frequency (peak marker) ----
p_energy = win.addPlot(row=2, col=0, title="Mean CWT Energy vs Frequency (Peak should track chirp band)")
p_energy.setLabel("bottom", "Frequency", units="MHz")
p_energy.setLabel("left", "Mean energy (linear)")
p_energy.showGrid(x=True, y=True, alpha=0.25)

p_energy.plot(f_mhz, energy_per_scale.astype(np.float64))
p_energy.addLine(x=f_peak / 1e6, pen=pg.mkPen(width=2))
p_energy.setXRange(0, 5, padding=0.0)

win.show()

# Keep window open
QtWidgets.QApplication.instance().exec()