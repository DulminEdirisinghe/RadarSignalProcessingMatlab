#!/usr/bin/env python3
import os
import re
import time
import struct

import numpy as np
import matplotlib.pyplot as plt
import pywt

# =================================================
# SETTINGS (match your receiver)
# =================================================
DATA_FOLDER = r"C:\radar_receiver\radar_new"   # same SAVE_DIR as radar_receiver.py
POLL_SLEEP = 0.05

# Radar sampling rate (same as your wavelet.py)
FS = 10e6

# Wavelet knobs (same idea as before)
SCALES  = np.arange(1, 33)
WAVELET = "cmor1.5-1.0"
W_MODE  = "complex"   # "complex" | "real" | "mag"
W_OUTPUT = "power"    # "power" | "magnitude"

# =================================================
# MATPLOTLIB INTERACTIVE SETUP
# =================================================
plt.ion()
_scalo_fig = None
_scalo_ax = None
_scalo_im = None
_scalo_cbar = None
_scalo_freqs = None
_scalo_t = None


def init_morlet_scalogram_plot(Ns, fs, scales, wavelet="cmor1.5-1.0", output="power"):
    """Create a single scalogram window that we update per frame."""
    global _scalo_fig, _scalo_ax, _scalo_im, _scalo_cbar, _scalo_freqs, _scalo_t

    sampling_period = 1.0 / fs

    # precompute freqs from scales (so plot y-axis is meaningful)
    freqs_hz = pywt.scale2frequency(wavelet, scales) / sampling_period
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)

    _scalo_freqs = freqs_hz
    _scalo_t = np.arange(Ns, dtype=np.float64) / fs

    # initial empty image
    S0 = np.zeros((len(scales), Ns), dtype=np.float32)

    _scalo_fig, _scalo_ax = plt.subplots(1, 1, figsize=(9, 4))
    extent = [_scalo_t[0], _scalo_t[-1], freqs_hz.min(), freqs_hz.max()]

    _scalo_im = _scalo_ax.imshow(
        S0,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    _scalo_ax.set_title("Morlet CWT Scalogram (per frame)")
    _scalo_ax.set_xlabel("Time (s)")
    _scalo_ax.set_ylabel("Frequency (Hz)")
    _scalo_ax.set_yscale("log")

    _scalo_cbar = _scalo_fig.colorbar(_scalo_im, ax=_scalo_ax)
    _scalo_cbar.set_label("Power" if output == "power" else "Magnitude")

    plt.show()


def update_morlet_scalogram_per_frame_1d(
    x,
    fs,
    scales,
    wavelet="cmor1.5-1.0",
    mode="complex",
    output="power",
    title_suffix=""
):
    """Update scalogram using 1D x (length Ns)."""
    global _scalo_fig, _scalo_ax, _scalo_im

    if mode == "real":
        x_use = np.real(x)
    elif mode == "mag":
        x_use = np.abs(x)
    elif mode == "complex":
        x_use = x
    else:
        raise ValueError("mode must be 'complex', 'mag', or 'real'")

    sampling_period = 1.0 / fs
    coeffs, _freqs_hz = pywt.cwt(x_use, scales, wavelet, sampling_period=sampling_period)

    if output == "magnitude":
        S = np.abs(coeffs).astype(np.float32)
    elif output == "power":
        S = (np.abs(coeffs) ** 2).astype(np.float32)
    else:
        raise ValueError("output must be 'power' or 'magnitude'")

    _scalo_im.set_data(S)

    vmin = float(np.min(S))
    vmax = float(np.max(S))
    if vmax > vmin:
        _scalo_im.set_clim(vmin, vmax)

    if title_suffix:
        _scalo_ax.set_title("Morlet CWT " + title_suffix)

    _scalo_fig.canvas.draw_idle()
    _scalo_fig.canvas.flush_events()


# =================================================
# WVT FILE READER
# Format:
#   "WVT1"
#   <6H: Ns, nframes, rx_total, chirp, loop, reserved>
#   repeated:
#       <H frame_idx>
#       2*Ns int16: I0,Q0,I1,Q1,...
# =================================================
def iter_wvt_frames(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"WVT1":
            raise ValueError("Not a WVT1 file: %s (magic=%r)" % (path, magic))

        hdr = f.read(12)
        if len(hdr) != 12:
            raise ValueError("Short header in %s" % path)

        Ns, nframes, rx, chirp, loop, _ = struct.unpack("<6H", hdr)

        frame_payload_bytes = 2 + (4 * Ns)  # 2 bytes frame_idx + (2*Ns int16)=4*Ns bytes
        # stream frames
        while True:
            h = f.read(2)
            if not h:
                break
            if len(h) != 2:
                break
            (frame_idx,) = struct.unpack("<H", h)

            raw = f.read(4 * Ns)
            if len(raw) != (4 * Ns):
                break

            iq = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            x = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

            yield Ns, nframes, rx, chirp, loop, frame_idx, x


def file_stable(path, checks=3, interval=0.05):
    """Simple 'size stable' check so we don't parse partial writes."""
    try:
        s0 = os.path.getsize(path)
    except OSError:
        return False
    for _ in range(checks):
        time.sleep(interval)
        try:
            s1 = os.path.getsize(path)
        except OSError:
            return False
        if s1 != s0 or s1 == 0:
            return False
        s0 = s1
    return True


# =================================================
# MAIN PIPELINE (watch folder, process each new .wvt)
# =================================================
def main():
    print("Watching:", DATA_FOLDER)
    processed = set()

    # detect .wvt files (any name)
    wvt_pat = re.compile(r".+\.wvt$", re.IGNORECASE)

    init_done = False

    while True:
        try:
            wvts = []
            for entry in os.scandir(DATA_FOLDER):
                if not entry.is_file():
                    continue
                if wvt_pat.match(entry.name):
                    wvts.append(entry.path)

            wvts.sort()  # stable ordering

            for path in wvts:
                if path in processed:
                    continue
                if not file_stable(path):
                    continue

                print("\n📥 WVT:", os.path.basename(path))

                first = True
                for Ns, nframes, rx, chirp, loop, frame_idx, x in iter_wvt_frames(path):
                    if not init_done:
                        init_morlet_scalogram_plot(
                            Ns=Ns, fs=FS, scales=SCALES, wavelet=WAVELET, output=W_OUTPUT
                        )
                        init_done = True

                    # same per-frame update behavior as before
                    title = "| rx=%d chirp=%d loop=%d | frame=%d/%d" % (rx, chirp, loop, frame_idx, nframes)
                    update_morlet_scalogram_per_frame_1d(
                        x, fs=FS,
                        scales=SCALES,
                        wavelet=WAVELET,
                        mode=W_MODE,
                        output=W_OUTPUT,
                        title_suffix=title
                    )

                processed.add(path)
                print("✅ Done:", os.path.basename(path))

        except Exception as e:
            print("⚠️ Error:", e)

        time.sleep(POLL_SLEEP)


if __name__ == "__main__":
    main()
