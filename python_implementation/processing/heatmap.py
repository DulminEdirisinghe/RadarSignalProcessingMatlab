import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt

# =================================================
# MATPLOTLIB INTERACTIVE SETUP
# =================================================
plt.ion()
_fig = None
_ax = None
_im = None
_ra_map_avg = None   # temporal accumulator


# =================================================
# IDX FILE READER
# =================================================
def get_valid_num_frames(idx_file_path):
    with open(idx_file_path, "rb") as f:
        h32 = np.fromfile(f, dtype=np.uint32, count=6)
        return int(h32[3])


# =================================================
# READ BIN FILE (4 RX)
# =================================================
def read_bin_file(file_path, frame_idx, Ns, Nc, Nl, numRX=4):
    samples_per_frame = Ns * Nc * Nl * numRX * 2
    offset = (frame_idx - 1) * samples_per_frame * 2

    with open(file_path, "rb") as f:
        f.seek(offset)
        raw = np.fromfile(f, dtype=np.uint16, count=samples_per_frame)

    raw = raw.astype(np.int32)
    raw[raw >= 2**15] -= 2**16
    iq = raw[0::2] + 1j * raw[1::2]

    iq = iq.reshape(numRX, Ns, Nc, Nl, order="F")
    iq = np.transpose(iq, (1, 3, 0, 2))  # (Ns, Nl, rx, Nc)
    return iq


# =================================================
# READ CASCADE (16 RX)
# =================================================
_cube_buf = None

def read_adc_bin_tda2_separate_files(folder, idx, frame, Ns, Nc, Nl):
    global _cube_buf
    if _cube_buf is None or _cube_buf.shape != (Ns, Nl, 16, Nc):
        _cube_buf = np.empty((Ns, Nl, 16, Nc), dtype=np.complex64)

    names = [
        f"master_{idx}_data.bin",
        f"slave1_{idx}_data.bin",
        f"slave2_{idx}_data.bin",
        f"slave3_{idx}_data.bin",
    ]

    cubes = [
        read_bin_file(os.path.join(folder, n), frame, Ns, Nc, Nl)
        for n in names
    ]

    _cube_buf[:, :, 0:4, :]   = cubes[0]
    _cube_buf[:, :, 4:8, :]   = cubes[1]
    _cube_buf[:, :, 8:12, :]  = cubes[2]
    _cube_buf[:, :, 12:16, :] = cubes[3]

    return _cube_buf


# =================================================
# RANGE FFT
# =================================================
def range_fft_fast(cube, Nfft, win_r):
    Ns, Nl, Nrx, Nc = cube.shape
    slow_time = Nl * Nc

    x = np.transpose(cube, (0, 1, 3, 2)).reshape(Ns, slow_time, Nrx, order="F")
    x = x - np.mean(x, axis=0, keepdims=True)
    x = x * win_r[:, None, None]

    return np.fft.fft(x, Nfft, axis=0).astype(np.complex64)


# =================================================
# DOPPLER FFT
# =================================================
def doppler_fft_fast(rng_fft, Nl, Nc, win_d):
    Nr, _, Nrx = rng_fft.shape

    rd = rng_fft.reshape(Nr, Nl, Nc, Nrx, order="F").mean(axis=2)
    rd = rd - rd.mean(axis=1, keepdims=True)
    rd = rd * win_d[None, :, None]

    return np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1).astype(np.complex64)


# =================================================
# ZERO-DOPPLER RANGE–AZIMUTH MAP (FIXED)
# =================================================
def build_zero_doppler_ra_map(rd, Na, win_a):
    """
    rd: (Nr, Nd, Nrx)
    """
    Nd = rd.shape[1]

    # --- Zero Doppler slice (STATIC OBJECTS)
    zd = rd[:, Nd // 2, :]

    # --- Remove static bias / ground clutter
    zd = zd - np.mean(zd, axis=0, keepdims=True)

    # --- Coherent RX summation
    v = zd * win_a[None, :]
    v = np.sum(v, axis=1, keepdims=True)

    # --- Angle FFT
    ang = np.fft.fftshift(
        np.fft.fft(v, Na, axis=1),
        axes=1
    )

    return np.abs(ang) ** 2   # (Nr, Na)


# =================================================
# HEATMAP INIT
# =================================================
def init_heatmap(Na, Nr, range_bin, angle_axis):
    global _fig, _ax, _im

    _fig, _ax = plt.subplots(figsize=(6, 10))

    extent = [
        np.degrees(angle_axis[0]),
        np.degrees(angle_axis[-1]),
        0,
        Nr * range_bin
    ]

    _im = _ax.imshow(
        np.zeros((Nr, Na)),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis"
    )

    _ax.set_xlabel("Azimuth (deg)")
    _ax.set_ylabel("Range (m)")
    _ax.set_title("Range–Azimuth Heatmap (Zero Doppler)")

    plt.colorbar(_im, ax=_ax, label="Power (dB)")
    plt.show()


# =================================================
# HEATMAP UPDATE (WITH TEMPORAL AVERAGING)
# =================================================
def update_heatmap(power_map, alpha=0.85):
    global _im, _fig, _ra_map_avg

    if _ra_map_avg is None:
        _ra_map_avg = power_map
    else:
        _ra_map_avg = alpha * _ra_map_avg + (1 - alpha) * power_map

    Z = 10 * np.log10(_ra_map_avg + 1e-6)

    _im.set_data(Z)
    _im.set_clim(
        vmin=np.percentile(Z, 40),
        vmax=np.percentile(Z, 99.8)
    )

    _fig.canvas.draw_idle()
    _fig.canvas.flush_events()


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\rangetest_2m"

    # Radar parameters
    Ns = 256
    Nc = 12
    Nl = 128
    Na = 64

    fs = 8e6
    slope_MHz_us = 79

    c = 3e8
    slope = slope_MHz_us * 1e12
    range_bin = (c * fs) / (2 * slope * Ns)

    # Windows
    win_r = np.hanning(Ns).astype(np.float32)
    win_d = np.hanning(Nl).astype(np.float32)
    win_a = np.hanning(16).astype(np.float32)

    angle_axis = np.arcsin(np.linspace(-1, 1, Na))

    init_heatmap(Na, Ns // 2, range_bin, angle_axis)

    processed = set()
    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    while True:
        try:
            for entry in os.scandir(DATA_FOLDER):
                if not entry.is_file():
                    continue

                m = idx_pat.match(entry.name)
                if not m:
                    continue

                idx = m.group(1)
                if idx in processed:
                    continue

                nframes = get_valid_num_frames(entry.path)
                print(f"\n📥 Capture {idx} | Frames = {nframes}")

                for frame in range(1, nframes + 1):
                    cube = read_adc_bin_tda2_separate_files(
                        DATA_FOLDER, idx, frame, Ns, Nc, Nl
                    )

                    rng = range_fft_fast(cube, Ns, win_r)
                    rd = doppler_fft_fast(rng, Nl, Nc, win_d)

                    ra_map = build_zero_doppler_ra_map(
                        rd[:Ns // 2], Na, win_a
                    )

                    update_heatmap(ra_map)

                processed.add(idx)
                print(f"✅ Capture {idx} done")

        except Exception as e:
            print("⚠️ Error:", e)

        time.sleep(0.05)
