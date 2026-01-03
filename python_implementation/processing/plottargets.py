import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt

# =================================================
# MATPLOTLIB INTERACTIVE SETUP
# =================================================
plt.ion()
_polar_fig = None
_polar_ax = None
_polar_scatter = None


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
    iq = np.transpose(iq, (1, 3, 0, 2))  # (Ns, loops, rx, chirps)
    return iq


# =================================================
# READ CASCADE (16 RX)
# =================================================
def read_adc_bin_tda2_separate_files(folder, idx, frame, Ns, Nc, Nl):
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

    cube = np.zeros((Ns, Nl, 16, Nc), dtype=np.complex64)
    cube[:, :, 0:4, :] = cubes[0]
    cube[:, :, 4:8, :] = cubes[1]
    cube[:, :, 8:12, :] = cubes[2]
    cube[:, :, 12:16, :] = cubes[3]
    return cube


# =================================================
# RANGE FFT
# =================================================
def range_fft(cube, Nfft):
    Ns, Nl, Nrx, Nc = cube.shape
    win = np.hanning(Ns)

    slow_time = Nl * Nc
    out = np.zeros((Nfft, slow_time, Nrx), dtype=np.complex64)

    for rx in range(Nrx):
        x = cube[:, :, rx, :].reshape(Ns, slow_time, order="F")
        x -= np.mean(x, axis=0)
        x *= win[:, None]
        out[:, :, rx] = np.fft.fft(x, Nfft, axis=0)

    return out


# =================================================
# DOPPLER FFT
# =================================================
def doppler_fft(range_fft, Nl, Nc):
    Nr, _, Nrx = range_fft.shape
    rd = range_fft.reshape(Nr, Nl, Nc, Nrx, order="F")
    rd = np.mean(rd, axis=2)

    win = np.hanning(Nl)
    out = np.zeros_like(rd)

    for r in range(Nr):
        for rx in range(Nrx):
            x = rd[r, :, rx]
            x -= np.mean(x)
            x *= win
            out[r, :, rx] = np.fft.fftshift(np.fft.fft(x))

    return out


# =================================================
# STATIC CLUTTER REMOVAL (ZERO DOPPLER)
# =================================================
def remove_static_clutter(rd):
    rd[:, rd.shape[1] // 2, :] = 0
    return rd


# =================================================
# 1D CA-CFAR (TI-style)
# =================================================
def ca_cfar_1d(x, nt=8, ng=4, pfa=1e-5):
    N = len(x)
    det = np.zeros(N, dtype=bool)

    alpha = nt * (pfa ** (-1 / nt) - 1)

    for i in range(nt + ng, N - nt - ng):
        noise = (
            np.sum(x[i - ng - nt : i - ng]) +
            np.sum(x[i + ng + 1 : i + ng + nt + 1])
        ) / (2 * nt)

        if x[i] > alpha * noise:
            det[i] = True

    return det


# =================================================
# ANGLE FFT
# =================================================
def angle_fft(rd, Na=64):
    Nr, Nd, Nrx = rd.shape
    win = np.hanning(Nrx)
    ang = np.zeros((Nr, Nd, Na), dtype=np.complex64)

    for r in range(Nr):
        for d in range(Nd):
            ang[r, d] = np.fft.fftshift(
                np.fft.fft(rd[r, d] * win, Na)
            )
    return ang


# =================================================
# POLAR PLOT INIT (SCATTER)
# =================================================
def init_polar_plot():
    global _polar_fig, _polar_ax, _polar_scatter

    _polar_fig = plt.figure(figsize=(8, 8))
    _polar_ax = _polar_fig.add_subplot(111, polar=True)

    _polar_scatter = _polar_ax.scatter([], [], s=80, c="red")

    _polar_ax.set_theta_zero_location("N")
    _polar_ax.set_theta_direction(-1)
    _polar_ax.set_rmax(20)
    _polar_ax.set_title("TI-style Radar Polar Detections", pad=20)

    plt.show()


# =================================================
# POLAR PLOT UPDATE (SCATTER)
# =================================================
def update_polar_plot_scatter(ang_cube, detections, fs, slope_MHz_us):
    global _polar_scatter, _polar_fig

    c = 3e8
    Nr, _, Na = ang_cube.shape
    slope = slope_MHz_us * 1e12
    range_bin = (c * fs) / (2 * slope * Nr)

    angle_axis = np.arcsin(np.linspace(-1, 1, Na))

    angles = []
    ranges = []

    for r, d in detections:
        if r >= Nr // 2:
            continue

        ang_spec = np.abs(ang_cube[r, d]) ** 2
        a_idx = np.argmax(ang_spec)

        angles.append(angle_axis[a_idx])
        ranges.append((r + 1) * range_bin)

    if len(ranges) == 0:
        _polar_scatter.set_offsets(np.empty((0, 2)))
    else:
        pts = np.column_stack((angles, ranges))
        _polar_scatter.set_offsets(pts)

    _polar_fig.canvas.draw_idle()
    _polar_fig.canvas.flush_events()


# =================================================
# TI-STYLE DETECTION PIPELINE
# =================================================
def detect_and_display(rng_fft, Nl, Nc, fs, slope_MHz_us):
    rd = doppler_fft(rng_fft, Nl, Nc)
    rd = remove_static_clutter(rd)

    rd_pow = np.sum(np.abs(rd) ** 2, axis=2)
    rd_pow = rd_pow[: rd_pow.shape[0] // 2]

    Nr, Nd = rd_pow.shape

    # -------------------------------
    # STAGE 1: RANGE CFAR
    # -------------------------------
    range_dets = []
    for d in range(Nd):
        det_r = ca_cfar_1d(rd_pow[:, d])
        for r in np.where(det_r)[0]:
            range_dets.append((r, d))

    if len(range_dets) == 0:
        print("No detections")
        return

    # -------------------------------
    # STAGE 2: DOPPLER CFAR
    # -------------------------------
    final_dets = []
    for r, _ in range_dets:
        det_d = ca_cfar_1d(rd_pow[r, :])
        for d in np.where(det_d)[0]:
            final_dets.append((r, d))

    if len(final_dets) == 0:
        print("No detections")
        return

    final_dets = np.array(final_dets)

    # -------------------------------
    # PEAK SELECTION (TI-style)
    # -------------------------------
    powers = np.array([rd_pow[r, d] for r, d in final_dets])
    r_sel, d_sel = final_dets[np.argmax(powers)]

    c = 3e8
    Nfft = rng_fft.shape[0]
    rbin = (c * fs) / (2 * (slope_MHz_us * 1e12) * Nfft)

    print(f"Detected Range = {(r_sel + 1) * rbin:.3f} m")

    # -------------------------------
    # ANGLE + POLAR DISPLAY
    # -------------------------------
    ang = angle_fft(rd)
    update_polar_plot_scatter(ang, final_dets, fs, slope_MHz_us)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\radar_receiver\radar_new"

    Ns = 256
    Nc = 12
    Nl = 128

    fs = 10e6
    slope_MHz_us = 40.024

    init_polar_plot()

    processed = set()

    while True:
        files = os.listdir(DATA_FOLDER)
        idx_files = sorted(
            f for f in files if re.match(r"master_\d{4}_idx\.bin", f)
        )

        for idxf in idx_files:
            idx = re.findall(r"\d{4}", idxf)[0]
            if idx in processed:
                continue

            nframes = get_valid_num_frames(os.path.join(DATA_FOLDER, idxf))
            print(f"\n📥 Capture {idx} | Frames = {nframes}")

            for frame in range(2, nframes + 1):
                cube = read_adc_bin_tda2_separate_files(
                    DATA_FOLDER, idx, frame, Ns, Nc, Nl
                )
                rng = range_fft(cube, Ns)
                print(f"[{idx} | Frame {frame}] ", end="")
                detect_and_display(rng, Nl, Nc, fs, slope_MHz_us)

            processed.add(idx)
            print(f"✅ Capture {idx} done")

        time.sleep(0.05)
