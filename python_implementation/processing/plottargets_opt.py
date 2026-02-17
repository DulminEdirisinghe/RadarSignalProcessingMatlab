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
# READ BIN FILE (4 RX)  [KEEP YOUR FIXUP EXACTLY]
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
# READ CASCADE (16 RX)  (same reading, but reuse buffer)
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

    # cubes[k] is (Ns, Nl, 4, Nc)
    _cube_buf[:, :, 0:4, :]   = cubes[0]
    _cube_buf[:, :, 4:8, :]   = cubes[1]
    _cube_buf[:, :, 8:12, :]  = cubes[2]
    _cube_buf[:, :, 12:16, :] = cubes[3]

    return _cube_buf


# =================================================
# RANGE FFT (vectorized)
# =================================================
def range_fft_fast(cube, Nfft, win_r):
    # cube: (Ns, Nl, Nrx, Nc) -> out: (Nfft, Nl*Nc, Nrx)
    Ns, Nl, Nrx, Nc = cube.shape
    slow_time = Nl * Nc

    # (Ns, Nl, Nc, Nrx) then (Ns, slow_time, Nrx) with Fortran ordering
    x = np.transpose(cube, (0, 1, 3, 2)).reshape(Ns, slow_time, Nrx, order="F")

    # remove mean over fast-time (per slow-time column, per rx)
    x = x - np.mean(x, axis=0, keepdims=True)

    # apply range window
    x = x * win_r[:, None, None]

    out = np.fft.fft(x, Nfft, axis=0).astype(np.complex64)
    return out


# =================================================
# DOPPLER FFT (vectorized)
# =================================================
def doppler_fft_fast(rng_fft, Nl, Nc, win_d):
    # rng_fft: (Nr, Nl*Nc, Nrx) -> rd: (Nr, Nl, Nrx)
    Nr, _, Nrx = rng_fft.shape

    rd = rng_fft.reshape(Nr, Nl, Nc, Nrx, order="F").mean(axis=2)

    # remove mean over doppler axis, apply doppler window
    rd = rd - rd.mean(axis=1, keepdims=True)
    rd = rd * win_d[None, :, None]

    rd = np.fft.fftshift(np.fft.fft(rd, axis=1), axes=1).astype(np.complex64)
    return rd


# =================================================
# STATIC CLUTTER REMOVAL (ZERO DOPPLER)
# =================================================
def remove_static_clutter(rd):
    rd[:, rd.shape[1] // 2, :] = 0
    return rd


# =================================================
# 1D CA-CFAR (fast via cumulative sum)
# =================================================
def ca_cfar_1d_fast(x, nt=8, ng=4, pfa=1e-5):
    x = np.asarray(x)
    N = x.size
    det = np.zeros(N, dtype=bool)

    guard = nt + ng
    if N < 2 * guard + 1:
        return det

    alpha = nt * (pfa ** (-1.0 / nt) - 1.0)

    cs = np.cumsum(x, dtype=np.float64)

    i = np.arange(guard, N - guard)

    l0 = i - ng - nt
    l1 = i - ng
    r0 = i + ng + 1
    r1 = i + ng + nt + 1

    left = cs[l1 - 1] - np.where(l0 > 0, cs[l0 - 1], 0.0)
    right = cs[r1 - 1] - cs[r0 - 1]
    noise = (left + right) / (2.0 * nt)

    det[i] = x[i] > (alpha * noise)
    return det


# =================================================
# ANGLE FFT only for selected detections (vectorized over K detections)
# =================================================
def angle_for_detections(rd, dets, Na, win_a):
    # rd: (Nr, Nd, Nrx), dets: (K,2)
    if dets.size == 0:
        return None

    r = dets[:, 0].astype(np.int32, copy=False)
    d = dets[:, 1].astype(np.int32, copy=False)

    # (K, Nrx)
    v = rd[r, d, :] * win_a[None, :]
    ang = np.fft.fftshift(np.fft.fft(v, Na, axis=1), axes=1).astype(np.complex64)
    return ang


# =================================================
# POLAR PLOT INIT (SCATTER)
# =================================================
def init_polar_plot(rmax=20):
    global _polar_fig, _polar_ax, _polar_scatter

    _polar_fig = plt.figure(figsize=(8, 8))
    _polar_ax = _polar_fig.add_subplot(111, polar=True)

    _polar_scatter = _polar_ax.scatter([], [], s=50)

    _polar_ax.set_theta_zero_location("N")
    _polar_ax.set_theta_direction(-1)
    _polar_ax.set_rmax(rmax)
    _polar_ax.set_title("TI-style Radar Polar Detections", pad=20)

    plt.show()


# =================================================
# POLAR PLOT UPDATE (SCATTER) - lightweight
# =================================================
def update_polar_plot_scatter_points(angles, ranges):
    global _polar_scatter, _polar_fig

    if len(ranges) == 0:
        _polar_scatter.set_offsets(np.empty((0, 2)))
    else:
        pts = np.column_stack((angles, ranges))
        _polar_scatter.set_offsets(pts)

    _polar_fig.canvas.draw_idle()
    _polar_fig.canvas.flush_events()


# =================================================
# TI-STYLE DETECTION PIPELINE (optimized)
# - Doppler FFT vectorized
# - CFAR faster
# - Angle FFT only for top-K detections
# =================================================
def detect_and_display_fast(
    rng_fft,
    Nl, Nc,
    fs, slope_MHz_us,
    win_d, win_a,
    angle_axis,
    range_bin_for_plot,
    rbin_for_print,
    Na=64,
    max_plot_dets=60
):
    # Doppler processing
    rd = doppler_fft_fast(rng_fft, Nl, Nc, win_d)
    rd = remove_static_clutter(rd)

    # Power map (sum over RX)
    rd_pow = np.sum(np.abs(rd) ** 2, axis=2)
    rd_pow = rd_pow[: rd_pow.shape[0] // 2]  # half-range
    Nr, Nd = rd_pow.shape

    # -------------------------------
    # STAGE 1: RANGE CFAR (per Doppler bin)
    # -------------------------------
    range_mask = np.zeros((Nr, Nd), dtype=bool)
    for d in range(Nd):
        range_mask[:, d] = ca_cfar_1d_fast(rd_pow[:, d])

    cand = np.argwhere(range_mask)
    if cand.size == 0:
        print("No detections")
        return

    # -------------------------------
    # STAGE 2: DOPPLER CFAR (only for ranges that triggered)
    # -------------------------------
    final_mask = np.zeros((Nr, Nd), dtype=bool)
    for r in np.unique(cand[:, 0]):
        det_d = ca_cfar_1d_fast(rd_pow[r, :])
        final_mask[r, det_d] = True

    dets = np.argwhere(final_mask)
    if dets.size == 0:
        print("No detections")
        return

    # -------------------------------
    # PEAK SELECTION (TI-style)
    # -------------------------------
    powers = rd_pow[dets[:, 0], dets[:, 1]]
    best = dets[np.argmax(powers)]
    r_sel, d_sel = int(best[0]), int(best[1])
    print(f"Detected Range = {(r_sel + 1) * rbin_for_print:.3f} m")

    # -------------------------------
    # LIMIT POINTS FOR PLOTTING (top-K)
    # -------------------------------
    if dets.shape[0] > max_plot_dets:
        idx = np.argpartition(powers, -max_plot_dets)[-max_plot_dets:]
        dets = dets[idx]

    # -------------------------------
    # ANGLE ESTIMATION only for selected points
    # -------------------------------
    ang = angle_for_detections(rd, dets, Na, win_a)
    if ang is None:
        update_polar_plot_scatter_points([], [])
        return

    ang_pow = np.abs(ang) ** 2
    a_idx = np.argmax(ang_pow, axis=1)

    angles = angle_axis[a_idx]
    ranges = (dets[:, 0].astype(np.float32) + 1.0) * range_bin_for_plot

    update_polar_plot_scatter_points(angles, ranges)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\rangetest_2m"

    Ns = 256
    Nc = 12
    Nl = 128

    fs = 8e6
    slope_MHz_us = 79

    # Plot 
    Na = 64
    MAX_PLOT_DETS = 64       
    PLOT_EVERY_N_FRAMES = 1  

    init_polar_plot(rmax=10)

    # Precompute windows & axes once
    win_r = np.hanning(Ns).astype(np.float32)
    win_d = np.hanning(Nl).astype(np.float32)
    win_a = np.hanning(16).astype(np.float32)  # Nrx=16

    angle_axis = np.arcsin(np.linspace(-1, 1, Na)).astype(np.float32)

    c = 3e8
    slope = slope_MHz_us * 1e12
    Nfft = Ns  # you used Ns as FFT length

    # For printing detected range (based on Nfft)
    rbin_for_print = (c * fs) / (2 * slope * Nfft)

    # For plotting ranges (based on half-range view)
    # We use Nr_total = Nfft; half-range means we keep Nfft//2 bins,
    # but the bin spacing is still set by full FFT size.
    range_bin_for_plot = (c * fs) / (2 * slope * Nfft)

    processed = set()

    # Precompile regex for speed
    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    frame_counter = 0

    while True:
        try:
            # os.scandir is faster than listdir for big folders
            idx_files = []
            for entry in os.scandir(DATA_FOLDER):
                if entry.is_file():
                    m = idx_pat.match(entry.name)
                    if m:
                        idx_files.append((m.group(1), entry.name))

            idx_files.sort(key=lambda x: x[0])

            for idx, idxf in idx_files:
                if idx in processed:
                    continue

                nframes = get_valid_num_frames(os.path.join(DATA_FOLDER, idxf))
                print(f"\n📥 Capture {idx} | Frames = {nframes}")

                for frame in range(1, nframes + 1):
                    t0 = time.perf_counter()

                    cube = read_adc_bin_tda2_separate_files(
                        DATA_FOLDER, idx, frame, Ns, Nc, Nl
                    )
                    t1 = time.perf_counter()

                    rng = range_fft_fast(cube, Ns, win_r)
                    t2 = time.perf_counter()

                    frame_counter += 1
                    if (frame_counter % PLOT_EVERY_N_FRAMES) == 0:
                        print(f"[{idx} | Frame {frame}] ", end="")
                        detect_and_display_fast(
                            rng, Nl, Nc, fs, slope_MHz_us,
                            win_d, win_a,
                            angle_axis,
                            range_bin_for_plot,
                            rbin_for_print,
                            Na=Na,
                            max_plot_dets=MAX_PLOT_DETS
                        )
                    else:
                        # still compute detection if you want (comment out to skip)
                        pass

                    t3 = time.perf_counter()
                    # uncomment if you want profiling numbers
                    # print(f"timing: read={t1-t0:.3f}s rangefft={t2-t1:.3f}s detect+plot={t3-t2:.3f}s")

                processed.add(idx)
                print(f"✅ Capture {idx} done")

        except Exception as e:
            print("⚠️ Error:", e)

        time.sleep(0.05)
