import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt
import pywt

# =================================================
# MATPLOTLIB INTERACTIVE SETUP
# =================================================
plt.ion()

# Polar plot globals
_polar_fig = None
_polar_ax = None
_polar_scatter = None

# Scalogram plot globals
_scalo_fig = None
_scalo_ax = None
_scalo_im = None
_scalo_cbar = None
_scalo_freqs = None
_scalo_t = None


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
        update_polar_plot_scatter_points([], [])
        return None, None  # Return None for range_bin and rd

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
        update_polar_plot_scatter_points([], [])
        return None, None

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
        return r_sel, rd

    ang_pow = np.abs(ang) ** 2
    a_idx = np.argmax(ang_pow, axis=1)

    angles = angle_axis[a_idx]
    ranges = (dets[:, 0].astype(np.float32) + 1.0) * range_bin_for_plot

    update_polar_plot_scatter_points(angles, ranges)
    
    return r_sel, rd  # Return detected range bin for micro-Doppler analysis


# =================================================
# SLOW-TIME MORLET SCALOGRAM (MICRO-DOPPLER)
# =================================================
def init_morlet_scalogram_slowtime(
    Nl: int,
    Nc: int,
    prf: float,  # Pulse Repetition Frequency
    scales,
    wavelet: str = "cmor2.5-1.0",
    output: str = "magnitude",   # "power" | "magnitude"
):
    """Initialize scalogram plot for slow-time (micro-Doppler) analysis"""
    global _scalo_fig, _scalo_ax, _scalo_im, _scalo_cbar, _scalo_freqs, _scalo_t

    # Total slow-time samples
    N_slow = Nl * Nc
    
    # Dummy signal to get freqs shape once
    x0 = np.zeros(N_slow, dtype=np.complex64)
    sampling_period = 1.0 / prf
    coeffs0, freqs_hz = pywt.cwt(x0, scales, wavelet, sampling_period=sampling_period)
    _scalo_freqs = freqs_hz.astype(np.float32)
    _scalo_t = (np.arange(N_slow, dtype=np.float32) / prf)

    if output == "magnitude":
        S0 = np.abs(coeffs0)
        cbar_label = "Magnitude"
    else:
        S0 = (np.abs(coeffs0) ** 2)
        cbar_label = "Power"

    fmin, fmax = float(np.min(_scalo_freqs)), float(np.max(_scalo_freqs))

    _scalo_fig = plt.figure(figsize=(12, 6))
    _scalo_ax = _scalo_fig.add_subplot(111)

    _scalo_im = _scalo_ax.imshow(
        S0,
        aspect="auto",
        origin="lower",
        extent=[_scalo_t[0], _scalo_t[-1], fmax, fmin],
        interpolation="bilinear",
        cmap="jet",
    )

    _scalo_ax.set_xlabel("Time (s)")
    _scalo_ax.set_ylabel("Micro-Doppler Frequency (Hz)")
    _scalo_ax.set_title("Slow-Time Morlet CWT (Micro-Doppler)")
    _scalo_ax.set_yscale("log")

    _scalo_cbar = _scalo_fig.colorbar(_scalo_im, ax=_scalo_ax)
    _scalo_cbar.set_label(cbar_label)

    plt.show()


def update_morlet_scalogram_slowtime(
    cube: np.ndarray,
    prf: float,
    range_bin: int,
    rx_idx: int,
    scales,
    wavelet: str = "cmor2.5-1.0",
    mode: str = "complex",   # "complex" | "real" | "mag"
    output: str = "magnitude",   # "power" | "magnitude"
):
    """
    Update scalogram with slow-time analysis for micro-Doppler
    
    cube: (Ns, Nl, Nrx, Nc)
    Extract slow-time signal: cube[range_bin, :, rx_idx, :] -> (Nl, Nc)
    Reshape to 1D slow-time vector: (Nl*Nc,) in Fortran order
    """
    global _scalo_fig, _scalo_ax, _scalo_im, _scalo_freqs, _scalo_t

    # cube: (Ns, Nl, Nrx, Nc)
    # Extract slow-time: (Nl, Nc) matrix for this range bin and RX
    slow_time_2d = cube[range_bin, :, rx_idx, :]
    
    # Reshape to 1D slow-time vector (Fortran order to maintain chirp sequence)
    x = slow_time_2d.reshape(-1, order='F')

    if mode == "real":
        x = np.real(x)
    elif mode == "mag":
        x = np.abs(x)
    elif mode == "complex":
        pass  # Keep as complex
    else:
        raise ValueError("mode must be 'complex', 'mag', or 'real'")

    sampling_period = 1.0 / prf
    coeffs, freqs_hz = pywt.cwt(x, scales, wavelet, sampling_period=sampling_period)

    if output == "magnitude":
        S = np.abs(coeffs)
    elif output == "power":
        S = (np.abs(coeffs) ** 2)
    else:
        raise ValueError("output must be 'power' or 'magnitude'")

    # Update image (no new window)
    _scalo_im.set_data(S)

    # Auto color scaling each frame
    vmin = float(np.min(S))
    vmax = float(np.max(S))
    if vmax > vmin:
        _scalo_im.set_clim(vmin, vmax)

    # Update title to show which slice you're plotting
    _scalo_ax.set_title(f"Slow-Time Morlet CWT | Range Bin={range_bin}, RX={rx_idx}")

    _scalo_fig.canvas.draw_idle()
    _scalo_fig.canvas.flush_events()


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\rangetest_4m"

    Ns = 256
    Nc = 12
    Nl = 128

    fs = 8e6  # ADC sampling frequency (fast-time)
    slope_MHz_us = 79

    # Slow-time parameters for micro-Doppler
    # PRF = 1 / (chirp duration + idle time)
    # For TI AWR chips, typical chirp time + idle ~ 100-500 us
    # Assuming ~200 us per chirp -> PRF ~ 5 kHz
    # Adjust based on your actual chirp configuration!
    CHIRP_DURATION_US = 200  # microseconds (ADJUST THIS!)
    PRF = 1e6 / CHIRP_DURATION_US  # Pulse Repetition Frequency in Hz

    # Plot / performance knobs
    Na = 64
    MAX_PLOT_DETS = 60
    PLOT_EVERY_N_FRAMES = 1

    # Wavelet knobs for SLOW-TIME (micro-Doppler)
    # Scales should cover the micro-Doppler frequencies you expect
    # For human motion: typically 0-20 Hz
    # For drones/vehicles: can be up to 100+ Hz
    SCALES = np.arange(1, 65)  # Wider range for micro-Doppler
    WAVELET = "cmor2.5-1.0"
    W_MODE = "complex"   # "complex" | "real" | "mag"
    W_OUTPUT = "power"   # "power" | "magnitude"

    # RX channel to analyze for micro-Doppler
    W_RX = 0  # 0..15

    # Initialize plots
    init_polar_plot(rmax=20)
    init_morlet_scalogram_slowtime(Nl=Nl, Nc=Nc, prf=PRF, scales=SCALES, 
                                     wavelet=WAVELET, output=W_OUTPUT)

    # Precompute windows & axes once
    win_r = np.hanning(Ns).astype(np.float32)
    win_d = np.hanning(Nl).astype(np.float32)
    win_a = np.hanning(16).astype(np.float32)

    angle_axis = np.arcsin(np.linspace(-1, 1, Na)).astype(np.float32)

    c = 3e8
    slope = slope_MHz_us * 1e12
    Nfft = Ns

    rbin_for_print = (c * fs) / (2 * slope * Nfft)
    range_bin_for_plot = (c * fs) / (2 * slope * Nfft)

    processed = set()
    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    frame_counter = 0

    print(f"\n{'='*60}")
    print(f"SLOW-TIME MICRO-DOPPLER ANALYSIS")
    print(f"{'='*60}")
    print(f"PRF: {PRF:.2f} Hz")
    print(f"Slow-time samples per frame: {Nl * Nc}")
    print(f"Max micro-Doppler freq: ±{PRF/2:.2f} Hz")
    print(f"{'='*60}\n")

    while True:
        try:
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

                for frame in range(2, nframes + 1):
                    cube = read_adc_bin_tda2_separate_files(DATA_FOLDER, idx, frame, Ns, Nc, Nl)
                    # cube shape: (Ns, Nl, 16, Nc)
                    frame_counter += 1

                    # ---- RANGE FFT + DETECTIONS ----
                    rng = range_fft_fast(cube, Ns, win_r)

                    if (frame_counter % PLOT_EVERY_N_FRAMES) == 0:
                        print(f"[{idx} | Frame {frame}] ", end="")
                        detected_range_bin, rd = detect_and_display_fast(
                            rng, Nl, Nc, fs, slope_MHz_us,
                            win_d, win_a,
                            angle_axis,
                            range_bin_for_plot,
                            rbin_for_print,
                            Na=Na,
                            max_plot_dets=MAX_PLOT_DETS
                        )

                        # ---- SLOW-TIME MICRO-DOPPLER ANALYSIS ----
                        # Only analyze if we have a valid detection
                        if detected_range_bin is not None:
                            # Use the detected range bin for micro-Doppler analysis
                            update_morlet_scalogram_slowtime(
                                cube, 
                                prf=PRF,
                                range_bin=detected_range_bin,
                                rx_idx=W_RX,
                                scales=SCALES,
                                wavelet=WAVELET,
                                mode=W_MODE,
                                output=W_OUTPUT
                            )
                        else:
                            # If no detection, you could analyze a fixed range bin
                            update_morlet_scalogram_slowtime(
                                cube, 
                                prf=PRF,
                                range_bin=50,
                                rx_idx=W_RX,
                                scales=SCALES,
                                wavelet=WAVELET,
                                mode=W_MODE,
                                output=W_OUTPUT
                            )
                            # or skip the update
                            pass

                processed.add(idx)
                print(f"✅ Capture {idx} done")

        except Exception as e:
            print("⚠️ Error:", e)
            import traceback
            traceback.print_exc()

        time.sleep(0.05)