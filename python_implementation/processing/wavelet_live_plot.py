import numpy as np
import os
import re
import time
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
import pywt

# =================================================
# USER RADAR CONFIG (from your Lua)  [UNCHANGED]
# =================================================
START_FREQ_GHZ      = 77.0
SLOPE_MHZ_PER_US    = 78.9857
IDLE_TIME_US        = 5.0
RAMP_END_TIME_US    = 40.0
ADC_START_TIME_US   = 6.0
ADC_SAMPLES         = 256
SAMPLE_FREQ_KSPS    = 8000.0   # ksps -> 8e6 sps
NCHIRP_LOOPS        = 128      # Nl
START_CHIRP_TX      = 0
END_CHIRP_TX        = 11
NC_CHIRPS_PER_LOOP  = (END_CHIRP_TX - START_CHIRP_TX + 1)  # Nc = 12

# Speed of light
C0 = 299_792_458.0

# Derived  [UNCHANGED]
FS_FAST = SAMPLE_FREQ_KSPS * 1e3  # Hz
SLOPE_HZ_PER_S = SLOPE_MHZ_PER_US * 1e12  # (MHz/us) -> Hz/s
TC_US = IDLE_TIME_US + RAMP_END_TIME_US
PRF_CHIRP_HZ = 1.0 / (TC_US * 1e-6)
SLOW_FS_LOOPS_HZ = PRF_CHIRP_HZ / NC_CHIRPS_PER_LOOP

# =================================================
# IDX FILE READER  [UNCHANGED]
# =================================================
def get_valid_num_frames(idx_file_path):
    with open(idx_file_path, "rb") as f:
        h32 = np.fromfile(f, dtype=np.uint32, count=6)
        return int(h32[3])

# =================================================
# READ BIN FILE (4 RX)  [UNCHANGED]
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
# READ CASCADE (16 RX)  [UNCHANGED]
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

    cubes = [read_bin_file(os.path.join(folder, n), frame, Ns, Nc, Nl) for n in names]

    _cube_buf[:, :, 0:4, :]   = cubes[0]
    _cube_buf[:, :, 4:8, :]   = cubes[1]
    _cube_buf[:, :, 8:12, :]  = cubes[2]
    _cube_buf[:, :, 12:16, :] = cubes[3]
    return _cube_buf

# =================================================
# SIMPLE 1D CA-CFAR  [UNCHANGED]
# =================================================
def ca_cfar_1d(power, guard=4, train=16, pfa=1e-3):
    x = np.asarray(power, dtype=np.float64)
    N = x.size
    thresh = np.full(N, np.nan, dtype=np.float64)
    det = np.zeros(N, dtype=bool)

    if train <= 0:
        return thresh, det

    alpha = train * (pfa ** (-1.0 / train) - 1.0)

    for i in range(N):
        l0 = i - guard - train
        l1 = i - guard
        r0 = i + guard + 1
        r1 = i + guard + train + 1
        if l0 < 0 or r1 > N:
            continue

        noise_cells = np.concatenate([x[l0:l1], x[r0:r1]])
        noise = np.mean(noise_cells) + 1e-12
        thr = alpha * noise
        thresh[i] = thr
        det[i] = (x[i] > thr)

    return thresh, det

# =================================================
# LIVE WAVELET PLOTTER (FAST-TIME)  [UNCHANGED]
# =================================================
class LiveMatlabStyleWavelet:
    def __init__(
        self,
        Fs=8e6,
        group_size=128,
        antenna_index_1based=1,
        wavelet="cmor1.0-1.5",
        fmin_hz=1e3,
        fmax_hz=5e6,
        voices_per_octave=12,
        contour_levels=10,
        enable_contours=True,
        dc_remove_mode="mean",
        hp_win=4096,
        time_range_sec=(0.0, None),
        time_stride=1,
    ):
        self.Fs = float(Fs)
        self.group_size = int(group_size)
        self.ant0 = int(antenna_index_1based - 1)

        self.wavelet = wavelet
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.voices_per_octave = int(voices_per_octave)

        self.contour_levels = int(contour_levels)
        self.enable_contours = bool(enable_contours)

        self.dc_remove_mode = dc_remove_mode
        self.hp_win = int(hp_win)

        self.time_range_sec = time_range_sec
        self.time_stride = int(time_stride)

        self.fig = None
        self.ax = None
        self.mesh = None
        self.cont = None

        self.sampling_period = 1.0 / self.Fs
        cf = pywt.central_frequency(self.wavelet)

        n_octaves = np.log2(self.fmax_hz / self.fmin_hz)
        n_freqs = int(np.ceil(n_octaves * self.voices_per_octave)) + 1
        freqs = self.fmax_hz / (2 ** (np.arange(n_freqs) / self.voices_per_octave))
        freqs = freqs[::-1]  # low->high

        self.scales = cf / (freqs * self.sampling_period)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.set_cmap("jet")
            self.ax.set_title("CWT on flattened fast-time samples (original way)")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Frequency (Hz)")
            self.ax.grid(True)
            self.ax.set_axisbelow(True)
            self.fig.show()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _remove_contours(self):
        if self.cont is None:
            return
        try:
            for coll in getattr(self.cont, "collections", []):
                try:
                    coll.remove()
                except Exception:
                    pass
        except Exception:
            pass
        self.cont = None

    def _remove_dc(self, x: np.ndarray) -> np.ndarray:
        mode = (self.dc_remove_mode or "mean").lower()
        if mode == "hp":
            win = self.hp_win
            if win <= 1 or win >= x.size:
                return x - np.mean(x)
            kernel = np.ones(win, dtype=np.float64) / win
            xr = np.real(x); xi = np.imag(x)
            xr_hp = xr - np.convolve(xr, kernel, mode="same")
            xi_hp = xi - np.convolve(xi, kernel, mode="same")
            return xr_hp + 1j * xi_hp
        return x - np.mean(x)

    def _apply_time_range(self, x: np.ndarray):
        t0, t1 = self.time_range_sec
        if t0 is None:
            t0 = 0.0
        t0 = float(max(0.0, t0))
        if t1 is not None:
            t1 = float(max(t0, t1))

        i0 = int(round(t0 * self.Fs))
        i1 = x.size if t1 is None else int(round(t1 * self.Fs))

        i0 = max(0, min(i0, x.size))
        i1 = max(i0 + 1, min(i1, x.size))
        return x[i0:i1], (i0 / self.Fs)

    def update(self, adcData, frame_index_val, groupIdx):
        self._ensure_fig()

        Ns, Nl, Nrx, Nc = adcData.shape
        adcF = np.asfortranarray(adcData)
        adc3 = adcF.reshape((Ns, Nl, Nrx * Nc), order="F")

        totalChirps = adc3.shape[1]
        numGroups = int(np.ceil(totalChirps / self.group_size))
        if groupIdx < 1 or groupIdx > numGroups:
            return

        startChirp = (groupIdx - 1) * self.group_size
        endChirp = min(groupIdx * self.group_size, totalChirps)

        selectedData = adc3[:, startChirp:endChirp, self.ant0]
        x = selectedData.reshape(-1, order="F").astype(np.complex128)

        x = self._remove_dc(x)
        x, t_offset = self._apply_time_range(x)

        if self.time_stride > 1:
            x = x[::self.time_stride]
            Fs_eff = self.Fs / self.time_stride
            sampling_period_eff = 1.0 / Fs_eff
        else:
            Fs_eff = self.Fs
            sampling_period_eff = self.sampling_period

        t = (np.arange(x.size, dtype=np.float64) / Fs_eff) + t_offset

        cfs_r, f_out = pywt.cwt(
            np.real(x), self.scales, self.wavelet,
            sampling_period=sampling_period_eff, method="fft"
        )
        cfs_i, _ = pywt.cwt(
            np.imag(x), self.scales, self.wavelet,
            sampling_period=sampling_period_eff, method="fft"
        )
        cfs = cfs_r + 1j * cfs_i

        P = (np.abs(cfs) ** 2).astype(np.float64)
        f = f_out.astype(np.float64)

        if f[0] > f[-1]:
            f = f[::-1]
            P = P[::-1, :]

        P_db = 10.0 * np.log10(P + 1e-12)
        vmin = float(np.nanpercentile(P_db, 5))
        vmax = float(np.nanpercentile(P_db, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(P_db))
            vmax = float(np.nanmax(P_db))
            if vmax <= vmin:
                vmax = vmin + 1.0

        self.ax.set_xlim(float(t[0]), float(t[-1]))
        self.ax.set_ylim(float(f[0]), float(f[-1]))

        if self.mesh is None:
            self.mesh = self.ax.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")
            self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
            self.cbar.set_label("Power (dB)")
        else:
            self.mesh.remove()
            self.mesh = self.ax.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")
            self.cbar.update_normal(self.mesh)

        self.mesh.set_clim(vmin, vmax)
        self.ax.set_title(f"FAST-TIME CWT Power (Group {groupIdx}) | Frame {frame_index_val}")

        self._remove_contours()
        if self.enable_contours:
            self.cont = self.ax.contour(t, f, P_db, levels=self.contour_levels, linewidths=1.0, colors="k")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# =================================================
# RANGE FFT + CFAR (range in meters) + DOPPLER-domain CWT  [UNCHANGED]
# =================================================
class LiveRangeCfarDopplerWavelet:
    def __init__(
        self,
        Fs_fast,
        slope_hz_per_s,
        group_size=128,
        antenna_index_1based=1,

        nfft_range=256,
        range_window="hann",

        cfar_guard=4,
        cfar_train=16,
        cfar_pfa=1e-3,

        slow_fs_hz=SLOW_FS_LOOPS_HZ,

        wavelet="cmor1.0-1.5",
        fmin_doppler_hz=5.0,
        fmax_doppler_hz=0.45,
        voices_per_octave=12,

        dc_remove_mode="mean",
        hp_win=31,

        time_range_sec=(0.0, None),

        doppler_nc_sel=0,

        range_detect_mode="sum_power",
    ):
        self.Fs_fast = float(Fs_fast)
        self.slope = float(slope_hz_per_s)
        self.group_size = int(group_size)
        self.ant0 = int(antenna_index_1based - 1)

        self.nfft_range = int(nfft_range)
        self.range_window = range_window

        self.cfar_guard = int(cfar_guard)
        self.cfar_train = int(cfar_train)
        self.cfar_pfa = float(cfar_pfa)

        self.slow_fs_hz = float(slow_fs_hz)

        self.wavelet = wavelet
        self.fmin_doppler_hz = float(fmin_doppler_hz)
        self.fmax_doppler_hz = float(fmax_doppler_hz)
        self.voices_per_octave = int(voices_per_octave)

        self.dc_remove_mode = dc_remove_mode
        self.hp_win = int(hp_win)

        self.time_range_sec = time_range_sec

        self.doppler_nc_sel = int(doppler_nc_sel)
        self.range_detect_mode = str(range_detect_mode).lower().strip()

        self.fig_r = None
        self.ax_r = None
        self.line_r = None
        self.line_thr = None
        self.scatter_det = None
        self.vline_sel = None

        self.fig_w = None
        self.ax_w = None
        self.mesh = None
        self.cbar = None

        self._build_doppler_scales()

    def _build_doppler_scales(self):
        sampling_period = 1.0 / self.slow_fs_hz
        cf = pywt.central_frequency(self.wavelet)

        if self.fmax_doppler_hz < 1.0:
            fmax = self.fmax_doppler_hz * self.slow_fs_hz
        else:
            fmax = self.fmax_doppler_hz

        fmax = min(fmax, 0.49 * self.slow_fs_hz)
        fmin = max(self.fmin_doppler_hz, 1e-6)

        n_octaves = np.log2(fmax / fmin)
        n_freqs = int(np.ceil(n_octaves * self.voices_per_octave)) + 1
        freqs = fmax / (2 ** (np.arange(n_freqs) / self.voices_per_octave))
        freqs = freqs[::-1]  # low->high

        self.doppler_scales = cf / (freqs * sampling_period)
        self.doppler_sampling_period = sampling_period

    def _ensure_figs(self):
        if self.fig_r is None:
            self.fig_r, self.ax_r = plt.subplots(figsize=(10, 4))
            self.ax_r.set_title("Range FFT + CFAR (range in meters)")
            self.ax_r.set_xlabel("Range (m)")
            self.ax_r.set_ylabel("Power (dB)")
            self.ax_r.grid(True)
            self.fig_r.show()

        if self.fig_w is None:
            self.fig_w, self.ax_w = plt.subplots(figsize=(12, 6))
            plt.set_cmap("jet")
            self.ax_w.set_title("Doppler-domain CWT (complex, same-TX across loops)")
            self.ax_w.set_xlabel("Slow time (s)")
            self.ax_w.set_ylabel("Doppler frequency (Hz)")
            self.ax_w.grid(True)
            self.ax_w.set_axisbelow(True)
            self.fig_w.show()

        self.fig_r.canvas.draw(); self.fig_r.canvas.flush_events()
        self.fig_w.canvas.draw(); self.fig_w.canvas.flush_events()

    def _remove_dc_slow(self, x):
        mode = (self.dc_remove_mode or "mean").lower()
        if mode == "hp":
            win = self.hp_win
            if win <= 1 or win >= x.size:
                return x - np.mean(x)
            kernel = np.ones(win, dtype=np.float64) / win
            xr = np.real(x); xi = np.imag(x)
            xr_hp = xr - np.convolve(xr, kernel, mode="same")
            xi_hp = xi - np.convolve(xi, kernel, mode="same")
            return xr_hp + 1j * xi_hp
        return x - np.mean(x)

    def _apply_time_range(self, x):
        t0, t1 = self.time_range_sec
        if t0 is None:
            t0 = 0.0
        t0 = float(max(0.0, t0))
        if t1 is not None:
            t1 = float(max(t0, t1))

        i0 = int(round(t0 * self.slow_fs_hz))
        i1 = x.size if t1 is None else int(round(t1 * self.slow_fs_hz))

        i0 = max(0, min(i0, x.size))
        i1 = max(i0 + 1, min(i1, x.size))
        return x[i0:i1], (i0 / self.slow_fs_hz)

    def _range_axis_m(self, K):
        fb = (np.arange(K, dtype=np.float64) * self.Fs_fast) / float(self.nfft_range)
        R = (C0 * fb) / (2.0 * self.slope)
        return R

    def update(self, adcData, frame_index_val, groupIdx):
        self._ensure_figs()

        Ns, Nl, Nrx, Nc = adcData.shape
        ant = self.ant0
        dop_nc = int(np.clip(self.doppler_nc_sel, 0, Nc - 1))

        numGroups = int(np.ceil(Nl / self.group_size))
        if groupIdx < 1 or groupIdx > numGroups:
            return
        s = (groupIdx - 1) * self.group_size
        e = min(groupIdx * self.group_size, Nl)
        M = e - s

        x_all = adcData[:, s:e, ant, :].astype(np.complex128)

        nfft = self.nfft_range
        if self.range_window == "hann":
            w = np.hanning(Ns).astype(np.float64)[:, None, None]
            X_all = np.fft.fft(x_all * w, n=nfft, axis=0)
        else:
            X_all = np.fft.fft(x_all, n=nfft, axis=0)

        K = nfft // 2
        Xp_all = X_all[:K, :, :]

        if self.range_detect_mode == "single":
            P_rng = np.mean(np.abs(Xp_all[:, :, dop_nc])**2, axis=1) + 1e-12
        else:
            P_rng = np.mean(np.mean(np.abs(Xp_all)**2, axis=1), axis=1) + 1e-12

        P_rng_db = 10.0 * np.log10(P_rng)

        thr, det = ca_cfar_1d(P_rng, guard=self.cfar_guard, train=self.cfar_train, pfa=self.cfar_pfa)
        thr_db = np.full_like(P_rng_db, np.nan)
        ok = np.isfinite(thr)
        thr_db[ok] = 10.0 * np.log10(thr[ok] + 1e-12)

        det_idx = np.where(det)[0]
        if det_idx.size > 0:
            snr_like = P_rng[det_idx] / (thr[det_idx] + 1e-12)
            sel_bin = int(det_idx[np.argmax(snr_like)])
        else:
            sel_bin = int(np.argmax(P_rng))

        slow_sig = Xp_all[sel_bin, :, dop_nc].astype(np.complex128)
        slow_sig = self._remove_dc_slow(slow_sig)

        slow_sig, t_offset = self._apply_time_range(slow_sig)
        t = (np.arange(slow_sig.size, dtype=np.float64) / self.slow_fs_hz) + t_offset

        cfs, f_out = pywt.cwt(
            slow_sig,
            self.doppler_scales,
            self.wavelet,
            sampling_period=self.doppler_sampling_period,
            method="fft",
        )

        P = (np.abs(cfs) ** 2).astype(np.float64)
        f = f_out.astype(np.float64)
        if f[0] > f[-1]:
            f = f[::-1]
            P = P[::-1, :]

        P_db = 10.0 * np.log10(P + 1e-12)
        vmin = float(np.nanpercentile(P_db, 5))
        vmax = float(np.nanpercentile(P_db, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(P_db))
            vmax = float(np.nanmax(P_db))
            if vmax <= vmin:
                vmax = vmin + 1.0

        Rm = self._range_axis_m(K)
        sel_range_m = float(Rm[sel_bin])

        if self.line_r is None:
            (self.line_r,) = self.ax_r.plot(Rm, P_rng_db, lw=1.5, label="Range power (agg)")
            (self.line_thr,) = self.ax_r.plot(Rm, thr_db, lw=1.0, label="CFAR threshold")
            self.scatter_det = self.ax_r.scatter(Rm[det_idx], P_rng_db[det_idx], s=20, label="Detections")
            self.vline_sel = self.ax_r.axvline(sel_range_m, linestyle="--", linewidth=1.5,
                                               label=f"Selected {sel_range_m:.2f} m")
            self.ax_r.legend(loc="best")
        else:
            self.line_r.set_xdata(Rm); self.line_r.set_ydata(P_rng_db)
            self.line_thr.set_xdata(Rm); self.line_thr.set_ydata(thr_db)
            self.scatter_det.remove()
            self.scatter_det = self.ax_r.scatter(Rm[det_idx], P_rng_db[det_idx], s=20)
            self.vline_sel.set_xdata([sel_range_m, sel_range_m])

        self.ax_r.set_xlim(float(Rm[0]), float(Rm[-1]))
        y_min = float(np.nanmin(P_rng_db)) - 5.0
        y_max = float(np.nanmax(P_rng_db)) + 5.0
        self.ax_r.set_ylim(y_min, y_max)
        self.ax_r.set_title(
            f"Range FFT + CFAR | Frame {frame_index_val} Group {groupIdx} | Range={sel_range_m:.2f} m | Doppler chirp={dop_nc}"
        )

        self.ax_w.set_xlim(float(t[0]), float(t[-1]))
        self.ax_w.set_ylim(float(f[0]), float(f[-1]))

        if self.mesh is None:
            self.mesh = self.ax_w.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")
            self.cbar = self.fig_w.colorbar(self.mesh, ax=self.ax_w)
            self.cbar.set_label("Power (dB)")
        else:
            self.mesh.remove()
            self.mesh = self.ax_w.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")
            self.cbar.update_normal(self.mesh)

        self.mesh.set_clim(vmin, vmax)
        self.ax_w.set_title(
            f"Doppler CWT (complex, same-TX) @ {sel_range_m:.2f} m | bin {sel_bin} | Frame {frame_index_val} Group {groupIdx} | chirp={dop_nc}"
        )

        self.fig_r.canvas.draw(); self.fig_r.canvas.flush_events()
        self.fig_w.canvas.draw(); self.fig_w.canvas.flush_events()

# =================================================
# MAIN  (NEW-FILE DETECTION = YOUR OLD APPROACH)
# =================================================
if __name__ == "__main__":
    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\4m_panton"

    Ns = ADC_SAMPLES
    Nc = NC_CHIRPS_PER_LOOP
    Nl = NCHIRP_LOOPS

    GROUP_SIZE = 128
    SHOW_GROUP = 1

    FAST_CWT_TIME_RANGE_SEC = (0.0, 0.0002)
    DOPPLER_TIME_RANGE_SEC = (0.0, None)

    DOPPLER_CHIRP_INDEX = 0

    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    print("Derived timing:")
    print(f"  Tc_us           = {TC_US:.3f} us")
    print(f"  PRF_chirp_hz     = {PRF_CHIRP_HZ:.3f} Hz")
    print(f"  slow_fs_loops_hz = {SLOW_FS_LOOPS_HZ:.3f} Hz (same-TX across loops; PRF_chirp/Nc)")
    print("Range model:")
    print(f"  Fs_fast          = {FS_FAST:.1f} Hz")
    print(f"  slope            = {SLOPE_HZ_PER_S:.3e} Hz/s")
    print(f"  Doppler chirp index (TX selector) = {DOPPLER_CHIRP_INDEX}")

    # 1) FAST-TIME CWT
    live_fast_cwt = LiveMatlabStyleWavelet(
        Fs=FS_FAST,
        group_size=GROUP_SIZE,
        antenna_index_1based=1,
        wavelet="cmor0.7-1.5",
        fmin_hz=1e3,
        fmax_hz=4e6,
        voices_per_octave=12,
        contour_levels=10,
        enable_contours=False,
        dc_remove_mode="mean",
        hp_win=4096,
        time_range_sec=FAST_CWT_TIME_RANGE_SEC,
        time_stride=1,
    )

    # 2) RANGE FFT + CFAR + DOPPLER CWT
    live_range_dopp = LiveRangeCfarDopplerWavelet(
        Fs_fast=FS_FAST,
        slope_hz_per_s=SLOPE_HZ_PER_S,
        group_size=GROUP_SIZE,
        antenna_index_1based=1,
        nfft_range=256,
        range_window="hann",
        cfar_guard=4,
        cfar_train=16,
        cfar_pfa=1e-3,
        slow_fs_hz=SLOW_FS_LOOPS_HZ,
        wavelet="cmor2.5-1.0",
        fmin_doppler_hz=5.0,
        fmax_doppler_hz=0.9,
        voices_per_octave=12,
        dc_remove_mode="mean",
        hp_win=31,
        time_range_sec=DOPPLER_TIME_RANGE_SEC,
        doppler_nc_sel=DOPPLER_CHIRP_INDEX,
        range_detect_mode="sum_power",
    )

    processed = set()

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

                for frame in range(1, nframes + 1):
                    cube = read_adc_bin_tda2_separate_files(DATA_FOLDER, idx, frame, Ns, Nc, Nl)

                    numGroups = int(np.ceil(Nl / GROUP_SIZE))
                    group_to_show = max(1, min(SHOW_GROUP, numGroups))

                    print(f"[{idx} | Frame {frame}] plots... (Group {group_to_show}/{numGroups})")

                    live_fast_cwt.update(cube, frame_index_val=frame, groupIdx=group_to_show)
                    live_range_dopp.update(cube, frame_index_val=frame, groupIdx=group_to_show)

                    # keep UI responsive
                    plt.pause(0.001)

                processed.add(idx)
                print(f" Capture {idx} done")

        except Exception as e:
            print(" Error:", e)

        time.sleep(0.05)

    # (unreached)
    # plt.show()
