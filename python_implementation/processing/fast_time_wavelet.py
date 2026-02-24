import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq


class LiveMatlabStyleWavelet:
    def __init__(
        self,
        Fs=8e6,                       # match snippet default
        group_size=128,
        antenna_index_1based=1,
        fmin_hz=1e3,
        fmax_hz=5e6,
        voices_per_octave=12,
        contour_levels=10,
        enable_contours=True,
        dc_remove_mode="hp",
        hp_win=4096,
        time_range_sec=(0.0, None),
        time_stride=1,
    ):
        self.Fs = float(Fs)
        self.group_size = int(group_size)
        self.ant0 = int(antenna_index_1based - 1)

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
        self.cbar = None

        self.sampling_period = 1.0 / self.Fs

        # Match your working snippet wavelet config
        self.ssq_wavelet = Wavelet(('GMW', {'beta': 60}))

    def _ensure_fig(self):
        if self.fig is None:
            # Match your snippet's spectrogram figure size
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_title("CWT Magnitude (ssqueezepy)")
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
            xr = np.real(x)
            xi = np.imag(x)
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

        if x.size < 4:
            return

        if self.time_stride > 1:
            x = x[::self.time_stride]
            Fs_eff = self.Fs / self.time_stride
        else:
            Fs_eff = self.Fs

        if x.size < 4:
            return

        t = (np.arange(x.size, dtype=np.float64) / Fs_eff) + t_offset

        # -------------------------------------------------------
        # CWT on I and Q separately 
        # -------------------------------------------------------
        I = np.real(x).astype(np.float32)
        Q = np.imag(x).astype(np.float32)

        WI, scales = cwt(I, wavelet=self.ssq_wavelet)
        WQ, _      = cwt(Q, wavelet=self.ssq_wavelet)

        # Combined IQ wavelet energy (same as snippet)
        Wx = (np.abs(WI) ** 2) + (np.abs(WQ) ** 2)

        # Convert scales -> frequencies (Hz), same approach as snippet
        f = scale_to_freq(scales, self.ssq_wavelet, N=len(x), fs=Fs_eff).astype(np.float64)

        # Keep requested frequency band
        mask = np.isfinite(f) & (f >= self.fmin_hz) & (f <= self.fmax_hz)
        if np.any(mask):
            f = f[mask]
            Wx = Wx[mask, :]
        else:
            mask = np.isfinite(f)
            f = f[mask]
            Wx = Wx[mask, :]
            if f.size == 0:
                return

        # IMPORTANT: frequencies may be nonuniform -> sort like your snippet
        sort_idx = np.argsort(f)
        f_plot = f[sort_idx]
        Wx_plot = Wx[sort_idx, :]

        # Display in dB (keep your live-view scaling behavior)
        P_db = 10.0 * np.log10(Wx_plot + 1e-12)
        #P_db = Wx_plot.astype(np.float64)  # Keep as linear magnitude for now, match your snippet's vmin/vmax
        vmin = float(np.nanpercentile(P_db, 5))
        vmax = float(np.nanpercentile(P_db, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(P_db))
            vmax = float(np.nanmax(P_db))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return
            if vmax <= vmin:
                vmax = vmin + 1.0

        self.ax.set_xlim(float(t[0]), float(t[-1]))
        self.ax.set_ylim(float(f_plot[0]), float(f_plot[-1]))

        # Match snippet style: no forced cmap ("turbo"), use default colors
        if self.mesh is None:
            self.mesh = self.ax.pcolormesh(t, f_plot, P_db, shading="auto",cmap="jet",vmin=vmin,vmax=vmax )
            self.mesh.set_edgecolor("none")
            self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
            self.cbar.set_label("|CWT|")
        else:
            self.mesh.remove()
            self.mesh = self.ax.pcolormesh(t, f_plot, P_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
            self.mesh.set_edgecolor("none")
            if self.cbar is not None:
                self.cbar.update_normal(self.mesh)

        self.mesh.set_clim(vmin, vmax)
        self.ax.set_title(f"CWT Magnitude (ssqueezepy) | Group {groupIdx} | Frame {frame_index_val}")

        self._remove_contours()
        if self.enable_contours:
            try:
                # Match your snippet's contour style (black)
                self.cont = self.ax.contour(
                    t, f_plot, P_db,
                    levels=self.contour_levels,
                    linewidths=1.0,
                    colors="k"
                )
            except Exception:
                self.cont = None

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()