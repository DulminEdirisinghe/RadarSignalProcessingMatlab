import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq


# ============================================================
# Simple profiling utility
# ============================================================

@dataclass
class _Stat:
    count: int = 0
    total: float = 0.0
    min_t: float = float("inf")
    max_t: float = 0.0

    def add(self, dt: float):
        self.count += 1
        self.total += dt
        if dt < self.min_t:
            self.min_t = dt
        if dt > self.max_t:
            self.max_t = dt


@dataclass
class TimeProfiler:
    name: str = "Profiler"
    stats: Dict[str, _Stat] = field(default_factory=dict)

    @contextmanager
    def section(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            if key not in self.stats:
                self.stats[key] = _Stat()
            self.stats[key].add(dt)

    def print_summary(self, top_n=None):
        if not self.stats:
            print(f"\n[{self.name}] No timing data collected yet.")
            return

        total_all = sum(s.total for s in self.stats.values())
        items = sorted(self.stats.items(), key=lambda kv: kv[1].total, reverse=True)
        if top_n is not None:
            items = items[:top_n]

        print(f"\n{'=' * 90}")
        print(f"{self.name} - Timing Summary")
        print(f"{'-' * 90}")
        print(f"{'Section':35s} {'Calls':>8s} {'Total(s)':>12s} {'Avg(ms)':>12s} {'Min(ms)':>12s} {'Max(ms)':>12s} {'%':>8s}")
        print(f"{'-' * 90}")

        for k, s in items:
            avg_ms = (s.total / s.count) * 1e3 if s.count else 0.0
            min_ms = s.min_t * 1e3 if s.count else 0.0
            max_ms = s.max_t * 1e3 if s.count else 0.0
            pct = (100.0 * s.total / total_all) if total_all > 0 else 0.0
            print(f"{k:35s} {s.count:8d} {s.total:12.6f} {avg_ms:12.3f} {min_ms:12.3f} {max_ms:12.3f} {pct:8.2f}")

        print(f"{'-' * 90}")
        print(f"{'TOTAL':35s} {'':8s} {total_all:12.6f}")
        print(f"{'=' * 90}\n")


# ============================================================
# Main class
# ============================================================

class LiveMatlabStyleWavelet:
    def __init__(
        self,
        Fs=8e6,
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
        enable_profiling=True,
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

        self.ssq_wavelet = Wavelet(('GMW', {'beta': 60}))

        self.enable_profiling = bool(enable_profiling)
        self.prof = TimeProfiler("WaveletProfiler")

    @contextmanager
    def _p(self, key: str):
        if self.enable_profiling:
            with self.prof.section(key):
                yield
        else:
            yield

    def print_profile_summary(self, top_n=None):
        self.prof.print_summary(top_n=top_n)

    def _ensure_fig(self):
        with self._p("plot.ensure_fig"):
            if self.fig is None:
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
        with self._p("plot.remove_contours"):
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
            with self._p("dc.hp_total"):
                win = self.hp_win
                if win <= 1 or win >= x.size:
                    with self._p("dc.mean_fallback"):
                        return x - np.mean(x)

                with self._p("dc.hp_kernel"):
                    kernel = np.ones(win, dtype=np.float64) / win

                with self._p("dc.hp_split_real_imag"):
                    xr = np.real(x)
                    xi = np.imag(x)

                with self._p("dc.hp_conv_real"):
                    xr_hp = xr - np.convolve(xr, kernel, mode="same")

                with self._p("dc.hp_conv_imag"):
                    xi_hp = xi - np.convolve(xi, kernel, mode="same")

                with self._p("dc.hp_recombine"):
                    return xr_hp + 1j * xi_hp

        with self._p("dc.mean"):
            return x - np.mean(x)

    def _apply_time_range(self, x: np.ndarray):
        with self._p("time.apply_range"):
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
        with self._p("update.total"):
            self._ensure_fig()

            with self._p("prep.shape_reshape"):
                Ns, Nl, Nrx, Nc = adcData.shape
                #adcF = np.asfortranarray(adcData)
                adc3 = adcData.reshape((Ns, Nl, Nrx * Nc))

            with self._p("prep.group_calc"):
                totalChirps = adc3.shape[1]
                numGroups = int(np.ceil(totalChirps / self.group_size))
                if groupIdx < 1 or groupIdx > numGroups:
                    return

                startChirp = (groupIdx - 1) * self.group_size
                endChirp = min(groupIdx * self.group_size, totalChirps)

            with self._p("prep.select_channel_flatten"):
                selectedData = adc3[:, startChirp:endChirp, self.ant0]
                x = selectedData.reshape(-1, order="F").astype(np.complex128)

            x = self._remove_dc(x)
            x, t_offset = self._apply_time_range(x)

            with self._p("prep.size_checks_1"):
                if x.size < 4:
                    return

            with self._p("prep.time_stride"):
                if self.time_stride > 1:
                    x = x[::self.time_stride]
                    Fs_eff = self.Fs / self.time_stride
                else:
                    Fs_eff = self.Fs

            with self._p("prep.size_checks_2"):
                if x.size < 4:
                    return

            with self._p("prep.time_vector"):
                t = (np.arange(x.size, dtype=np.float64) / Fs_eff) + t_offset

            # -------------------------------------------------------
            # CWT on I and Q separately
            # -------------------------------------------------------
            with self._p("cwt.split_IQ"):
                I = np.real(x).astype(np.float32)
                Q = np.imag(x).astype(np.float32)

            with self._p("cwt.I"):
                WI, scales = cwt(I, wavelet=self.ssq_wavelet)

            with self._p("cwt.Q"):
                WQ, _ = cwt(Q, wavelet=self.ssq_wavelet)

            with self._p("cwt.combine_energy"):
                Wx = (np.abs(WI) ** 2) + (np.abs(WQ) ** 2)

            with self._p("cwt.scale_to_freq"):
                f = scale_to_freq(scales, self.ssq_wavelet, N=len(x), fs=Fs_eff).astype(np.float64)

            with self._p("cwt.band_mask"):
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

            with self._p("cwt.sort_freq"):
                sort_idx = np.argsort(f)
                f_plot = f[sort_idx]
                Wx_plot = Wx[sort_idx, :]

            with self._p("post.db_convert"):
                #P_db = 10.0 * np.log10(Wx_plot + 1e-12)
                P_db = Wx_plot

            with self._p("post.percentiles"):
                vmin = float(np.nanpercentile(P_db, 5))
                vmax = float(np.nanpercentile(P_db, 99.5))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmin = float(np.nanmin(P_db))
                    vmax = float(np.nanmax(P_db))
                    if not np.isfinite(vmin) or not np.isfinite(vmax):
                        return
                    if vmax <= vmin:
                        vmax = vmin + 1.0

            with self._p("plot.axes_limits"):
                self.ax.set_xlim(float(t[0]), float(t[-1]))
                self.ax.set_ylim(float(f_plot[0]), float(f_plot[-1]))

            with self._p("plot.mesh"):
                if self.mesh is None:
                    self.mesh = self.ax.pcolormesh(t, f_plot, P_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
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
                with self._p("plot.contours"):
                    try:
                        self.cont = self.ax.contour(
                            t, f_plot, P_db,
                            levels=self.contour_levels,
                            linewidths=1.0,
                            colors="k"
                        )
                    except Exception:
                        self.cont = None

            with self._p("plot.canvas_draw"):
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()