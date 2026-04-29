import torch
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import os
os.environ["SSQ_GPU"] = "0"   
from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


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

        print(f"\n{'=' * 95}")
        print(f"{self.name} - Timing Summary")
        print(f"{'-' * 95}")
        print(f"{'Section':38s} {'Calls':>8s} {'Total(s)':>12s} {'Avg(ms)':>12s} {'Min(ms)':>12s} {'Max(ms)':>12s} {'%':>8s}")
        print(f"{'-' * 95}")
        for k, s in items:
            avg_ms = (s.total / s.count) * 1e3 if s.count else 0.0
            min_ms = s.min_t * 1e3 if s.count else 0.0
            max_ms = s.max_t * 1e3 if s.count else 0.0
            pct = (100.0 * s.total / total_all) if total_all > 0 else 0.0
            print(f"{k:38s} {s.count:8d} {s.total:12.6f} {avg_ms:12.3f} {min_ms:12.3f} {max_ms:12.3f} {pct:8.2f}")
        print(f"{'-' * 95}")
        print(f"{'TOTAL':38s} {'':8s} {total_all:12.6f}")
        print(f"{'=' * 95}\n")


# ============================================================
# PyQtGraph-based live wavelet viewer
# ============================================================

class LiveFastWaveletPG:
    """
    PyQtGraph replacement for the Matplotlib wavelet plot.
    Optimized for fast image updates (heatmap style).
    """

    def __init__(
        self,
        Fs=8e6,
        group_size=128,
        antenna_index_1based=1,
        fmin_hz=1e3,
        fmax_hz=5e6,
        n_freq_bins=512,              
        contour_levels=10,            
        enable_contours=False,        
        dc_remove_mode="hp",
        hp_win=4096,
        time_range_sec=(0.0, None),
        time_stride=1,
        enable_profiling=True,
        enable_plot=True,
        draw_every_n=1,               # draw only every N updates
        save_images_dir=None,         # directory to save heatmap images (None = disabled)
        output_size=(256, 256),       # standardized output image size (height, width)
        power_vmin=None,              # fixed global min power (None = auto calculate from frame)
        power_vmax=None,              # fixed global max power (None = auto calculate from frame)
    ):
        self.Fs = float(Fs)
        self.group_size = int(group_size)
        self.ant0 = int(antenna_index_1based - 1)

        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.n_freq_bins = int(n_freq_bins)

        self.contour_levels = int(contour_levels)
        self.enable_contours = bool(enable_contours)

        self.dc_remove_mode = dc_remove_mode
        self.hp_win = int(hp_win)

        self.time_range_sec = time_range_sec
        self.time_stride = int(time_stride)

        self.enable_profiling = bool(enable_profiling)
        self.prof = TimeProfiler("WaveletProfiler-PyQtGraph")

        self.enable_plot = bool(enable_plot)
        self.draw_every_n = max(1, int(draw_every_n))
        self._update_counter = 0

        self.ssq_wavelet = Wavelet('morlet') 

        # Image saving configuration
        self.save_images_dir = save_images_dir
        self.output_size = tuple(output_size) if output_size else (256, 256)
        if self.save_images_dir and not os.path.exists(self.save_images_dir):
            os.makedirs(self.save_images_dir, exist_ok=True)

        # PyQtGraph UI objects
        self.app = None
        self.win = None
        self.plot = None
        self.img = None
        self.cbar = None
        self._last_shape = None

        # Fixed power scale settings
        self.power_vmin = power_vmin
        self.power_vmax = power_vmax

        # Image save counter
        self._image_save_counter = 0

    @contextmanager
    def _p(self, key: str):
        if self.enable_profiling:
            with self.prof.section(key):
                yield
        else:
            yield

    def print_profile_summary(self, top_n=None):
        self.prof.print_summary(top_n=top_n)

    # -----------------------------
    # UI setup
    # -----------------------------
    def _ensure_ui(self):
        if not self.enable_plot:
            return

        with self._p("plot.ensure_ui"):
            if self.win is not None:
                return

            self.app = pg.mkQApp("CWT Magnitude (PyQtGraph)")

            pg.setConfigOptions(imageAxisOrder='row-major')
            self.win = pg.GraphicsLayoutWidget(show=False, title="CWT Magnitude (PyQtGraph)")
            self.win.resize(1100, 600)

            self.plot = self.win.addPlot(row=0, col=0)
            self.plot.setLabel("bottom", "Time", units="s")
            self.plot.setLabel("left", "Frequency", units="Hz")
            self.plot.showGrid(x=True, y=True, alpha=0.2)

            self.img = pg.ImageItem(axisOrder='row-major')
            self.plot.addItem(self.img)

            # Jet colormap via matplotlib LUT
            import matplotlib.pyplot as plt
            mpl_cmap = plt.get_cmap("jet", 256)
            lut = (mpl_cmap(np.linspace(0, 1, 256)) * 255).astype(np.ubyte)  # RGBA uint8
            cmap = pg.ColorMap(pos=np.linspace(0, 1, 256), color=lut)
            self.img.setColorMap(cmap)

            try:
                self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=cmap, label="|CWT| (dB)")
                self.cbar.setImageItem(self.img, insert_in=self.plot)
            except Exception:
                try:
                    self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=cmap, label="|CWT| (dB)")
                    self.cbar.setImageItem(self.img)
                except Exception:
                    self.cbar = None

            self.win.show()
            self.win.raise_()
            self.win.activateWindow()

            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()
                app.processEvents()

    # -----------------------------
    # Signal prep helpers
    # -----------------------------
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

    def _resample_to_uniform_freq(
        self, P_db: np.ndarray, f_plot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Resample P_db (non-uniform freq axis) onto a uniform linear freq grid.

        core fix:
          - f_plot comes from scale_to_freq → logarithmically spaced
          - ImageItem.setRect maps pixels linearly → wrong axis without resampling
          - After resampling to f_uniform (linear), each pixel row = equal Hz step
            so setRect(t0, f0, dt, df) is exact.

        Uses nearest-neighbour lookup (argmin per target bin) — very fast for
        typical sizes (<1024 freq bins × <2000 time samples).
        """
        with self._p("post.freq_resample"):
            n_out = self.n_freq_bins
            f_uniform = np.linspace(f_plot[0], f_plot[-1], n_out)

            # For each uniform target freq, find the closest non-uniform source row
            indices = np.searchsorted(f_plot, f_uniform)
            indices = np.clip(indices, 0, len(f_plot) - 1)

            # Snap to nearest (not just lower) by checking both neighbours
            indices_prev = np.clip(indices - 1, 0, len(f_plot) - 1)
            dist_next = np.abs(f_plot[indices] - f_uniform)
            dist_prev = np.abs(f_plot[indices_prev] - f_uniform)
            nearest = np.where(dist_prev < dist_next, indices_prev, indices)

            P_resampled = P_db[nearest, :]  # shape: (n_out, n_time)

        return P_resampled, f_uniform

    def _heatmap_to_rgb(self, heatmap_data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        Convert heatmap data to RGB using 'jet' colormap.
        
        Args:
            heatmap_data: 2D array (height, width)
            vmin: minimum value for normalization
            vmax: maximum value for normalization
            
        Returns:
            RGB image array (height, width, 3) with uint8 values
        """
        with self._p("save.heatmap_to_rgb"):
            # Normalize to [0, 1]
            if vmax <= vmin:
                normalized = np.zeros_like(heatmap_data, dtype=np.float32)
            else:
                normalized = np.clip((heatmap_data - vmin) / (vmax - vmin), 0, 1).astype(np.float32)
            
            # Apply jet colormap
            cmap = cm.get_cmap('jet')
            rgba = cmap(normalized)  # shape: (H, W, 4)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8, drop alpha
            
        return rgb

    def _save_heatmap_image(self, 
                           heatmap_data: np.ndarray, 
                           vmin: float, 
                           vmax: float,
                           frame_index: int, 
                           group_index: int) -> str:
        """
        Save heatmap as a standardized-size image.
        
        Args:
            heatmap_data: 2D normalized heatmap array
            vmin: minimum value for colormap normalization
            vmax: maximum value for colormap normalization
            frame_index: frame number for filename
            group_index: group number for filename
            
        Returns:
            Path to saved image, or empty string if save disabled
        """
        if not self.save_images_dir:
            return ""
            
        try:
            with self._p("save.total"):
                # Flip vertically so low frequencies appear at bottom (standard frequency plot orientation)
                heatmap_data_flipped = np.flipud(heatmap_data)
                
                # Convert to RGB
                rgb_image = self._heatmap_to_rgb(heatmap_data_flipped, vmin, vmax)
                
                # Convert to PIL Image
                with self._p("save.pil_convert"):
                    img = Image.fromarray(rgb_image, mode='RGB')
                
                # Resize to standardized output size
                with self._p("save.resize"):
                    h, w = self.output_size
                    img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
                
                # Generate filename
                with self._p("save.filename"):
                    filename = f"frame_{frame_index:06d}_group_{group_index:03d}_{self._image_save_counter:06d}.png"
                    filepath = os.path.join(self.save_images_dir, filename)
                
                # Save image
                with self._p("save.disk_write"):
                    img_resized.save(filepath, quality=95)
                    self._image_save_counter += 1
                    
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")
            return ""
            
        return filepath

    # -----------------------------
    # Main update
    # -----------------------------
    def update(self, adcData, frame_index_val, groupIdx):
        with self._p("update.total"):
            self._ensure_ui()

            self._update_counter += 1
            should_draw = self.enable_plot and ((self._update_counter % self.draw_every_n) == 0)

            # --- reshape/select ---
            with self._p("prep.shape_reshape"):
                Ns, Nl, Nrx, Nc = adcData.shape
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

            # --- CWT ---
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
                P_db =Wx_plot

            # -------------------------------------------------------
            # KEY FIX: resample onto a uniform linear frequency grid
            # so that ImageItem.setRect() maps pixels correctly to Hz.
            # Without this, CWT's log-spaced rows appear linearly
            # stretched — low freqs look compressed, high freqs expanded.
            # -------------------------------------------------------
            P_db_uniform, f_uniform = self._resample_to_uniform_freq(P_db, f_plot)

            with self._p("post.percentiles"):
                # Use fixed global power scale if provided, otherwise calculate from current frame
                if self.power_vmin is not None and self.power_vmax is not None:
                    vmin = float(self.power_vmin)
                    vmax = float(self.power_vmax)
                else:
                    vmin = float(np.nanpercentile(P_db_uniform, 5))
                    vmax = float(np.nanpercentile(P_db_uniform, 99.5))
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin = float(np.nanmin(P_db_uniform))
                        vmax = float(np.nanmax(P_db_uniform))
                        if not np.isfinite(vmin) or not np.isfinite(vmax):
                            return
                        if vmax <= vmin:
                            vmax = vmin + 1.0

            # --- plot (PyQtGraph ImageItem) ---
            if should_draw:
                with self._p("plot.imageitem_update"):
                    t0_val = float(t[0])
                    t1_val = float(t[-1])
                    f0_val = float(f_uniform[0])
                    f1_val = float(f_uniform[-1])

                    if t1_val <= t0_val:
                        t1_val = t0_val + 1e-9
                    if f1_val <= f0_val:
                        f1_val = f0_val + 1e-9

                    # P_db_uniform is [n_freq_bins, n_time] in row-major:
                    # row 0 = lowest freq, last row = highest freq → correct orientation
                    self.img.setImage(P_db_uniform, autoLevels=False)

                    # Now pixels are uniformly spaced in both time and freq,
                    # so this linear rect mapping is exact.
                    rect = QtCore.QRectF(
                        t0_val,
                        f0_val,
                        (t1_val - t0_val),
                        (f1_val - f0_val),
                    )
                    self.img.setRect(rect)
                    self.img.setLevels((vmin, vmax))

                    self.plot.setTitle(
                        f"CWT Magnitude | Group {groupIdx} | Frame {frame_index_val}"
                    )
                    self.plot.setXRange(t0_val, t1_val, padding=0.0)
                    self.plot.setYRange(f0_val, f1_val, padding=0.0)

                with self._p("plot.process_events"):
                    app = QtWidgets.QApplication.instance()
                    if app is not None:
                        app.processEvents()

            # --- save image ---
            if self.save_images_dir:
                self._save_heatmap_image(P_db_uniform, vmin, vmax, frame_index_val, groupIdx)
