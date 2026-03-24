import torch
import os
os.environ["SSQ_GPU"] = "0"  
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from fmcw_parameters import *
from fast_time_wavelet import LiveFastWaveletPG
from helpers import *


# ============================================================
# Profiling utility
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

        print(f"\n{'=' * 100}")
        print(f"{self.name} - Timing Summary")
        print(f"{'-' * 100}")
        print(f"{'Section':40s} {'Calls':>8s} {'Total(s)':>12s} {'Avg(ms)':>12s} {'Min(ms)':>12s} {'Max(ms)':>12s} {'%':>8s}")
        print(f"{'-' * 100}")
        for k, s in items:
            avg_ms = (s.total / s.count) * 1e3 if s.count else 0.0
            min_ms = s.min_t * 1e3 if s.count else 0.0
            max_ms = s.max_t * 1e3 if s.count else 0.0
            pct = (100.0 * s.total / total_all) if total_all > 0 else 0.0
            print(f"{k:40s} {s.count:8d} {s.total:12.6f} {avg_ms:12.3f} {min_ms:12.3f} {max_ms:12.3f} {pct:8.2f}")
        print(f"{'-' * 100}")
        print(f"{'TOTAL':40s} {'':8s} {total_all:12.6f}")
        print(f"{'=' * 100}\n")


def print_combined_summary(main_prof: TimeProfiler, wavelet_obj: LiveFastWaveletPG):
    print("\n\n########## DESCRIPTIVE PROFILING SUMMARY ##########")
    main_prof.print_summary()
    wavelet_obj.print_profile_summary()


if __name__ == "__main__":
    # -------------------------
    # User settings
    # -------------------------
    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\phantom_2m_128_03242024_3"
    GROUP_SIZE = 128
    Ns = ADC_SAMPLES
    Nc = NC_CHIRPS_PER_LOOP
    Nl = NCHIRP_LOOPS
    FAST_CWT_TIME_RANGE_SEC = (0.0, 0.0002)

    # Set to None to run forever, or an int to stop after N total frames
    MAX_PROFILE_FRAMES = None

    # Set to True to print profiling summary on exit
    ENABLE_PROFILING = False

    DRAW_EVERY_N = 1
    N_FREQ_BINS = 512
    
    # ML Dataset configuration
    SAVE_IMAGES = True  # Set to False to disable image saving
    IMAGES_OUTPUT_DIR = os.path.join(r"C:\radar_receiver\ml_dataset", os.path.basename(DATA_FOLDER))  # Directory to save standardized images
    IMAGE_OUTPUT_SIZE = (256, 256)  # Standardized output image size (height, width)

    # Polling interval (seconds) when waiting for new files or new frames
    POLL_INTERVAL_SEC = 1.0

    prof = TimeProfiler("MainLoopProfiler-PyQtGraph")

    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    # frames_done[idx_str] = number of frames already processed for that capture
    frames_done: Dict[str, int] = {}

    app = pg.mkQApp("Wavelet Viewer")

    live_fast_cwt = LiveFastWaveletPG(
        Fs=FS_FAST,
        antenna_index_1based=1,
        fmin_hz=1e3,
        fmax_hz=3e6,
        n_freq_bins=N_FREQ_BINS,
        contour_levels=10,
        enable_contours=False,
        dc_remove_mode="mean",
        hp_win=1024,
        time_range_sec=FAST_CWT_TIME_RANGE_SEC,
        time_stride=1,
        enable_profiling=ENABLE_PROFILING,
        enable_plot=True,
        draw_every_n=DRAW_EVERY_N,
        save_images_dir=IMAGES_OUTPUT_DIR if SAVE_IMAGES else None,
        output_size=IMAGE_OUTPUT_SIZE,
        power_vmin=None,#0,              # Fixed global minimum power
        power_vmax=None, #1e4,              # Fixed global maximum power (adjust as needed)
    )

    total_frames_processed = 0

    print(f"Watching {DATA_FOLDER} for captures... (Ctrl+C to stop)")

    try:
        while True:

            # ── 1. Scan folder for all idx files ──────────────────────
            try:
                idx_files = []
                for entry in os.scandir(DATA_FOLDER):
                    if entry.is_file():
                        m = idx_pat.match(entry.name)
                        if m:
                            idx_files.append((m.group(1), entry.name))
                idx_files.sort(key=lambda x: x[0])
            except Exception as e:
                print(f"[scan error] {e}")
                time.sleep(POLL_INTERVAL_SEC)
                app.processEvents()
                continue

            if not idx_files:
                # No captures yet — keep polling silently
                time.sleep(POLL_INTERVAL_SEC)
                app.processEvents()
                continue

            # ── 2. For each capture, process any frames not yet seen ──
            made_progress = False

            for idx, idxf in idx_files:
                idx_path = os.path.join(DATA_FOLDER, idxf)

                try:
                    nframes = get_valid_num_frames(idx_path)
                except Exception as e:
                    print(f"[read error {idxf}] {e}")
                    continue

                already_done = frames_done.get(idx, 0)

                if nframes <= already_done:
                    # No new frames for this capture yet
                    continue

                print(f"\n Capture {idx} | frames available = {nframes} | resuming from frame {already_done + 1}")

                for frame in range(already_done + 1, nframes + 1):
                    try:
                        cube = read_master_bin(DATA_FOLDER, idx, frame, Ns, Nc, Nl)

                        numGroups = int(np.ceil(Nl / GROUP_SIZE))
                        group_to_show = max(1, min(1, numGroups))

                        print(f"  [{idx} | Frame {frame}/{nframes}] shape {cube.shape}")

                        live_fast_cwt.update(cube, frame_index_val=frame, groupIdx=group_to_show)
                        app.processEvents()

                        frames_done[idx] = frame
                        total_frames_processed += 1
                        made_progress = True

                    except Exception as e:
                        print(f"  [frame error idx={idx} frame={frame}] {e}")
                        continue

                    if MAX_PROFILE_FRAMES is not None and total_frames_processed >= MAX_PROFILE_FRAMES:
                        print(f"\n Reached MAX_PROFILE_FRAMES={MAX_PROFILE_FRAMES}, stopping.")
                        raise KeyboardInterrupt

                print(f"Capture {idx}: processed up to frame {frames_done[idx]}/{nframes}")

            # ── 3. Nothing new this scan → sleep and poll again ────────
            if not made_progress:
                time.sleep(POLL_INTERVAL_SEC)
                app.processEvents()

    except KeyboardInterrupt:
        print("\n Stopped by user (Ctrl+C).")

    finally:
        if ENABLE_PROFILING:
            print_combined_summary(prof, live_fast_cwt)
        if live_fast_cwt.enable_plot and live_fast_cwt.win is not None:
            print("Close the plot window to exit...")
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exec()