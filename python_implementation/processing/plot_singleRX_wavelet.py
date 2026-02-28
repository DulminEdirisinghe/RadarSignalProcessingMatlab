import torch
import os
os.environ["SSQ_GPU"] = "0"   # CWT on CPU; GPU used only by Qt for rendering
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
    DATA_FOLDER = r"D:\radardata_fmcw\phantom_2m_128frame022326"
    GROUP_SIZE = 128
    Ns = ADC_SAMPLES
    Nc = NC_CHIRPS_PER_LOOP
    Nl = NCHIRP_LOOPS
    FAST_CWT_TIME_RANGE_SEC = (0.0, 0.0002)

    # Profile only a few frames, then exit
    MAX_PROFILE_FRAMES = 127

    # Draw every N frames (set to 2 or 5 for faster overall performance)
    DRAW_EVERY_N = 1

    # Number of uniform frequency bins for display.
    # Higher = more freq resolution in the plot (but slightly more RAM / resample time).
    # 512 is a good default; 256 is faster; 1024 gives finer detail.
    N_FREQ_BINS = 512

    prof = TimeProfiler("MainLoopProfiler-PyQtGraph")

    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")
    processed = set()

    # Ensure a Qt app exists (PyQtGraph UI)
    app = pg.mkQApp("Wavelet Viewer")

    live_fast_cwt = LiveFastWaveletPG(
        Fs=FS_FAST,
        antenna_index_1based=1,
        fmin_hz=1e3,
        fmax_hz=4e6,
        voices_per_octave=2048,
        n_freq_bins=N_FREQ_BINS,      # uniform freq grid size for correct axis display
        contour_levels=10,
        enable_contours=False,
        dc_remove_mode="mean",
        hp_win=1024,
        time_range_sec=FAST_CWT_TIME_RANGE_SEC,
        time_stride=1,
        enable_profiling=True,
        enable_plot=True,
        draw_every_n=DRAW_EVERY_N,
    )

    total_frames_processed = 0
    done = False

    try:
        while not done:
            try:
                with prof.section("loop.scan_folder"):
                    idx_files = []
                    for entry in os.scandir(DATA_FOLDER):
                        if entry.is_file():
                            m = idx_pat.match(entry.name)
                            if m:
                                idx_files.append((m.group(1), entry.name))

                with prof.section("loop.sort_idx_files"):
                    idx_files.sort(key=lambda x: x[0])

                if not idx_files:
                    print("No capture files found.")
                    break

                for idx, idxf in idx_files:
                    if idx in processed:
                        continue

                    with prof.section("file.get_valid_num_frames"):
                        nframes = get_valid_num_frames(os.path.join(DATA_FOLDER, idxf))

                    print(f"\n📥 Capture {idx} | Frames = {nframes}")

                    for frame in range(1, nframes + 1):
                        with prof.section("frame.total"):
                            with prof.section("frame.read_master_bin"):
                                cube = read_master_bin(DATA_FOLDER, idx, frame, Ns, Nc, Nl)

                            with prof.section("frame.group_calc"):
                                numGroups = int(np.ceil(Nl / GROUP_SIZE))
                                group_to_show = max(1, min(1, numGroups))

                            print(f"[{idx} | Frame {frame}] plots... (Group {group_to_show}/{numGroups})")
                            print(cube.shape)

                            with prof.section("frame.wavelet_update"):
                                live_fast_cwt.update(cube, frame_index_val=frame, groupIdx=group_to_show)

                            with prof.section("frame.qt_process_events"):
                                app.processEvents()

                        total_frames_processed += 1

                        if total_frames_processed >= MAX_PROFILE_FRAMES:
                            done = True
                            break

                    with prof.section("loop.mark_processed"):
                        processed.add(idx)

                    print(f" Capture {idx} done")

                    if done:
                        break

                if not done:
                    print("Finished available captures before reaching frame limit.")
                    break

            except Exception as e:
                print("Error:", e)
                break

            with prof.section("loop.sleep"):
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")

    finally:
        print_combined_summary(prof, live_fast_cwt)
        if live_fast_cwt.enable_plot and live_fast_cwt.win is not None:
            print("Close the plot window to exit...")
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.exec()
