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
        self.min_t = min(self.min_t, dt)
        self.max_t = max(self.max_t, dt)


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
        print(
            f"{'Section':40s} {'Calls':>8s} {'Total(s)':>12s} "
            f"{'Avg(ms)':>12s} {'Min(ms)':>12s} {'Max(ms)':>12s} {'%':>8s}"
        )
        print(f"{'-' * 100}")

        for k, s in items:
            avg_ms = (s.total / s.count) * 1e3 if s.count else 0.0
            min_ms = s.min_t * 1e3 if s.count else 0.0
            max_ms = s.max_t * 1e3 if s.count else 0.0
            pct = (100.0 * s.total / total_all) if total_all > 0 else 0.0

            print(
                f"{k:40s} {s.count:8d} {s.total:12.6f} "
                f"{avg_ms:12.3f} {min_ms:12.3f} {max_ms:12.3f} {pct:8.2f}"
            )

        print(f"{'-' * 100}")
        print(f"{'TOTAL':40s} {'':8s} {total_all:12.6f}")
        print(f"{'=' * 100}\n")


def print_combined_summary(main_prof: TimeProfiler, wavelet_obj: LiveFastWaveletPG):
    print("\n\n########## DESCRIPTIVE PROFILING SUMMARY ##########")
    main_prof.print_summary()
    wavelet_obj.print_profile_summary()


if __name__ == "__main__":

    # ============================================================
    # User settings
    # ============================================================

    ROOT_DATA_FOLDER = r"D:\MLDataset-RAW\vimana"
    ROOT_IMAGES_OUTPUT_DIR = r"C:\radar_receiver\ml_dataset"

    GROUP_SIZE = 128
    Ns = ADC_SAMPLES
    Nc = NC_CHIRPS_PER_LOOP
    Nl = NCHIRP_LOOPS

    FAST_CWT_TIME_RANGE_SEC = (0.0, 0.0002)

    MAX_PROFILE_FRAMES = None
    ENABLE_PROFILING = False

    DRAW_EVERY_N = 1
    N_FREQ_BINS = 512

    SAVE_IMAGES = True
    IMAGE_OUTPUT_SIZE = (1024, 1024)

    POLL_INTERVAL_SEC = 1.0

    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    prof = TimeProfiler("MainLoopProfiler-PyQtGraph")

    app = pg.mkQApp("Wavelet Viewer")

    # ============================================================
    # Get all mavic folders
    # ============================================================

    DATA_FOLDERS = [
        os.path.join(ROOT_DATA_FOLDER, d)
        for d in os.listdir(ROOT_DATA_FOLDER)
        if os.path.isdir(os.path.join(ROOT_DATA_FOLDER, d))
        and d.startswith("vimana_")
    ]

    DATA_FOLDERS.sort()

    print("\nFolders found:")
    for folder in DATA_FOLDERS:
        print("  ", folder)

    # ============================================================
    # Process each folder
    # ============================================================

    try:
        for DATA_FOLDER in DATA_FOLDERS:

            folder_name = os.path.basename(DATA_FOLDER)
            IMAGES_OUTPUT_DIR = os.path.join(ROOT_IMAGES_OUTPUT_DIR, folder_name)

            os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Processing folder : {DATA_FOLDER}")
            print(f"Saving images to  : {IMAGES_OUTPUT_DIR}")
            print("=" * 80)

            frames_done: Dict[str, int] = {}
            total_frames_processed = 0

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
                power_vmin=None,
                power_vmax=None,
            )

            while True:

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
                    print(f"No idx files found in {folder_name}. Skipping.")
                    break

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
                        continue

                    print(
                        f"\nCapture {idx} | frames available = {nframes} | "
                        f"resuming from frame {already_done + 1}"
                    )

                    for frame in range(already_done + 1, nframes + 1):

                        try:
                            cube = read_master_bin(
                                DATA_FOLDER,
                                idx,
                                frame,
                                Ns,
                                Nc,
                                Nl
                            )

                            numGroups = int(np.ceil(Nl / GROUP_SIZE))
                            group_to_show = max(1, min(1, numGroups))

                            print(f"  [{idx} | Frame {frame}/{nframes}] shape {cube.shape}")

                            live_fast_cwt.update(
                                cube,
                                frame_index_val=frame,
                                groupIdx=group_to_show
                            )

                            app.processEvents()

                            frames_done[idx] = frame
                            total_frames_processed += 1
                            made_progress = True

                        except Exception as e:
                            print(f"  [frame error idx={idx} frame={frame}] {e}")
                            continue

                        if (
                            MAX_PROFILE_FRAMES is not None
                            and total_frames_processed >= MAX_PROFILE_FRAMES
                        ):
                            print(
                                f"\nReached MAX_PROFILE_FRAMES={MAX_PROFILE_FRAMES}, stopping."
                            )
                            raise KeyboardInterrupt

                    print(
                        f"Capture {idx}: processed up to "
                        f"frame {frames_done[idx]}/{nframes}"
                    )

                if not made_progress:
                    print(f"\nFinished folder: {folder_name}")
                    break

            if ENABLE_PROFILING:
                print_combined_summary(prof, live_fast_cwt)

        print("\nAll folders processed successfully.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        if QtWidgets.QApplication.instance() is not None:
            print("Processing ended.")