import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt

import re
import time

from fmcw_parameters import *
from fast_time_wavelet import LiveMatlabStyleWavelet
from helpers import *



if __name__ == "__main__":
    DATA_FOLDER = r"C:\radar_receiver\radar_new"
    GROUP_SIZE = 128
    Ns = ADC_SAMPLES
    Nc = NC_CHIRPS_PER_LOOP
    Nl = NCHIRP_LOOPS
    FAST_CWT_TIME_RANGE_SEC = (0.0, 0.0002)
    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")
    processed = set()
    live_fast_cwt = LiveMatlabStyleWavelet(
        Fs=FS_FAST,
        antenna_index_1based=1,
        fmin_hz=1e3,
        fmax_hz=4e6,
        voices_per_octave=2048,
        contour_levels=10,
        enable_contours=False,
        dc_remove_mode="mean",
        hp_win=1024,
        time_range_sec=FAST_CWT_TIME_RANGE_SEC,
        time_stride=1,
    )

    while True:
        try:
            idx_files =[]
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
                    cube = read_master_bin(DATA_FOLDER, idx, frame, Ns, Nc, Nl)

                    numGroups = int(np.ceil(Nl / GROUP_SIZE))
                    group_to_show = max(1, min(1, numGroups))

                    print(f"[{idx} | Frame {frame}] plots... (Group {group_to_show}/{numGroups})")
                    print(cube.shape)
                    live_fast_cwt.update(cube, frame_index_val=frame, groupIdx=group_to_show)
                   
                    # keep UI responsive
                    plt.pause(0.001)

                processed.add(idx)
                print(f" Capture {idx} done")

        except Exception as e:
            print(" Error:", e)

        time.sleep(0.05)
            