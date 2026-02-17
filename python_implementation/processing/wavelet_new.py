import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt
import pywt

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


import numpy as np
import os
import matplotlib.pyplot as plt
import pywt

def plot_wavelet_for_chirps(
    adcData: np.ndarray,
    outputDir: str,
    frame_index_val: int,
    Fs: float = 8e6,
    group_size: int = 128,
    antenna_index_1based: int = 1,
    fmax_MHz: float = 0.2,
    xlim_us: float = 2010.0,
    voices_per_octave: int = 64,     # MATLAB-like
    fmin_hz: float = 1e3,            # avoid 0 Hz
    wavelet: str = "cmor1.0-1.5",    # good starting match for MATLAB 'amor'
    contour_levels: int = 10,
):
    os.makedirs(outputDir, exist_ok=True)

    Ns, Nl, Nrx, Nc = adcData.shape

    # --- match MATLAB reshape/linearization (column-major) ---
    adcF = np.asfortranarray(adcData)
    adc3 = adcF.reshape((Ns, Nl, Nrx * Nc), order="F")  # (Ns, loops, rx*chirps)

    number_of_loops = adc3.shape[1]
    numGroups = int(np.ceil(number_of_loops / group_size))

    ant0 = int(antenna_index_1based - 1)

    # --- MATLAB-like (log) frequency grid using voices/octave ---
    fmax_hz = fmax_MHz * 1e6
    n_octaves = np.log2(fmax_hz / fmin_hz)
    n_freqs = int(np.ceil(n_octaves * voices_per_octave)) + 1
    freqs_hz = fmax_hz / (2 ** (np.arange(n_freqs) / voices_per_octave))
    freqs_hz = freqs_hz[::-1]  # low -> high for plotting

    sampling_period = 1.0 / Fs
    cf = pywt.central_frequency(wavelet)
    scales = cf / (freqs_hz * sampling_period)

    for groupIdx in range(1, numGroups + 1):
        startChirp = (groupIdx - 1) * group_size
        endChirp = min(groupIdx * group_size, number_of_loops)

        selectedData = adc3[:, startChirp:endChirp, ant0]  # (Ns, groupLen)
        x = selectedData.reshape(-1, order="F").astype(np.complex64)

        t_us = (np.arange(x.size, dtype=np.float64) / Fs) * 1e6

        # --- MATLAB-style: CWT(real) and CWT(imag), then recombine ---
        cfs_r, f_out = pywt.cwt(np.real(x), scales, wavelet, sampling_period=sampling_period, method="fft")
        cfs_i, _     = pywt.cwt(np.imag(x), scales, wavelet, sampling_period=sampling_period, method="fft")
        cfs = cfs_r + 1j * cfs_i

        S = np.abs(cfs) ** 2
        f_MHz = (f_out / 1e6)

        # pywt sometimes returns f descending depending on inputs; enforce ascending
        if f_MHz[0] > f_MHz[-1]:
            f_MHz = f_MHz[::-1]
            S = S[::-1, :]

        # --- MATLAB-like pcolor ---
        fig, ax = plt.subplots(figsize=(12, 6))
        vmax = np.percentile(S, 95)  # clip at 95th percentile to see low values better
        pcm = ax.pcolormesh(t_us, f_MHz, S, shading="auto", cmap="jet", vmin=0, vmax=vmax)
        fig.colorbar(pcm, ax=ax)

        ax.set_xlabel("Time (μs)", fontweight="bold")
        ax.set_ylabel("Frequency (MHz)", fontweight="bold")
        ax.set_title(f"Wavelet Scalogram - Frame {frame_index_val}, Group {groupIdx}",
                     fontweight="bold")

        ax.set_ylim(0, fmax_MHz)
        ax.set_xlim(0, min(xlim_us, t_us[-1]))
        ax.grid(True, alpha=0.3)

        # contour overlay like MATLAB
        ax.contour(t_us, f_MHz, S, levels=contour_levels, linewidths=1.0, colors="k")

        out_path = os.path.join(
            outputDir,
            f"frame{frame_index_val}_wavelet_group{groupIdx}_magnitude_paper.png"
        )
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Wavelet plots generated (MATLAB-like).")

# =================================================
# MAIN (WAVELET ONLY)
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\config_test_low_bandwidth_comparison"
    OUT_FOLDER  = os.path.join(DATA_FOLDER, "wavelets_out")

    Ns = 256
    Nc = 12
    Nl = 128

    fs = 8e6

    # Process settings
    PLOT_EVERY_N_FRAMES = 1  # change if you want to skip frames
    GROUP_SIZE = 128         # same as your MATLAB (rx_channels variable used as group size)

    processed = set()
    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    frame_counter = 0

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
                    cube = read_adc_bin_tda2_separate_files(
                        DATA_FOLDER, idx, frame, Ns, Nc, Nl
                    )

                    frame_counter += 1
                    if (frame_counter % PLOT_EVERY_N_FRAMES) == 0:
                        print(f"[{idx} | Frame {frame}] plotting wavelets...")
                        frame_out = os.path.join(OUT_FOLDER, f"cap_{idx}", f"frame_{frame:04d}")
                        plot_wavelet_for_chirps(
                            adcData=cube,
                            outputDir=frame_out,
                            frame_index_val=frame,
                            Fs=fs,
                            group_size=GROUP_SIZE,
                            antenna_index_1based=1,
                            fmax_MHz=2,
                            xlim_us=2010.0,
                            voices_per_octave=12,
                            wavelet="cmor",
                        )

                processed.add(idx)
                print(f"✅ Capture {idx} done")

        except Exception as e:
            print("⚠️ Error:", e)

        time.sleep(0.05)
