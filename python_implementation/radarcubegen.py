import numpy as np
import os
import time
import re

# =================================================
# IDX FILE READER (TI FORMAT)
# =================================================
def get_valid_num_frames(idx_file_path):
    with open(idx_file_path, "rb") as f:
        h32 = np.fromfile(f, dtype=np.uint32, count=6)
        num_frames = int(h32[3])
    return num_frames


# =================================================
# READ ONE BIN FILE (ONE DEVICE, ONE FRAME, 4 RX)
# EXACT MATLAB readBinFile()
# =================================================
def read_bin_file(
    file_path,
    frame_idx,
    numSamplePerChirp,
    numChirpPerLoop,
    numLoops,
    numRXPerDevice=4,
):
    samples_per_frame = (
        numSamplePerChirp
        * numChirpPerLoop
        * numLoops
        * numRXPerDevice
        * 2
    )

    offset_bytes = (frame_idx - 1) * samples_per_frame * 2

    with open(file_path, "rb") as f:
        f.seek(offset_bytes, os.SEEK_SET)
        raw = np.fromfile(f, dtype=np.uint16, count=samples_per_frame)

    # MATLAB-style signed conversion
    raw = raw.astype(np.int32)
    raw[raw >= 2**15] -= 2**16

    # IQ reconstruction
    iq = raw[0::2] + 1j * raw[1::2]

    # MATLAB:
    # reshape(adcData1, numRX, numSamplePerChirp, numChirpPerLoop, numLoops)
    iq = iq.reshape(
        numRXPerDevice,
        numSamplePerChirp,
        numChirpPerLoop,
        numLoops,
        order="F",
    )

    # MATLAB permute [2 4 1 3]
    iq = np.transpose(iq, (1, 3, 0, 2))

    # (samples, loops, rx=4, chirps)
    return iq


# =================================================
# READ CASCADE FILES → ONE FRAME (16 RX)
# =================================================
def read_adc_bin_tda2_separate_files(
    data_folder,
    idx,
    frame_idx,
    Ns,
    numChirpPerLoop,
    numLoops,
):
    names = [
        f"master_{idx}_data.bin",
        f"slave1_{idx}_data.bin",
        f"slave2_{idx}_data.bin",
        f"slave3_{idx}_data.bin",
    ]

    cubes = [
        read_bin_file(
            os.path.join(data_folder, name),
            frame_idx,
            Ns,
            numChirpPerLoop,
            numLoops,
            numRXPerDevice=4,
        )
        for name in names
    ]

    # Final cube: (samples, loops, rx=16, chirps)
    radar_cube = np.zeros(
        (Ns, numLoops, 16, numChirpPerLoop), dtype=np.complex64
    )

    radar_cube[:, :, 0:4, :] = cubes[0]
    radar_cube[:, :, 4:8, :] = cubes[1]
    radar_cube[:, :, 8:12, :] = cubes[2]
    radar_cube[:, :, 12:16, :] = cubes[3]

    #print("Radar cube shape:", radar_cube.shape)
    #print("RAW ADC RX0 :", radar_cube[0, 0, 0, 0])
    #print("RAW ADC RX4 :", radar_cube[0, 0, 4, 0])
    #print("RAW ADC RX8 :", radar_cube[0, 0, 8, 0])
    #print("RAW ADC RX12:", radar_cube[0, 0, 12, 0])

    return radar_cube


# =================================================
# RANGE FFT (TI rangeProcCascade.m EXACT)
# =================================================
def range_processing_ti(radar_cube, fft_size):
    # radar_cube: (samples, loops, rx, chirps)
    Ns, Nl, Nrx, Nc = radar_cube.shape

    # MATLAB range window (Hanning)
    win = np.hanning(Ns).astype(np.float32)

    slow_time = Nl * Nc
    out = np.zeros((fft_size, slow_time, Nrx), dtype=np.complex64)

    for rx in range(Nrx):
        mat = radar_cube[:, :, rx, :].reshape(Ns, slow_time, order="F")

        # DC removal
        mat -= np.mean(mat, axis=0, keepdims=True)

        # Window
        mat *= win[:, None]

        # FFT
        out[:, :, rx] = np.fft.fft(mat, fft_size, axis=0)

    return out


# =================================================
# RANGE ESTIMATION (EXACT MATLAB LOGIC)
# =================================================
def print_peak_range(range_fft, fs, slope_MHz_us):
    c = 3e8
    Nfft = range_fft.shape[0]

    # Convert slope (MHz/us → Hz/s)
    slope = slope_MHz_us * 1e12

    # EXACT MATLAB rangeBinSize
    rangeBinSize = (c * fs) / (2 * slope * Nfft)

    # Range power profile (integrated over slow-time & RX)
    mag = np.abs(range_fft) ** 2
    profile = np.max(mag, axis=(1, 2))

    # Positive frequencies only
    half = Nfft // 2
    profile = profile[:half]

    # -------------------------------------------------
    # EXACT MATLAB behavior: ignore near-range clutter
    # -------------------------------------------------
    min_range_bin = 5          # MATLAB: minRangeBinKeep
    max_range_bin = half - 1   # keep full usable band

    search_profile = profile[min_range_bin:max_range_bin]

    # Peak detection (relative index)
    k_rel = np.argmax(search_profile)

    # Convert back to absolute FFT bin
    k = k_rel + min_range_bin

    # Optional quadratic interpolation (safe, MATLAB-consistent)
    if 1 <= k < half - 1:
        delta = (profile[k + 1] - profile[k - 1]) / (
            2 * (2 * profile[k] - profile[k + 1] - profile[k - 1])
        )
        k = k + delta

    # MATLAB indexing: range = (rangeInd + 1) * rangeBinSize
    range_m = (k + 1) * rangeBinSize

    print(f"Peak Range = {range_m:.3f} m")

# =================================================
# DIRECTORY MONITOR
# =================================================
def monitor_directory(
    data_folder,
    Ns,
    numChirpPerLoop,
    numLoops,
    fs,
    slope_MHz_us,
    poll_sec=2,
):
    processed = set()
    print("Monitoring:", data_folder)

    while True:
        files = os.listdir(data_folder)
        idx_files = sorted(
            f for f in files if re.match(r"master_\d{4}_idx\.bin", f)
        )

        for idx_file in idx_files:
            idx = re.findall(r"\d{4}", idx_file)[0]
            if idx in processed:
                continue

            needed = [
                f"master_{idx}_data.bin",
                f"slave1_{idx}_data.bin",
                f"slave2_{idx}_data.bin",
                f"slave3_{idx}_data.bin",
                f"master_{idx}_idx.bin",
            ]

            if not all(f in files for f in needed):
                continue

            nframes = get_valid_num_frames(
                os.path.join(data_folder, idx_file)
            )
            print(f"\n📥 Capture {idx} | Frames = {nframes}")

            for frame in range(2, nframes + 1):
                cube = read_adc_bin_tda2_separate_files(
                    data_folder,
                    idx,
                    frame,
                    Ns,
                    numChirpPerLoop,
                    numLoops,
                )

                rng_fft = range_processing_ti(cube, Ns)

                print(f"[{idx} | Frame {frame}] ", end="")
                print_peak_range(rng_fft, fs, slope_MHz_us)

            processed.add(idx)
            print(f"✅ Capture {idx} done")

        time.sleep(poll_sec)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\radar_receiver\radar_new"

    # From mmWave Studio JSON
    Ns = 256
    numChirpPerLoop = 12
    numLoops = 128

    fs = 10e6                          # Hz
    slope_MHz_us = 40.024#78.9857#29.982000350952148 # MHz/us

    monitor_directory(
        DATA_FOLDER,
        Ns,
        numChirpPerLoop,
        numLoops,
        fs,
        slope_MHz_us,
    )
