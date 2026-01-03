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
        return int(h32[3])


# =================================================
# READ ONE BIN FILE (ONE DEVICE, ONE FRAME, 4 RX)
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

    raw = raw.astype(np.int32)
    raw[raw >= 2**15] -= 2**16

    iq = raw[0::2] + 1j * raw[1::2]

    iq = iq.reshape(
        numRXPerDevice,
        numSamplePerChirp,
        numChirpPerLoop,
        numLoops,
        order="F",
    )

    iq = np.transpose(iq, (1, 3, 0, 2))
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
        )
        for name in names
    ]

    radar_cube = np.zeros((Ns, numLoops, 16, numChirpPerLoop), dtype=np.complex64)
    radar_cube[:, :, 0:4, :] = cubes[0]
    radar_cube[:, :, 4:8, :] = cubes[1]
    radar_cube[:, :, 8:12, :] = cubes[2]
    radar_cube[:, :, 12:16, :] = cubes[3]

    return radar_cube


# =================================================
# RANGE FFT
# =================================================
def range_processing(radar_cube, fft_size):
    Ns, Nl, Nrx, Nc = radar_cube.shape
    win = np.hanning(Ns).astype(np.float32)

    slow_time = Nl * Nc
    out = np.zeros((fft_size, slow_time, Nrx), dtype=np.complex64)

    for rx in range(Nrx):
        mat = radar_cube[:, :, rx, :].reshape(Ns, slow_time, order="F")
        mat -= np.mean(mat, axis=0, keepdims=True)
        mat *= win[:, None]
        out[:, :, rx] = np.fft.fft(mat, fft_size, axis=0)

    return out


# =================================================
# DOPPLER FFT
# =================================================
def doppler_processing(range_fft, numLoops, numChirpPerLoop):
    Nr, slow_time, Nrx = range_fft.shape

    rd = range_fft.reshape(
        Nr, numLoops, numChirpPerLoop, Nrx, order="F"
    )

    rd = np.mean(rd, axis=2)

    win = np.hanning(numLoops).astype(np.float32)
    out = np.zeros_like(rd)

    for r in range(Nr):
        for rx in range(Nrx):
            x = rd[r, :, rx]
            x -= np.mean(x)
            x *= win
            out[r, :, rx] = np.fft.fftshift(np.fft.fft(x))

    return out


# =================================================
# STATIC CLUTTER REMOVAL
# =================================================
def remove_static_clutter(rd_cube):
    zero_doppler = rd_cube.shape[1] // 2
    rd_cube[:, zero_doppler, :] = 0
    return rd_cube


# =================================================
# 2D CA-CFAR
# =================================================
def ca_cfar_2d(
    rd_map,
    train_r=8,
    train_d=4,
    guard_r=2,
    guard_d=1,
    pfa=1e-5,
):
    Nr, Nd = rd_map.shape
    detections = np.zeros((Nr, Nd), dtype=bool)

    num_train = (
        (2 * (train_r + guard_r) + 1) *
        (2 * (train_d + guard_d) + 1)
        - (2 * guard_r + 1) * (2 * guard_d + 1)
    )

    alpha = num_train * (pfa ** (-1 / num_train) - 1)

    for r in range(train_r + guard_r, Nr - train_r - guard_r):
        for d in range(train_d + guard_d, Nd - train_d - guard_d):
            noise = 0.0
            for rr in range(r - train_r - guard_r, r + train_r + guard_r + 1):
                for dd in range(d - train_d - guard_d, d + train_d + guard_d + 1):
                    if abs(rr - r) <= guard_r and abs(dd - d) <= guard_d:
                        continue
                    noise += rd_map[rr, dd]

            noise /= num_train
            if rd_map[r, d] > alpha * noise:
                detections[r, d] = True

    return detections


# =================================================
# FINAL DETECTION + RANGE PRINT
# =================================================
def detect_and_print_range(range_fft, numLoops, numChirpPerLoop, fs, slope_MHz_us):
    c = 3e8
    Nfft = range_fft.shape[0]
    slope = slope_MHz_us * 1e12
    range_bin = (c * fs) / (2 * slope * Nfft)

    rd = doppler_processing(range_fft, numLoops, numChirpPerLoop)
    rd = remove_static_clutter(rd)

    rd_power = np.sum(np.abs(rd) ** 2, axis=2)
    rd_power = rd_power[: Nfft // 2, :]

    detections = ca_cfar_2d(rd_power)
    dets = np.argwhere(detections)

    if len(dets) == 0:
        print("No detections")
        return

    r_idx, d_idx = dets[np.argmax([rd_power[r, d] for r, d in dets])]
    print(f"Detected Range = {(r_idx + 1) * range_bin:.3f} m")


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
):
    processed = set()

    while True:
        files = os.listdir(data_folder)
        idx_files = sorted(f for f in files if re.match(r"master_\d{4}_idx\.bin", f))

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

            nframes = get_valid_num_frames(os.path.join(data_folder, idx_file))
            print(f"\n📥 Capture {idx} | Frames = {nframes}")

            for frame in range(2, nframes + 1):
                cube = read_adc_bin_tda2_separate_files(
                    data_folder, idx, frame, Ns, numChirpPerLoop, numLoops
                )
                rng_fft = range_processing(cube, Ns)
                print(f"[{idx} | Frame {frame}] ", end="")
                detect_and_print_range(
                    rng_fft, numLoops, numChirpPerLoop, fs, slope_MHz_us
                )

            processed.add(idx)
            print(f"✅ Capture {idx} done")

        time.sleep(2)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    DATA_FOLDER = r"C:\radar_receiver\radar_new"

    Ns = 256
    numChirpPerLoop = 12
    numLoops = 128

    fs = 10e6
    slope_MHz_us = 40.024

    monitor_directory(
        DATA_FOLDER,
        Ns,
        numChirpPerLoop,
        numLoops,
        fs,
        slope_MHz_us,
    )
