import numpy as np
import os
import re
import matplotlib
matplotlib.use("TkAgg", force=True)
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
# READ BIN FILE (4 RX)
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
# READ CASCADE (16 RX)
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

    cubes = [read_bin_file(os.path.join(folder, n), frame, Ns, Nc, Nl) for n in names]

    _cube_buf[:, :, 0:4, :]   = cubes[0]
    _cube_buf[:, :, 4:8, :]   = cubes[1]
    _cube_buf[:, :, 8:12, :]  = cubes[2]
    _cube_buf[:, :, 12:16, :] = cubes[3]
    return _cube_buf

# -------------------------------------------------
# LIVE WAVELET PLOTTER
# -------------------------------------------------
class LiveMatlabStyleWavelet:
    def __init__(
        self,
        Fs=8e6,
        group_size=128,
        antenna_index_1based=1,
        wavelet="cmor1.0-1.5",
        fmin_hz=1e3,
        fmax_hz=5e6,
        voices_per_octave=12,
        contour_levels=10,
        enable_contours=True,
    ):
        self.Fs = Fs
        self.group_size = group_size
        self.ant0 = int(antenna_index_1based - 1)

        self.wavelet = wavelet
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        self.voices_per_octave = voices_per_octave

        self.contour_levels = contour_levels
        self.enable_contours = enable_contours

        self.fig = None
        self.ax = None
        self.mesh = None
        self.cont = None

        sampling_period = 1.0 / self.Fs
        cf = pywt.central_frequency(self.wavelet)

        n_octaves = np.log2(self.fmax_hz / self.fmin_hz)
        n_freqs = int(np.ceil(n_octaves * self.voices_per_octave)) + 1
        freqs = self.fmax_hz / (2 ** (np.arange(n_freqs) / self.voices_per_octave))
        freqs = freqs[::-1]  # low->high

        self.scales = cf / (freqs * sampling_period)
        self.sampling_period = sampling_period

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.set_cmap("jet")
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

        t = (np.arange(x.size, dtype=np.float64) / self.Fs)

        cfs_r, f_out = pywt.cwt(
            np.real(x), self.scales, self.wavelet,
            sampling_period=self.sampling_period, method="fft"
        )
        cfs_i, _ = pywt.cwt(
            np.imag(x), self.scales, self.wavelet,
            sampling_period=self.sampling_period, method="fft"
        )
        cfs = cfs_r + 1j * cfs_i

        P = (np.abs(cfs) ** 2).astype(np.float64)
        f = f_out.astype(np.float64)

        if f[0] > f[-1]:
            f = f[::-1]
            P = P[::-1, :]

        # auto-limits
        self.ax.set_xlim(float(t[0]), float(t[-1]))
        self.ax.set_ylim(float(f[0]), float(f[-1]))

        # --- POWER -> dB (much better contrast) ---
        P_db = 10.0 * np.log10(P + 1e-12)

        
        # --- Robust color limits (ignore extreme outliers) ---
        vmin = float(np.nanpercentile(P_db, 5))
        vmax = float(np.nanpercentile(P_db, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(P_db))
            vmax = float(np.nanmax(P_db))
            if vmax <= vmin:
                vmax = vmin + 1.0

        # RECOMMENDED: shading="auto" fixes your exact error
        # plot (use P_db now)
        if self.mesh is None:
            self.mesh = self.ax.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")
            # ✅ ADD COLORBAR ONCE (first time only)
            if not hasattr(self, "cbar"):
                self.cbar = self.fig.colorbar(self.mesh, ax=self.ax)
                self.cbar.set_label("Power (dB)")
        else:
            self.mesh.remove()
            self.mesh = self.ax.pcolormesh(t, f, P_db, shading="auto", cmap="jet")
            self.mesh.set_edgecolor("none")

        # ✅ UPDATE COLORBAR TO NEW MESH
        if hasattr(self, "cbar"):
            self.cbar.update_normal(self.mesh)

        self.mesh.set_clim(vmin, vmax)
        self.ax.set_title(f"CWT Power (Group {groupIdx})")

        self._remove_contours()
        if self.enable_contours:
            self.cont = self.ax.contour(
                t, f, P,
                levels=self.contour_levels,
                linewidths=1.0,
                colors="k",
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    DATA_FOLDER = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\config_test_high_bandwidth_comparison"

    Ns = 256
    Nc = 12
    Nl = 128
    fs = 8e6

    GROUP_SIZE = 128
    SHOW_GROUP = 1

    idx_pat = re.compile(r"master_(\d{4})_idx\.bin$")

    live = LiveMatlabStyleWavelet(
        Fs=fs,
        group_size=GROUP_SIZE,
        antenna_index_1based=1,
        wavelet="cmor1.0-1.5",
        fmin_hz=1e3,
        fmax_hz=4e6,
        voices_per_octave=12,
        contour_levels=10,
        enable_contours=False,
    )

    live._ensure_fig()
    print("Backend:", matplotlib.get_backend())
    print("Figure nums:", plt.get_fignums())

    idx_files = []
    for entry in os.scandir(DATA_FOLDER):
        if entry.is_file():
            m = idx_pat.match(entry.name)
            if m:
                idx_files.append((m.group(1), entry.name))
    idx_files.sort(key=lambda x: x[0])

    for idx, idxf in idx_files:
        nframes = get_valid_num_frames(os.path.join(DATA_FOLDER, idxf))
        print(f"\n📥 Capture {idx} | Frames = {nframes}")

        for frame in range(1, nframes + 1):
            cube = read_adc_bin_tda2_separate_files(DATA_FOLDER, idx, frame, Ns, Nc, Nl)

            numGroups = int(np.ceil(Nl / GROUP_SIZE))
            group_to_show = max(1, min(SHOW_GROUP, numGroups))

            print(f"[{idx} | Frame {frame}] plotting wavelets... (Group {group_to_show}/{numGroups})")
            live.update(cube, frame_index_val=frame, groupIdx=group_to_show)

        print(f"✅ Capture {idx} done")

    print("\n✅ Done. Keeping plot window open...")
    plt.show()
