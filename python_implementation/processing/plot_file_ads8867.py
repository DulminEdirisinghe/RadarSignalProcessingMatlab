import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--fs", type=float, default=100000.0, help="Assumed sample rate (Hz)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    ap.add_argument("--dur", type=float, default=0.05, help="Duration to plot (seconds)")
    ap.add_argument("--dc_block", action="store_true")
    ap.add_argument("--fft", action="store_true")
    args = ap.parse_args()

    raw = np.fromfile(args.file, dtype="<i2")  # int16 little-endian
    if raw.size == 0:
        raise SystemExit("File is empty or not int16 data.")

    fs = args.fs
    start_idx = int(args.start * fs)
    n = int(args.dur * fs)
    end_idx = min(start_idx + n, raw.size)

    x = raw[start_idx:end_idx].astype(np.float32)
    if x.size < 10:
        raise SystemExit("Selected segment too short. Increase --dur or reduce --start.")

    if args.dc_block:
        x -= x.mean()

    t = np.arange(x.size) / fs + args.start

    # Time plot
    plt.figure()
    plt.plot(t, x)
    plt.title(f"Time domain: {args.file}  (start={args.start}s, dur={args.dur}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("ADC counts")
    plt.grid(True)

    # FFT plot (optional)
    if args.fft:
        nfft = 1 << int(np.ceil(np.log2(x.size)))
        w = np.hanning(x.size)
        X = np.fft.rfft(x * w, n=nfft)
        f = np.fft.rfftfreq(nfft, d=1.0/fs)
        mag = np.abs(X)

        plt.figure()
        plt.plot(f, mag)
        plt.title("FFT magnitude")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|X(f)|")
        plt.grid(True)
        plt.xlim(0, fs/2)

    plt.show()

if __name__ == "__main__":
    main()
