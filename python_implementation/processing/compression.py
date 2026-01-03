# compress_bin_path.py  (Python 2)

import zlib
import os
import time

# -------- CONFIG --------
INPUT_PATH  = r"C:\radar_receiver\radar_new\master_0000_data.bin"
OUTPUT_PATH = r"C:\radar_receiver\radar_new\master_0000_data.bin.z"
COMPRESSION_LEVEL = 1   # low latency
# ------------------------

t_start_total = time.time()

# Read file
with open(INPUT_PATH, "rb") as f:
    data = f.read()

# Compress
t_start_comp = time.time()
compressed = zlib.compress(data, COMPRESSION_LEVEL)
t_end_comp = time.time()

# Write output
with open(OUTPUT_PATH, "wb") as f:
    f.write(compressed)

t_end_total = time.time()

orig_size = os.path.getsize(INPUT_PATH)
comp_size = os.path.getsize(OUTPUT_PATH)

print("Input file       :", INPUT_PATH)
print("Output file      :", OUTPUT_PATH)
print("Original size    :", orig_size, "bytes")
print("Compressed size  :", comp_size, "bytes")
print("Compression      : %.2f%%" % (100.0 * comp_size / orig_size))
print("Compression time : %.3f ms" % ((t_end_comp - t_start_comp) * 1000))
print("Total time       : %.3f ms" % ((t_end_total - t_start_total) * 1000))
