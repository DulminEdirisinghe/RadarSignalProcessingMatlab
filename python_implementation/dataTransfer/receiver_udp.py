#!/usr/bin/env python3
import os
import socket
import struct
from datetime import datetime

LISTEN_IP = "192.168.33.30"
LISTEN_PORT = 9999
SAVE_DIR = r"C:\radar_receiver\radar_new"

# Packet types (must match sender)
PKT_START = 1
PKT_DATA  = 2
PKT_END   = 3

BUF = 65536  # UDP receive buffer (64 KB)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log(f"Save directory: {SAVE_DIR}")

def main():
    ensure_dir()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))

    log(f"UDP listening on {LISTEN_IP}:{LISTEN_PORT}")

    files = {}  # file_id -> dict with state

    while True:
        data, addr = sock.recvfrom(BUF)

        pkt_type = data[0]

        # ---------------- FILE START ----------------
        if pkt_type == PKT_START:
            _, file_id, name_len = struct.unpack("!B I H", data[:7])
            name = data[7:7 + name_len].decode("utf-8", "replace")
            name = os.path.basename(name)

            tmp_path = os.path.join(SAVE_DIR, name + ".partial")

            f = open(tmp_path, "wb", buffering=0)
            files[file_id] = {
                "file": f,
                "path": tmp_path,
                "name": name,
                "received": 0,
                "last_seq": -1
            }

            log(f"START {name} from {addr[0]}")

        # ---------------- FILE DATA ----------------
        elif pkt_type == PKT_DATA:
            _, file_id, seq = struct.unpack("!B I I", data[:9])
            payload = data[9:]

            if file_id not in files:
                continue  # late or unknown packet

            f = files[file_id]["file"]

            # NOTE: assumes packets arrive in order
            f.write(payload)
            files[file_id]["received"] += len(payload)
            files[file_id]["last_seq"] = seq

        # ---------------- FILE END ----------------
        elif pkt_type == PKT_END:
            _, file_id, seq, expected_size = struct.unpack("!B I I Q", data[:17])

            if file_id not in files:
                continue

            info = files[file_id]
            f = info["file"]
            f.close()

            final_path = os.path.join(SAVE_DIR, info["name"])

            if info["received"] == expected_size:
                os.replace(info["path"], final_path)
                log(f"SAVED {info['name']} ({expected_size:,} bytes)")
            else:
                log(
                    f"SIZE MISMATCH for {info['name']} "
                    f"(got {info['received']}, expected {expected_size})"
                )
                os.remove(info["path"])

            del files[file_id]

        # ---------------- UNKNOWN ----------------
        else:
            log("Unknown packet type")

if __name__ == "__main__":
    main()
