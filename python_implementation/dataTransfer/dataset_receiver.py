#!/usr/bin/env python3
import os
import socket
import struct
from datetime import datetime

LISTEN_IP = "192.168.33.30"
LISTEN_PORT = 9999
SAVE_DIR = r"D:\MLDataset-RAW\New folder"
BUF = 1024 * 1024

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return bytes(data)

def ensure_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)
    log(f"Save directory: {SAVE_DIR}")

def safe_relpath(name):
    name = name.replace("\\", "/")
    norm = os.path.normpath(name)

    if os.path.isabs(norm) or norm.startswith("..") or ".." in norm.split(os.sep):
        raise ValueError(f"unsafe path received: {name!r}")

    return norm

def handle_client(c):
    while True:
        hdr = c.recv(4)
        if not hdr:
            return

        if len(hdr) < 4:
            hdr += recvall(c, 4 - len(hdr))

        (name_len,) = struct.unpack("!I", hdr)
        raw_name = recvall(c, name_len).decode("utf-8", "replace")
        (fsize,) = struct.unpack("!Q", recvall(c, 8))

        rel_path = safe_relpath(raw_name)
        final_path = os.path.join(SAVE_DIR, rel_path)
        tmp_path = final_path + ".partial"

        parent_dir = os.path.dirname(final_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Tell sender whether file is needed
        if os.path.exists(final_path) and os.path.getsize(final_path) == fsize:
            log(f"Skipping {rel_path} (already exists)")
            c.sendall(b"SKIP\n")
            continue
        else:
            c.sendall(b"SEND\n")

        log(f"Receiving {rel_path} ({fsize:,} bytes)")
        remaining = fsize

        with open(tmp_path, "wb", buffering=0) as f:
            while remaining:
                chunk = c.recv(min(BUF, remaining))
                if not chunk:
                    raise ConnectionError("socket closed mid-file")
                f.write(chunk)
                remaining -= len(chunk)

        os.replace(tmp_path, final_path)
        log(f"Saved {rel_path}")
        c.sendall(b"OK\n")

def main():
    ensure_dir()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((LISTEN_IP, LISTEN_PORT))
    s.listen(5)
    log(f"Listening on {LISTEN_IP}:{LISTEN_PORT}")

    while True:
        c, addr = s.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except Exception:
            pass

        log(f"Connection from {addr[0]}:{addr[1]}")
        try:
            handle_client(c)
        except Exception as e:
            log(f"Client error: {e}")
        finally:
            c.close()
            log("Connection closed\n")

if __name__ == "__main__":
    main()