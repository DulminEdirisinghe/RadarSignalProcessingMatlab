#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import socket
import struct
from datetime import datetime

PC_IP = "192.168.33.30"
PC_PORT = 9999
WATCH_DIR = "/mnt/ssd/13"

BUF = 60 * 1024   # UDP-safe payload (< MTU)
DELETE_AFTER_TRANSFER = True

# --- Stability tuning (UNCHANGED) ---
POLL_SLEEP        = 0.05
CLOSED_TIMEOUT    = 60.0
STABLE_INTERVAL   = 0.05
STABLE_CHECKS     = 5
MIN_MTIME_AGE     = 0.20

PKT_START = 1
PKT_DATA  = 2
PKT_END   = 3

def log(msg):
    print "[%s] %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return s

# -------- file safety logic (UNCHANGED) --------
def is_open_by_any_process(path):
    try:
        real = os.path.realpath(path)
    except:
        return True

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fd_dir = "/proc/%s/fd" % pid
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.path.realpath(os.readlink(fd_dir + "/" + fd)) == real:
                        return True
                except:
                    pass
        except:
            pass
    return False

def wait_until_closed_and_stable(fp):
    start = time.time()

    while is_open_by_any_process(fp):
        if time.time() - start > CLOSED_TIMEOUT:
            return False
        time.sleep(0.02)

    last = (-1, -1)
    stable = 0

    while True:
        st = os.stat(fp)
        cur = (st.st_size, st.st_mtime)

        if cur == last:
            stable += 1
        else:
            stable = 0
            last = cur

        if stable >= STABLE_CHECKS and (time.time() - st.st_mtime) >= MIN_MTIME_AGE:
            return True

        if time.time() - start > CLOSED_TIMEOUT:
            return False

        time.sleep(STABLE_INTERVAL)

# ----------------------------------------------

def list_candidate_files():
    files = glob.glob(WATCH_DIR + "/**/*.bin")
    files.sort(key=lambda x: os.path.getmtime(x))
    return files

def send_one(sock, fp):
    if not wait_until_closed_and_stable(fp):
        log("SKIP: %s" % fp)
        return False

    name = os.path.basename(fp)
    size = os.path.getsize(fp)
    file_id = int(time.time() * 1000) & 0xFFFFFFFF

    # --- FILE START ---
    hdr = struct.pack("!B I H", PKT_START, file_id, len(name)) + name
    sock.sendto(hdr, (PC_IP, PC_PORT))

    seq = 0
    sent = 0

    f = open(fp, "rb")
    try:
        while True:
            data = f.read(BUF)
            if not data:
                break
            pkt = struct.pack("!B I I", PKT_DATA, file_id, seq) + data
            sock.sendto(pkt, (PC_IP, PC_PORT))
            seq += 1
            sent += len(data)
    finally:
        f.close()

    # --- FILE END ---
    end_pkt = struct.pack("!B I I Q", PKT_END, file_id, seq, size)
    sock.sendto(end_pkt, (PC_IP, PC_PORT))

    log("UDP sent %s (%d bytes)" % (name, sent))

    if DELETE_AFTER_TRANSFER:
        os.remove(fp)
        log("DELETED %s" % name)

    return True

def monitor():
    log("UDP streaming to %s:%d" % (PC_IP, PC_PORT))
    sock = connect()
    sent = set()

    while True:
        try:
            for fp in list_candidate_files():
                if fp not in sent:
                    if send_one(sock, fp):
                        sent.add(fp)
            time.sleep(POLL_SLEEP)
        except KeyboardInterrupt:
            log("Stopping")
            return
        except Exception as e:
            log("Error: %s" % e)
            time.sleep(0.5)

if __name__ == "__main__":
    monitor()
