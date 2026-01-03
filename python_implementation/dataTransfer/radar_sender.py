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
BUF = 256 * 1024

DELETE_AFTER_TRANSFER = True

# Tuning (safe defaults)
POLL_SLEEP        = 0.05   # 50ms directory poll
CLOSED_TIMEOUT    = 60.0   # max time to wait for writer to close file
STABLE_INTERVAL   = 0.05   # 50ms
STABLE_CHECKS     = 5      # require 5 stable samples after closed
MIN_MTIME_AGE     = 0.20   # 200ms since last modification after closed

def log(msg):
    print "[%s] %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    except:
        pass
    s.settimeout(10)
    s.connect((PC_IP, PC_PORT))
    s.settimeout(None)
    return s

def is_open_by_any_process(path):
    """
    Returns True if ANY process currently has this file open.
    Scans /proc/*/fd -> readlink -> compare to realpath.
    No external packages needed.
    """
    try:
        real = os.path.realpath(path)
    except:
        return True  # if we can't resolve, be conservative

    proc = "/proc"
    try:
        pids = os.listdir(proc)
    except:
        return True

    for pid in pids:
        if not pid.isdigit():
            continue
        fd_dir = os.path.join(proc, pid, "fd")
        try:
            fds = os.listdir(fd_dir)
        except:
            continue  # permission denied or process vanished
        for fd in fds:
            link = os.path.join(fd_dir, fd)
            try:
                target = os.readlink(link)
            except:
                continue
            # quick exact compare
            if target == real:
                return True
            # sometimes target is relative or contains " (deleted)"
            try:
                if os.path.realpath(target) == real:
                    return True
            except:
                pass
    return False

def wait_until_closed_and_stable(filepath,
                                closed_timeout=CLOSED_TIMEOUT,
                                interval=STABLE_INTERVAL,
                                stable_checks=STABLE_CHECKS,
                                min_mtime_age=MIN_MTIME_AGE):
    """
    1) Wait until file is NOT open by any process (writer closed it)
    2) Then require size+mtime stability for stable_checks samples
       and require last mtime to be older than min_mtime_age
    """
    start = time.time()

    # 1) wait until closed
    while True:
        if not os.path.exists(filepath):
            return False
        if not is_open_by_any_process(filepath):
            break
        if (time.time() - start) > closed_timeout:
            return False
        time.sleep(0.02)  # very fast check while open

    # 2) once closed, wait for stable size/mtime
    stable = 0
    last_size = -1
    last_mtime = -1

    while True:
        if not os.path.exists(filepath):
            return False
        try:
            st = os.stat(filepath)
        except:
            time.sleep(interval)
            continue

        size = st.st_size
        mtime = st.st_mtime

        if size == last_size and mtime == last_mtime:
            stable += 1
        else:
            stable = 0
            last_size = size
            last_mtime = mtime

        if stable >= stable_checks and (time.time() - mtime) >= min_mtime_age:
            # final re-stat after open to catch weird late updates
            try:
                f = open(filepath, "rb")
                try:
                    if size > 0:
                        f.seek(size - 1)
                        f.read(1)
                finally:
                    f.close()
                st2 = os.stat(filepath)
                if st2.st_size == size and st2.st_mtime == mtime:
                    return True
                else:
                    stable = 0
                    last_size = st2.st_size
                    last_mtime = st2.st_mtime
            except:
                time.sleep(interval)

        # Don't wait forever after closed; closed_timeout is enough overall
        if (time.time() - start) > closed_timeout:
            return False

        time.sleep(interval)

def list_candidate_files():
    patterns = [
        os.path.join(WATCH_DIR, "*.bin"),
        os.path.join(WATCH_DIR, "*", "*.bin"),
        os.path.join(WATCH_DIR, "*", "*", "*.bin"),
    ]
    files = []
    for p in patterns:
        files += glob.glob(p)
    files.sort(key=lambda x: os.path.getmtime(x))
    return files

def send_one(sock, filepath):
    if not wait_until_closed_and_stable(filepath):
        log("SKIP (not closed/stable yet or timeout): %s" % filepath)
        return False

    
    name = os.path.basename(filepath)
    st = os.stat(filepath)
    size = st.st_size

    # unique key: inode+mtime+size
    key = (st.st_ino, st.st_mtime, st.st_size)

    name_bytes = name
    try:
        if isinstance(name_bytes, unicode):
            name_bytes = name_bytes.encode("utf-8")
    except:
        pass

    # framed header: [4B name_len][name][8B size]
    hdr = struct.pack("!I", len(name_bytes)) + name_bytes + struct.pack("!Q", size)
    sock.sendall(hdr)

    sent = 0
    f = open(filepath, "rb")
    try:
        while True:
            chunk = f.read(BUF)
            if not chunk:
                break
            sock.sendall(chunk)
            sent += len(chunk)
    finally:
        f.close()

    # wait ACK "OK\n"
    ack = ""
    while "\n" not in ack:
        b = sock.recv(16)
        if not b:
            raise Exception("no ACK (socket closed)")
        ack += b
    if not ack.startswith("OK"):
        raise Exception("bad ACK: %r" % ack)

    log("Sent %s (%d bytes)" % (name, sent))

    if DELETE_AFTER_TRANSFER:
        try:
            os.remove(filepath)
            log("DELETED %s" % name)
        except Exception as e:
            log("Delete failed: %s" % e)

    return key

def monitor():
    log("Watching: %s" % WATCH_DIR)
    log("Streaming to %s:%d" % (PC_IP, PC_PORT))
    log("Waiting for writer CLOSE + stability before sending")
    if DELETE_AFTER_TRANSFER:
        log("AUTO-DELETE: enabled")
    else:
        log("AUTO-DELETE: disabled")

    sock = None
    sent_keys = set()

    while True:
        try:
            if sock is None:
                sock = connect()
                log("Connected")

            for fp in list_candidate_files():
                if not os.path.isfile(fp):
                    continue

                # quick skip if already sent (based on current inode/mtime/size)
                try:
                    st = os.stat(fp)
                    key_now = (st.st_ino, st.st_mtime, st.st_size)
                except:
                    continue
                if key_now in sent_keys:
                    continue

                key = send_one(sock, fp)
                if key:
                    sent_keys.add(key)

            time.sleep(POLL_SLEEP)

        except KeyboardInterrupt:
            log("Stopping...")
            try:
                if sock:
                    sock.close()
            except:
                pass
            return

        except Exception as e:
            log("Error: %s (reconnecting)" % e)
            try:
                if sock:
                    sock.close()
            except:
                pass
            sock = None
            time.sleep(0.5)

if __name__ == "__main__":
    monitor()
