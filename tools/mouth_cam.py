"""Live view and recording for the mouth streamer firmware (JPEG frames over Serial).

  python mouth_cam.py [COMx] [--size qvga|cif|vga] [--q N] [--snap out.jpg]
Keys in the window: R start/stop recording (out/cam/rec_YYYYmmdd_HHMMSS/), S snapshot,
L cycle the flash LED (off / dim / mid), Q or Esc quit.
A recording is frames.mjpeg (concatenated JPEGs) + frames.csv (seq, device ms, PC time, offset, len).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from serial.tools import list_ports

OUT = Path(__file__).resolve().parent.parent / "out" / "cam"
BAUD = 921600


def find_port():
    for p in list_ports.comports():
        if "CH340" in (p.description or ""):
            return p.device
    sys.exit("no CH340 port found; pass COMx")


class Stream:
    def __init__(self, port):
        self.s = serial.Serial(None, BAUD, timeout=1)
        self.s.port = port
        self.s.dtr = False      # set before open: keeps the board out of reset / boot mode
        self.s.rts = False
        self.s.open()
        self.buf = b""
        self.bad = 0

    def send(self, cmd):
        self.s.write((cmd + "\n").encode())

    def frames(self):
        """Yields (header fields, jpeg bytes) or (None, text line)."""
        while True:
            chunk = self.s.read(max(1, self.s.in_waiting))
            if not chunk:
                continue
            self.buf += chunk
            while True:
                nl = self.buf.find(b"\n")
                if nl < 0:
                    break
                j = self.buf.find(b"JPG ", 0, nl)
                if j > 0:               # joined mid-frame: JPEG tail glued to the next header
                    self.buf = self.buf[j:]
                    nl -= j
                head = self.buf[:nl].decode("ascii", "replace").strip()
                if head.startswith("JPG "):
                    parts = head.split()
                    if len(parts) != 6 or not all(x.isdigit() for x in parts[1:]):
                        self.buf = self.buf[nl + 1:]
                        continue
                    n = int(parts[4])
                    if len(self.buf) < nl + 1 + n:
                        break
                    data = self.buf[nl + 1:nl + 1 + n]
                    self.buf = self.buf[nl + 1 + n:]
                    if data[:2] != b"\xff\xd8" or not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
                        self.bad += 1   # bytes lost on the line: skip, resync on the next header
                        continue
                    yield [int(x) for x in parts[1:]], data
                else:
                    self.buf = self.buf[nl + 1:]
                    if head and head.split()[0] in ("INFO", "ERR", "STATUS", "REG"):
                        yield None, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("--size", default=None)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--snap", help="save one frame to this file and exit")
    ap.add_argument("--bench", type=float, help="count good/bad frames for this many seconds and exit")
    args = ap.parse_args()
    st = Stream(args.port or find_port())
    time.sleep(0.3)
    if args.size:
        st.send(f"size {args.size}")
    if args.q:
        st.send(f"q {args.q}")
    st.send("stream on")
    print("Keys work in the picture window (click it first): R record start/stop, S snapshot, "
          "L flash LED, Q quit. Recordings: out/cam/rec_*", flush=True)
    rec = None
    led = [0, 20, 80]
    led_i = 0
    t_last, n_fps, fps = time.time(), 0, 0.0
    for head, data in st.frames():
        if head is None:
            print(data)
            continue
        seq, w, h, n, ms = head
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if args.bench:
            n_fps += 1
            if time.time() - t_last >= args.bench:
                print(f"{n_fps / args.bench:.1f} fps good, {st.bad} bad, last {w}x{h} {n} B")
                return
            continue
        if args.snap:
            Path(args.snap).write_bytes(data)
            print("saved", args.snap, w, h)
            return
        if rec:
            off = rec["mj"].tell()
            rec["mj"].write(data)
            rec["csv"].write(f"{seq},{ms},{time.time():.4f},{off},{n}\n")
            rec["n"] += 1
        n_fps += 1
        if time.time() - t_last >= 1:
            fps, n_fps, t_last = n_fps / (time.time() - t_last), 0, time.time()
        view = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2), cv2.COLOR_GRAY2BGR)
        txt = f"{w}x{h} {fps:.1f} fps {n / 1024:.1f} KB bad {st.bad}" + (f"  REC {rec['n']}" if rec else "")
        cv2.putText(view, txt, (8, 24), 0, 0.7, (0, 0, 255) if rec else (0, 255, 0), 2)
        cv2.imshow("mouth cam", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            OUT.mkdir(parents=True, exist_ok=True)
            p = OUT / time.strftime("snap_%Y%m%d_%H%M%S.jpg")
            p.write_bytes(data)
            print("saved", p)
        if k == ord("l"):
            led_i = (led_i + 1) % len(led)
            st.send(f"led {led[led_i]}")
        if k == ord("r"):
            if rec:
                rec["mj"].close(); rec["csv"].close()
                print("recorded", rec["n"], "frames to", rec["dir"], flush=True)
                rec = None
            else:
                d = OUT / time.strftime("rec_%Y%m%d_%H%M%S")
                d.mkdir(parents=True)
                rec = {"dir": d, "n": 0, "mj": open(d / "frames.mjpeg", "wb"), "csv": open(d / "frames.csv", "w")}
                rec["csv"].write("seq,dev_ms,pc_time,offset,len\n")
    if rec:
        rec["mj"].close(); rec["csv"].close()
        print("recorded", rec["n"], "frames to", rec["dir"])


if __name__ == "__main__":
    main()
