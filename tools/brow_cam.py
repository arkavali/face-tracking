"""PC helper for the brow camera firmware (firmware/src/main.cpp) over Serial.

  python brow_cam.py COM8 snap [out.png]   save the current camera frame
  python brow_cam.py COM8 setup            snapshot -> draw both brow ROIs with the mouse
                                           -> calibrate on a neutral face -> range up/down
  python brow_cam.py COM8 monitor          live plot of the BROW lines (Ctrl+C to stop)
  python brow_cam.py COM8 cmd "status"     send one raw command and print the reply

Not tested against real hardware yet (written before the camera was mounted).
Opening the port with DTR/RTS low avoids resetting the ESP32 on connect.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import serial

BAUD = 921600


def open_port(port: str) -> serial.Serial:
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, BAUD, 0.5
    s.dtr = False
    s.rts = False
    s.open()
    time.sleep(0.2)
    s.reset_input_buffer()
    return s


def command(s: serial.Serial, cmd: str, wait: float = 0.5) -> list:
    s.write((cmd + "\n").encode())
    t0, out = time.time(), []
    while time.time() - t0 < wait:
        ln = s.readline().decode(errors="replace").strip()
        if ln and not ln.startswith("BROW"):
            out.append(ln)
    return out


def snap(s: serial.Serial) -> np.ndarray:
    s.write(b"stream off\nsnap\n")
    t0 = time.time()
    while time.time() - t0 < 5:
        ln = s.readline().decode(errors="replace").strip()
        if ln.startswith("SNAP"):
            _, w, h = ln.split()
            w, h = int(w), int(h)
            data = s.read(w * h)
            if len(data) == w * h:
                return np.frombuffer(data, np.uint8).reshape(h, w)
    raise RuntimeError("no SNAP reply")


def setup(s: serial.Serial) -> None:
    img = snap(s)
    big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    print("Draw the LEFT-on-image brow ROI, Enter; then the RIGHT one, Enter; then Esc.")
    print("Each ROI: from below the hairline to below the brow, with room above for a full raise.")
    rois = cv2.selectROIs("brow ROIs (image left first)", big, showCrosshair=False)
    cv2.destroyAllWindows()
    if len(rois) != 2:
        sys.exit("need exactly two ROIs")
    for (x, y, w, h), side in zip(rois, "LR"):
        x0, y0, x1, y1 = (int(round(v / 3)) for v in (x, y, x + w, y + h))
        # the nose is between the brows: the inner side of the left ROI is its right side
        inner = "right" if side == "L" else "left"
        print(" ", command(s, f"roi {side} {x0} {y0} {x1} {y1} {inner}"))
    input("Relax your brows (neutral face) and press Enter...")
    print(" ", command(s, "calib 15", wait=3.0))
    s.write(b"stream on\n")
    input("Raise both brows as high as you can and press Enter while holding...")
    print(" ", command(s, "range up"))
    input("Lower / frown both brows as far as you can and press Enter while holding...")
    print(" ", command(s, "range down"))
    print(" ", command(s, "status"))


def monitor(s: serial.Serial) -> None:
    import matplotlib.pyplot as plt

    s.write(b"stream on\n")
    keys = ["rL", "rR", "iL", "iR"]
    hist = {k: deque(maxlen=600) for k in keys}
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    lines = {k: ax.plot([], [], label=k)[0] for k in keys}
    ax.legend(loc="upper left")
    ax.set_ylim(-30, 30)
    last = time.time()
    try:
        while True:
            ln = s.readline().decode(errors="replace").strip()
            if not ln.startswith("BROW"):
                continue
            kv = dict(p.split("=", 1) for p in ln.split()[1:] if "=" in p)
            for k in keys:
                hist[k].append(float(kv.get(k, "nan")))
            if time.time() - last > 0.1:
                for k in keys:
                    lines[k].set_data(range(len(hist[k])), list(hist[k]))
                ax.set_xlim(0, max(len(hist["rL"]), 10))
                fig.canvas.draw_idle()
                plt.pause(0.001)
                last = time.time()
    except KeyboardInterrupt:
        s.write(b"stream off\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("action", choices=["snap", "setup", "monitor", "cmd"])
    ap.add_argument("arg", nargs="?")
    args = ap.parse_args()
    s = open_port(args.port)
    if args.action == "snap":
        img = snap(s)
        out = args.arg or "brow_snap.png"
        cv2.imwrite(out, img)
        print(out, img.shape)
    elif args.action == "setup":
        setup(s)
    elif args.action == "monitor":
        monitor(s)
    else:
        for ln in command(s, args.arg or "status", wait=1.0):
            print(ln)


if __name__ == "__main__":
    main()
