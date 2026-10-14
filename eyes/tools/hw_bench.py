"""Run the eye core on a real ESP32-S3 (firmware env s3bench) and compare with Python.

python hw_bench.py COM7 [out/seq/<name>.npz] [--frames 200]
Sends frames over the UART, collects the chip's results and times, and prints the same
comparison as parity_test.py (the chip computes in float).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import serial

from parity_test import compare, frames_from, python_run, OUT


def open_port(port):
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 2000000, 2.0
    s.dtr = False
    s.rts = False
    s.open()
    time.sleep(0.3)
    s.reset_input_buffer()
    return s


def ask(s, line: str, payload: bytes = b"", prefix: str = "") -> str:
    s.write(line.encode() + b"\n" + payload)
    t0 = time.time()
    while time.time() - t0 < 5:
        r = s.readline().decode(errors="replace").strip()
        if r and (not prefix or r.startswith(prefix)):
            return r
    raise RuntimeError(f"no reply to {line!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("src", nargs="?", default=str(OUT / "seq" / "thru_ledside.npz"))
    ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args()
    imgs, f = frames_from(a.src, a.frames)
    s = open_port(a.port)
    print(ask(s, "info", prefix="chip"))
    print(ask(s, "reset", prefix="OK"))
    h, w = imgs.shape[1:]
    rows, ms = [], []
    t0 = time.time()
    for img in imgs:
        r = ask(s, f"F {w} {h}", np.ascontiguousarray(img, np.uint8).tobytes(), prefix="R ")
        v = [float(x) for x in r.split()[1:]]
        ms.append(v[0])
        rows.append(v[1:])
    print(f"{len(imgs)} frames in {time.time() - t0:.0f} s over the UART")
    got = np.array(rows)
    out = np.zeros((len(got), 26))
    out[:, :got.shape[1]] = got
    ms = np.array(ms)
    found = got[:, 0] == 1
    print(f"chip time per frame (detect + gate + model + glints): median {np.median(ms):.2f} ms, "
          f"p90 {np.percentile(ms, 90):.2f}, max {ms.max():.2f}; with pupil {np.median(ms[found]):.2f}, "
          f"without {np.median(ms[~found]) if (~found).any() else float('nan'):.2f}")
    print(ask(s, "bench 20", prefix="B"), "ms per detect (last frame, 20 repeats)")
    ref = python_run(imgs, f)
    compare(ref, out, "ESP32-S3 float vs Python")
    s.close()


if __name__ == "__main__":
    main()
