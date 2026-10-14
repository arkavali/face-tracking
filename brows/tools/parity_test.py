"""Check that the C++ port (firmware/lib/brow_core) matches tools/tracker.py.

Builds the core as a host DLL with zig (pip package "ziglang") in double and float
precision, runs both on the same real frames and reports the largest differences
plus the host time per update.
Usage: python parity_test.py <video> --roi NAME x0 y0 x1 y1 left|right [...]
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import OUT, REPO
from track_video import to_grey
from tracker import BrowTracker, TrackerConfig

CORE = REPO / "firmware" / "lib" / "brow_core"
HOST = REPO / "tools" / "host"


def build(real: str) -> Path:
    out = OUT / "host" / f"brow_{real}.dll"
    out.parent.mkdir(parents=True, exist_ok=True)
    srcs = [CORE / "brow_core.cpp", HOST / "brow_capi.cpp"]
    if not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in srcs + [CORE / "brow_core.h"]):
        cmd = [sys.executable, "-m", "ziglang", "c++", "-shared", "-O2", "-w", "-std=c++17", f"-DBROW_REAL={real}",
               f"-I{CORE}", "-o", str(out), *map(str, srcs), "-target", "x86_64-windows-gnu"]
        subprocess.run(cmd, check=True)
    return out


def load(dll: Path):
    lib = ctypes.CDLL(str(dll))
    lib.brow_new.restype = ctypes.c_void_p
    lib.brow_new.argtypes = [ctypes.c_int] * 6 + [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.brow_free.argtypes = [ctypes.c_void_p]
    lib.brow_calibrate.restype = ctypes.c_int
    lib.brow_calibrate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.brow_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.brow_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--roi", nargs=6, action="append", required=True)
    ap.add_argument("--size", default="320x180")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--calib", type=int, default=30)
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    cap = cv2.VideoCapture(str(args.video))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (args.end and i >= args.end):
            break
        if i >= args.start:
            frames.append(np.ascontiguousarray(to_grey(f, size, 0.85)))
        i += 1
    w, h = size
    calib = np.ascontiguousarray(np.stack(frames[:args.calib]))
    cfg = TrackerConfig(method="ncc")
    libs = {r: load(build(r)) for r in ("double", "float")}
    for name, x0, y0, x1, y1, inner in args.roi:
        roi = (int(x0), int(y0), int(x1), int(y1))
        py = BrowTracker(roi, cfg, inner_is_right=(inner == "right"))
        assert py.calibrate(list(calib))
        print(f"== {name}: python curve {np.round(py.curve, 6)} strips {py.strip_x} "
              f"K={py.cfg.half_range} m={py.cfg.edge_half} S={py.cfg.search} h={py.cfg.ncc_half}")
        for real, lib in libs.items():
            t = lib.brow_new(*roi, int(inner == "right"), cfg.strips, cfg.range_mm, cfg.cand_frac, cfg.lost_frames)
            ok = lib.brow_calibrate(t, calib.ctypes.data, len(calib), w, h)
            info = np.zeros(7 + 2 * cfg.strips)
            lib.brow_info(t, info.ctypes.data)
            print(f"   {real:6s} calibrated={ok} curve {np.round(info[:3], 6)} "
                  f"strips {info[7:].astype(int).reshape(-1, 2).tolist()} K={int(info[3])} m={int(info[4])} "
                  f"S={int(info[5])} h={int(info[6])}")
            out = np.zeros(4 + 3 * cfg.strips)
            py2 = BrowTracker(roi, cfg, inner_is_right=(inner == "right"))
            py2.calibrate(list(calib))
            d_raise, d_off, n_valid_diff, t_c = [], [], 0, 0.0
            for img in frames:
                rp = py2.update(img)
                t0 = time.perf_counter()
                lib.brow_update(t, img.ctypes.data, w, h, out.ctypes.data)
                t_c += time.perf_counter() - t0
                d_raise.append(abs(rp["raise"] - out[0]))
                d_off.append(np.abs(rp["off"] - out[4::3][:cfg.strips]).max())
                n_valid_diff += int(rp["valid"] != bool(out[3]))
            lib.brow_free(t)
            d_raise, d_off = np.array(d_raise), np.array(d_off)
            print(f"   {real:6s} |raise diff| max {d_raise.max():.2e} p99 {np.percentile(d_raise, 99):.2e}; "
                  f"|strip diff| max {d_off.max():.2e}; frames >0.1 px: {(d_off > 0.1).sum()}/{len(frames)}; "
                  f"valid mismatches {n_valid_diff}; host C++ {1e3 * t_c / len(frames):.3f} ms/frame")


if __name__ == "__main__":
    main()
