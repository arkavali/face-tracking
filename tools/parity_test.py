"""C++ mouth_net (host DLL built with zig) vs the ONNX model on recorded frames.

  python parity_test.py [out/calib/cal_...]      (default: the newest calibration session)
Checks the 4x downscale against cv2.INTER_AREA and the net outputs against onnxruntime.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import train

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "firmware" / "lib" / "mouth_net"
DLL = ROOT / "out" / "mouth_net_host.dll"


def build():
    DLL.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "ziglang", "c++", "-shared", "-O2", "-w", "-std=c++17", "-DMOUTH_HOST",
           str(LIB / "mouth_net.cpp"), "-o", str(DLL)]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(DLL))
    lib.mouth_scratch.restype = ctypes.c_int
    return lib


def main():
    sess_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "out" / "calib").glob("cal_*"))[-1]
    lib = build()
    n = lib.mouth_scratch()
    a = np.zeros(n, np.float32)
    b = np.zeros(n, np.float32)
    ort_s = ort.InferenceSession(str(train.OUT / "mouth.onnx"), providers=["CPUExecutionProvider"])
    raw = (sess_dir / "frames.mjpeg").read_bytes()
    import csv
    rows = list(csv.DictReader(open(sess_dir / "frames.csv")))[::25]
    dmax, omax, t_c = 0, 0.0, 0.0
    fp = lambda x: x.ctypes.data_as(ctypes.c_void_p)
    for r in rows:
        o, ln = int(r["offset"]), int(r["len"])
        g = np.ascontiguousarray(cv2.imdecode(np.frombuffer(raw[o:o + ln], np.uint8), cv2.IMREAD_GRAYSCALE))
        ref_small = cv2.resize(g, (train.IN_W, train.IN_H), interpolation=cv2.INTER_AREA)
        small = np.zeros((train.IN_H, train.IN_W), np.uint8)
        lib.mouth_down4(fp(g), g.shape[1], g.shape[0], fp(small))
        dmax = max(dmax, int(np.abs(small.astype(int) - ref_small).max()))
        out = np.zeros(22, np.float32)
        t0 = time.perf_counter()
        lib.mouth_run(fp(small), fp(a), fp(b), fp(out))
        t_c += time.perf_counter() - t0
        ref = ort_s.run(None, {"img": (small.astype(np.float32) / 255)[None, None]})[0][0]
        omax = max(omax, float(np.abs(out[:len(ref)] - ref).max()))
    print(f"{len(rows)} frames from {sess_dir.name}: downscale max diff {dmax} (0 = same as cv2), "
          f"output max diff {omax:.2e}, C++ {t_c / len(rows) * 1000:.2f} ms/frame on the PC, scratch {n} floats")


if __name__ == "__main__":
    main()
