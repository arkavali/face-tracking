"""Check that the C++ port (firmware/lib/eye_core) matches pupil.py / eyemodel.py.

Builds the core as a host DLL with zig (pip package "ziglang", in brow_tracker's venv) in
double and float precision, runs both and the Python reference on the same frames and
reports the largest differences plus the host time per frame.
Usage: python parity_test.py [out/seq/<name>.npz | video.mp4] [--frames N]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eyemodel import EyeModel
from pupil import GlintPairer, PupilDetector, PupilGate, find_glints

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "firmware" / "lib" / "eye_core"
HOST = REPO / "tools" / "host"
OUT = REPO / "out"


def build(real: str) -> Path:
    out = OUT / "host" / f"eye_{real}.dll"
    out.parent.mkdir(parents=True, exist_ok=True)
    srcs = [CORE / "eye_core.cpp", HOST / "eye_capi.cpp"]
    if not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in srcs + [CORE / "eye_core.h"]):
        cmd = [sys.executable, "-m", "ziglang", "c++", "-shared", "-O2", "-w", "-std=c++17", f"-DEYE_REAL={real}",
               f"-I{CORE}", "-o", str(out), *map(str, srcs), "-target", "x86_64-windows-gnu"]
        subprocess.run(cmd, check=True)
    return out


def load(dll: Path):
    lib = ctypes.CDLL(str(dll))
    lib.eye_new.restype = ctypes.c_void_p
    lib.eye_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
    lib.eye_free.argtypes = [ctypes.c_void_p]
    lib.eye_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return lib


def frames_from(src: str, n: int):
    if src.endswith(".npz"):
        d = np.load(src)
        return d["imgs"][:n], float(d["f"])
    import cv2
    cap = cv2.VideoCapture(src)
    out = []
    while len(out) < n:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (320, 240), interpolation=cv2.INTER_AREA))
    return np.array(out), 308.0


def python_run(imgs, f):
    h, w = imgs.shape[1:]
    det, gate, model, pairer = PupilDetector(), PupilGate(), EyeModel(f, w, h), GlintPairer()
    rows = []
    for img in imgs:
        e = det.detect(img)
        r = np.zeros(26)
        if e is not None:
            r[:9] = [1, e.cx, e.cy, e.a, e.b, e.angle, e.support, e.contrast, e.top_vis]
        if gate(e):
            r[9] = 1
            model.update(e)
            g = model.gaze_cam(e)
            if g is not None:
                r[14:17] = g
            gl = find_glints(img, e)
            r[17] = len(gl)
            pair = pairer(gl)
            if pair is not None:
                r[18:22] = [pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
        r[10] = model.ready
        if model.c is not None:
            r[11:13] = model.c
        r[13] = model.R or 0.0
        rows.append(r)
    return np.array(rows)


def cpp_run(lib, imgs, f):
    h, w = imgs.shape[1:]
    ctx = lib.eye_new(w, h, f)
    out = np.zeros(26)
    rows = []
    t0 = time.perf_counter()
    for img in imgs:
        a = np.ascontiguousarray(img, np.uint8)
        lib.eye_update(ctx, a.ctypes.data, w, h, out.ctypes.data)
        rows.append(out.copy())
    ms = (time.perf_counter() - t0) / len(imgs) * 1000
    lib.eye_free(ctx)
    return np.array(rows), ms


COLS = ["found", "cx", "cy", "a", "b", "angle", "support", "contrast", "top_vis", "gated", "ready",
        "mcx", "mcy", "R", "gx", "gy", "gz", "n_glints"]


def compare(ref, got, label):
    print(f"--- {label}")
    both = (ref[:, 0] == 1) & (got[:, 0] == 1)
    print(f"found: python {int(ref[:, 0].sum())}, c++ {int(got[:, 0].sum())}, "
          f"disagree {int((ref[:, 0] != got[:, 0]).sum())} / {len(ref)}")
    for k, name in enumerate(COLS[1:9], start=1):
        d = np.abs(ref[both, k] - got[both, k])
        if name == "angle":   # an ellipse's axis direction is defined modulo pi
            d = np.abs((ref[both, k] - got[both, k] + np.pi / 2) % np.pi - np.pi / 2)
        print(f"  {name:9s} max |diff| {d.max() if d.size else 0:.2e}")
    for k in (9, 10, 17):
        print(f"  {COLS[k]:9s} disagree {int((ref[:, k] != got[:, k]).sum())}")
    m = (ref[:, 10] == 1) & (got[:, 10] == 1)
    print(f"  glint pair disagree {int((np.abs(ref[:, 18:22] - got[:, 18:22]).max(1) > 0.01).sum())}")
    for k in (11, 12, 13, 14, 15, 16):
        d = np.abs(ref[m, k] - got[m, k])
        print(f"  {COLS[k]:9s} max |diff| {d.max() if d.size else 0:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=str(OUT / "seq" / "thru_ledside.npz"))
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    imgs, f = frames_from(a.src, a.frames)
    t0 = time.perf_counter()
    ref = python_run(imgs, f)
    py_ms = (time.perf_counter() - t0) / len(imgs) * 1000
    print(f"{len(imgs)} frames from {a.src}; python {py_ms:.1f} ms/frame")
    for real in ("double", "float"):
        got, ms = cpp_run(load(build(real)), imgs, f)
        compare(ref, got, f"C++ {real}: {ms:.2f} ms/frame on the host")
    np.save(OUT / "parity_ref.npy", ref)
    (OUT / "parity_cols.json").write_text(json.dumps(COLS))


if __name__ == "__main__":
    main()
