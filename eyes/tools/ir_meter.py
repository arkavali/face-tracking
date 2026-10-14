"""Relative IR irradiance meter: compare an LED with a reference light using a camera.

Idea (the user's): a camera with locked exposure is a crude radiometer. Light a white
sheet placed where the eye would be, once with the reference (e.g. the GC0308 webcam's
own IR LEDs at its normal working distance), once with our LED at its real distance in
the mask, plus a dark frame; the ratio of the dark-subtracted brightness is the ratio of
irradiances at the sheet - valid only when the camera is linear and not saturated.

  python ir_meter.py capture dark --cam 0        all LEDs off
  python ir_meter.py capture ref  --cam 0        reference light on
  python ir_meter.py capture test --cam 0        our LED on (reference LEDs covered)
  python ir_meter.py linearity --cam 0           light fixed, exposure stepped: is it linear?
  python ir_meter.py compare                     ratio test / ref (and both vs dark)

UVC cameras through OpenCV/DirectShow: auto exposure is switched off and the exposure set
with --exp (log2 seconds, e.g. -6 = 1/64 s); not every driver honours it, which the
linearity check shows. Stats (centre ROI mean, saturation share) go to out/ir_meter/.
Not yet run against the real cameras.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "out" / "ir_meter"


def open_cam(idx: int, exp: float):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"camera {idx} not found")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow: 0.25 = manual
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    for _ in range(15):                          # let the driver apply the settings
        cap.read()
    return cap


def grab(cap, n: int = 30) -> np.ndarray:
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if not frames:
        raise SystemExit("no frames")
    return np.mean(frames, 0)


def stats(img: np.ndarray, roi: float) -> dict:
    h, w = img.shape
    y0, y1 = int(h * (0.5 - roi / 2)), int(h * (0.5 + roi / 2))
    x0, x1 = int(w * (0.5 - roi / 2)), int(w * (0.5 + roi / 2))
    c = img[y0:y1, x0:x1]
    return dict(mean=float(c.mean()), p99=float(np.percentile(c, 99)), sat=float((c >= 250).mean()),
                frame_max=float(img.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["capture", "linearity", "compare"])
    ap.add_argument("label", nargs="?")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--exp", type=float, default=-6)
    ap.add_argument("--roi", type=float, default=0.2, help="centre ROI size as a share of the frame")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.action == "capture":
        cap = open_cam(a.cam, a.exp)
        img = grab(cap)
        s = stats(img, a.roi) | dict(exp=a.exp, time=time.time())
        cv2.imwrite(str(OUT / f"{a.label}.png"), img.astype(np.uint8))
        (OUT / f"{a.label}.json").write_text(json.dumps(s, indent=1))
        print(a.label, s)
        if s["sat"] > 0.001 or s["p99"] > 240:
            print("WARNING: saturated - lower --exp or move the light away; the ratio would be wrong")
    elif a.action == "linearity":
        cap = open_cam(a.cam, a.exp)
        rows = []
        for e in np.arange(a.exp - 3, a.exp + 1):
            cap.set(cv2.CAP_PROP_EXPOSURE, float(e))
            for _ in range(10):
                cap.read()
            s = stats(grab(cap, 15), a.roi)
            rows.append((float(e), s["mean"]))
            print(f"exp {e:+.0f}: mean {s['mean']:.1f}")
        m = np.array(rows)
        ok = m[:, 1] < 240
        if ok.sum() >= 3:
            slope = np.polyfit(m[ok, 0], np.log2(np.maximum(m[ok, 1], 1e-3)), 1)[0]
            print(f"log2(brightness) per log2(exposure) = {slope:.2f} (1.0 = linear; ~0.45 = gamma is on)")
    else:
        d = {k: json.loads((OUT / f"{k}.json").read_text()) for k in ("dark", "ref", "test")
             if (OUT / f"{k}.json").exists()}
        if not {"ref", "test"} <= d.keys():
            raise SystemExit("capture ref and test first (and dark)")
        dark = d.get("dark", {"mean": 0.0})["mean"]
        r, t = d["ref"]["mean"] - dark, d["test"]["mean"] - dark
        print(f"ref {r:.1f}, test {t:.1f} grey levels above dark -> test / ref = {t / max(r, 1e-6):.2f}")
        if d["ref"]["exp"] != d["test"]["exp"]:
            print("WARNING: different exposures")


if __name__ == "__main__":
    main()
