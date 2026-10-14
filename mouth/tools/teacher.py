"""Babble face model as a teacher: 45 Unified-Expressions values per recorded frame.

  python teacher.py out/cam/rec_*        -> <rec>/babble.npz (+ babble.png plot)

data/babble_face.onnx = Baballonia's faceModel.onnx (EfficientNet-B0, 224x224 grey, /255 input;
Babble Software Distribution License: non-commercial). Preprocessing as BabbleApp: grey frame
(optionally cropped), resize to 224x224, /255, no filtering here (raw outputs, clipped 0..1).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "babble_face.onnx"
NAMES = [
    "cheekPuffLeft", "cheekPuffRight", "cheekSuckLeft", "cheekSuckRight", "jawOpen", "jawForward",
    "jawLeft", "jawRight", "noseSneerLeft", "noseSneerRight", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight", "mouthRollUpper", "mouthRollLower", "mouthShrugUpper",
    "mouthShrugLower", "mouthClose", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthStretchLeft", "mouthStretchRight", "tongueOut", "tongueUp",
    "tongueDown", "tongueLeft", "tongueRight", "tongueRoll", "tongueBendDown", "tongueCurlUp",
    "tongueSquish", "tongueFlat", "tongueTwistLeft", "tongueTwistRight",
]
PLOT = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthFunnel", "mouthPucker", "mouthLeft",
        "mouthRight", "cheekPuffLeft", "cheekPuffRight", "tongueOut", "mouthUpperUpLeft", "mouthLowerDownLeft"]


def load_recording(d: Path):
    raw = (d / "frames.mjpeg").read_bytes()
    rows = list(csv.DictReader(open(d / "frames.csv")))
    for r in rows:
        o, n = int(r["offset"]), int(r["len"])
        img = cv2.imdecode(np.frombuffer(raw[o:o + n], np.uint8), cv2.IMREAD_GRAYSCALE)
        yield float(r["pc_time"]), img


def session():
    return ort.InferenceSession(str(MODEL), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def infer(sess, imgs, crop=None, flip=False, batch=64):
    """imgs: list of grey frames -> (N, 45). crop = (x0, y0, x1, y1) in pixels."""
    name = sess.get_inputs()[0].name
    out = []
    for a in range(0, len(imgs), batch):
        x = []
        for im in imgs[a:a + batch]:
            if crop:
                im = im[crop[1]:crop[3], crop[0]:crop[2]]
            if flip:
                im = im[:, ::-1]
            x.append(cv2.resize(im, (224, 224)).astype(np.float32) / 255.0)
        x = np.stack(x)[:, None]
        # the exported graph has batch 1 fixed: run frame by frame if needed
        try:
            y = sess.run(None, {name: x})[0]
        except Exception:
            y = np.concatenate([sess.run(None, {name: xi[None]})[0] for xi in x])
        out.append(y)
    return np.clip(np.concatenate(out), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--flip", action="store_true", help="mirror the frames (camera mounted mirrored)")
    args = ap.parse_args()
    sess = session()
    for r in args.recs:
        d = Path(r)
        if not (d / "frames.mjpeg").exists():
            continue
        t, imgs = zip(*load_recording(d))
        t = np.array(t) - t[0]
        y = infer(sess, list(imgs), flip=args.flip)
        np.savez(d / "babble.npz", t=t, y=y, names=np.array(NAMES))
        fig, axs = plt.subplots(len(PLOT), 1, figsize=(14, 1.1 * len(PLOT)), sharex=True)
        for ax, n in zip(axs, PLOT):
            ax.plot(t, y[:, NAMES.index(n)], lw=0.8)
            ax.set_ylim(0, 1); ax.set_ylabel(n, rotation=0, ha="right", fontsize=8)
        axs[-1].set_xlabel("s")
        plt.tight_layout(); plt.savefig(d / "babble.png", dpi=80); plt.close(fig)
        top = np.argsort(-y.max(0))[:12]
        print(d.name, len(t), "frames; max per channel:", ", ".join(f"{NAMES[i]} {y[:, i].max():.2f}" for i in top))


if __name__ == "__main__":
    main()
