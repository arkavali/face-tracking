"""Run the brow tracker directly on a real video (e.g. a phone held above the brows).

Input: a video file, a folder of images, or a recording folder from the brow camera UI
(out/cam/rec_*, frames/*.png).

ROIs are given by hand, like the semi-manual calibration on the device:
  --roi NAME x0 y0 x1 y1 INNER   (pixels at the processing size, INNER = left|right:
                                  which ROI side is towards the nose)
Without --roi a window opens on the first frame to draw them with the mouse.
The first --calib frames are taken as the neutral face.
Writes out/video/<stem>_<method>.{npz,png} and an overlay sheet at the extremes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT
from tracker import BrowTracker, TrackerConfig
from tracker_v3 import BrowTrackerV3


def to_grey(frame: np.ndarray, size: tuple, red_weight: float) -> np.ndarray:
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = cv2.split(small)
    return np.clip(red_weight * r + (1 - red_weight) * g, 0, 255).astype(np.uint8)


def read_frames(src: Path, start: int, end: int):
    """Colour or grey frames from a video file or an image folder, and the frame rate."""
    if src.is_dir():
        folder = src / "frames" if (src / "frames").is_dir() else src
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        files = files[start:end or None]
        fps = 5.0
        times = src / "frames.csv"  # written by the brow camera UI: frame,t,seq
        if times.exists():
            t = np.loadtxt(times, delimiter=",", skiprows=1, ndmin=2)[:, 1]
            if len(t) > 1 and t[-1] > t[0]:
                fps = (len(t) - 1) / (t[-1] - t[0])
        return [cv2.imread(str(f)) for f in files], fps
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (end and i >= end):
            break
        if i >= start:
            out.append(f)
        i += 1
    cap.release()
    return out, fps


def pick_rois(img: np.ndarray) -> list:
    """Draw the brow ROIs with the mouse on a neutral frame (Enter after each, Esc to finish)."""
    zoom = max(1, 960 // img.shape[1])
    big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    print("Draw one rectangle per brow (from below the hairline to below the brow, with room above it")
    print("for a full raise); Enter after each, Esc when done.")
    rects = cv2.selectROIs("brow ROIs - Enter after each, Esc to finish", big, showCrosshair=False)
    cv2.destroyAllWindows()
    rois = []
    for x, y, w, h in sorted((tuple(int(v) for v in r) for r in rects), key=lambda r: r[0]):
        x0, y0, x1, y1 = (int(round(v / zoom)) for v in (x, y, x + w, y + h))
        # the nose is towards the image centre
        inner = "right" if (x0 + x1) / 2 < img.shape[1] / 2 else "left"
        rois.append([f"img{'L' if inner == 'right' else 'R'}", x0, y0, x1, y1, inner])
    if not rois:
        raise SystemExit("no ROI drawn")
    print("same ROIs from the command line:", " ".join("--roi " + " ".join(map(str, r)) for r in rois))
    return rois


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="video file, image folder or brow camera recording folder")
    ap.add_argument("--size", default="", help="processing size WxH; default 320 wide, source aspect")
    ap.add_argument("--roi", nargs=6, action="append",
                    metavar=("NAME", "X0", "Y0", "X1", "Y1", "INNER"), help="omit to draw them with the mouse")
    ap.add_argument("--open", action="store_true", help="open the result images when done")
    ap.add_argument("--calib", type=int, default=None,
                    help="neutral = first N frames (default 30 for videos, 10 for ~5 fps image folders); "
                         "0 = every 10th frame of the whole range (median face)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--method", default="ncc")
    ap.add_argument("--v3", action="store_true", help="use the experimental tracker_v3")
    ap.add_argument("--red", type=float, default=0.85)
    args = ap.parse_args()
    raw, fps = read_frames(args.video, args.start, args.end)
    if not raw:
        raise SystemExit(f"no frames in {args.video}")
    if args.size:
        size = tuple(int(v) for v in args.size.split("x"))
    else:
        h0, w0 = raw[0].shape[:2]
        size = (320, int(round(320 * h0 / w0)))
    frames = [to_grey(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR), size, args.red) for f in raw]
    del raw
    if not args.roi:
        args.roi = pick_rois(frames[0])

    cfg = TrackerConfig(method=args.method)
    trackers = {}
    if args.calib is None:
        args.calib = 10 if args.video.is_dir() else 30
    calib = frames[:args.calib] if args.calib else frames[::10]
    for name, x0, y0, x1, y1, inner in args.roi:
        tr = (BrowTrackerV3 if args.v3 else BrowTracker)((int(x0), int(y0), int(x1), int(y1)), cfg,
                                                         inner_is_right=(inner == "right"))
        if not tr.calibrate(calib):
            raise SystemExit(f"calibration failed for {name}")
        trackers[name] = tr
    res = {n: [] for n in trackers}
    for img in frames:
        for n, tr in trackers.items():
            res[n].append(tr.update(img))

    out_dir = OUT / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video.stem or args.video.resolve().name}_{'v3' if args.v3 else args.method}"
    pack = {}
    for n, rs in res.items():
        for k in ("raise", "inner", "outer", "valid"):
            pack[f"{n}_{k}"] = np.array([r[k] for r in rs])
        pack[f"{n}_off"] = np.array([r["off"] for r in rs])
        pack[f"{n}_ok"] = np.array([r["ok"] for r in rs])
    np.savez_compressed(out_dir / f"{stem}.npz", fps=fps, **pack)

    t = (np.arange(len(frames)) + args.start) / fps
    fig, axes = plt.subplots(len(trackers), 1, figsize=(14, 3.2 * len(trackers)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, trackers):
        ax.plot(t, pack[f"{n}_raise"], color="#d9822b", lw=1.4, label="raise (mean of strips)")
        ax.plot(t, pack[f"{n}_inner"], color="#1c7ed6", lw=0.9, label="inner end")
        ax.plot(t, pack[f"{n}_outer"], color="#2b8a3e", lw=0.9, label="outer end")
        bad = ~pack[f"{n}_valid"].astype(bool)
        lo, hi = ax.get_ylim()
        ax.fill_between(t, lo, hi, where=bad, color="red", alpha=0.12, step="mid")
        ax.set_ylabel(f"{n}, px up")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("s")
    fig.suptitle(f"{args.video.name} - {args.method}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=90)
    plt.close(fig)

    # Overlay sheet: neutral, and the extremes of each brow.
    picks = [0]
    for n in trackers:
        r = pack[f"{n}_raise"]
        picks += [int(np.argmax(r)), int(np.argmin(r))]
    picks += list(np.linspace(0, len(frames) - 1, 6).astype(int))
    tiles = []
    for fi in picks:
        img = cv2.cvtColor(frames[fi], cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (size[0] * 2, size[1] * 2), interpolation=cv2.INTER_NEAREST)
        for n, tr in trackers.items():
            cv2.rectangle(img, (tr.x0 * 2, tr.y0 * 2), (tr.x1 * 2 - 1, tr.y1 * 2 - 1), (90, 90, 90), 1)
            for seg in tr.curve_points(np.zeros(cfg.strips)):
                cv2.polylines(img, [np.round(seg * 2).astype(np.int32)], False, (0, 150, 0), 1, cv2.LINE_AA)
            off = pack[f"{n}_off"][fi]
            ok = pack[f"{n}_ok"][fi]
            for j, seg in enumerate(tr.curve_points(off)):
                col = (0, 200, 255) if ok[j] else (0, 0, 255)
                cv2.polylines(img, [np.round(seg * 2).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        label = f"f{fi + args.start} " + " ".join(f"{n}:{pack[f'{n}_raise'][fi]:+.1f}" for n in trackers)
        cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 0), 1)
        tiles.append(img)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(str(out_dir / f"{stem}_sheet.png"), np.vstack(rows))
    print(out_dir / f"{stem}.png")
    if args.open:
        import os

        for f in (f"{stem}.png", f"{stem}_sheet.png"):
            os.startfile(out_dir / f)


if __name__ == "__main__":
    main()
