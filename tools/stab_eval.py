"""Tracker vs MediaPipe on REAL frontal footage, without the 3D re-rendering.

Each frame is aligned to a neutral reference frame with a similarity transform fitted
to rigid landmarks (eye corners, nose), which mimics a camera fixed to the head. The
tracker then runs on the aligned grey frames with ROIs placed from the reference brow
landmarks, and its output is compared with MediaPipe's own brow landmarks moved by the
same transform. This checks the tracker on the real brow/glasses texture; the mask
camera's viewing angle and lighting are NOT simulated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BROW, OUT
from tracker import BrowTracker, TrackerConfig
from tracker_v3 import BrowTrackerV3

RIGID = [33, 133, 362, 263, 6, 197, 195, 5, 4, 1, 98, 327]
IPD_MM = 63.0


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", type=int, nargs=2, required=True, metavar=("START", "END"))
    ap.add_argument("--px-per-mm", type=float, default=3.0, help="scale of the aligned frames")
    ap.add_argument("--method", default="ncc")
    ap.add_argument("--v3", action="store_true", help="use the experimental tracker_v3")
    ap.add_argument("--v3-margin", type=float, default=0.15)
    ap.add_argument("--v3-range", type=float, default=16.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cfg", nargs="*", default=[], help="TrackerConfig overrides, key=value")
    ap.add_argument("--tag", default="")
    ap.add_argument("--red", type=float, default=0.85)
    args = ap.parse_args()

    d = np.load(args.video.parent.parent / "teacher" / f"{args.video.stem}.npz")
    w, h = int(d["w"]), int(d["h"])
    lm = d["lm"][:, :, :2] * [w, h]
    valid = ~np.isnan(lm[:, 0, 0])
    a, b = args.calib
    ref = np.nanmedian(lm[a:b][valid[a:b]], axis=0)
    eye_r = ref[[33, 133]].mean(0)
    eye_l = ref[[263, 362]].mean(0)
    s0 = args.px_per_mm * IPD_MM / np.linalg.norm(eye_l - eye_r)
    # Output canvas: brows region, ~70 mm wide x 45 mm tall around the brow centres.
    bc = ref[BROW["R"]["upper"] + BROW["L"]["upper"]].mean(0)
    W, H = int(80 * args.px_per_mm), int(50 * args.px_per_mm)
    T0 = np.array([[s0, 0, W / 2 - s0 * bc[0]], [0, s0, H * 0.45 - s0 * bc[1]]])  # ref px -> canvas
    ref_c = ref @ T0[:, :2].T + T0[:, 2]

    cap = cv2.VideoCapture(str(args.video))
    frames, gts = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i < len(valid) and valid[i]:
            M, _ = cv2.estimateAffinePartial2D(lm[i, RIGID].astype(np.float32), ref[RIGID].astype(np.float32))
            A = np.vstack([T0, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]])
            g = (args.red * f[:, :, 2] + (1 - args.red) * f[:, :, 1]).astype(np.float32)
            frames.append(cv2.warpAffine(g, A[:2], (W, H), flags=cv2.INTER_AREA).astype(np.uint8))
            gts.append(lm[i] @ A[:2, :2].T + A[:2, 2])
        else:
            frames.append(None)
            gts.append(None)
        i += 1
    cap.release()

    from dataclasses import fields, replace
    cfg = TrackerConfig(method=args.method)
    types = {f.name: f.type for f in fields(TrackerConfig)}
    for kv in args.cfg:
        k, v = kv.split("=")
        cfg = replace(cfg, **{k: (float(v) if types[k] in ("float", float) else int(v) if types[k] in ("int", int) else v)})
    res = {}
    pmm = args.px_per_mm
    for s in "RL":
        pts = ref_c[BROW[s]["upper"] + BROW[s]["lower"]]
        roi = (int(pts[:, 0].min() - 3 * pmm), int(pts[:, 1].min() - 14 * pmm),
               int(pts[:, 0].max() + 3 * pmm), int(pts[:, 1].max() + 8 * pmm))
        roi = (max(roi[0], 0), max(roi[1], 0), min(roi[2], W), min(roi[3], H))
        # subject's right brow is on the image left; its inner end is towards image right
        if args.v3:
            t = BrowTrackerV3(roi, cfg, inner_is_right=(s == "R"), jump_margin=args.v3_margin,
                              range_mm=args.v3_range)
        else:
            t = BrowTracker(roi, cfg, inner_is_right=(s == "R"))
        if not t.calibrate([frames[k] for k in range(a, b) if frames[k] is not None]):
            raise SystemExit(f"calibration failed for {s}")
        up0 = ref_c[BROW[s]["upper"]]
        rows = []
        for k, (img, gt) in enumerate(zip(frames, gts)):
            if img is None:
                continue
            r = t.update(img)
            dy = -(gt[BROW[s]["upper"] + BROW[s]["lower"], 1] - np.concatenate([up0, ref_c[BROW[s]["lower"]]])[:, 1])
            rows.append((k, r["raise"] / pmm, dy.mean() / pmm, r["inner"] / pmm, dy[[3, 4, 8, 9]].mean() / pmm,
                         r["outer"] / pmm, dy[[0, 1, 5, 6]].mean() / pmm, r["valid"]))
        res[s] = (t, np.array(rows))

    out_dir = OUT / "stab"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video.stem}_{'v3' if args.v3 else args.method}{args.tag}"
    summary = {}
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    for ax, s in zip(axes, "RL"):
        t, R = res[s]
        k, trk, mp = R[:, 0], R[:, 1], R[:, 2]
        neutral = (k >= a) & (k < b)
        summary[s] = {
            "r_raise": pearson(trk, mp), "r_inner": pearson(R[:, 3], R[:, 4]), "r_outer": pearson(R[:, 5], R[:, 6]),
            "slope": float(np.polyfit(mp, trk, 1)[0]), "noise_mm": float(np.std(trk[neutral])),
            "mp_span_mm": float(np.percentile(mp, 95) - np.percentile(mp, 5)),
            "rmse_after_fit_mm": float(np.sqrt(np.mean((np.polyval(np.polyfit(trk, mp, 1), trk) - mp) ** 2))),
            "valid": float(R[:, 7].mean()), "frames": int(len(R)),
        }
        fps = 60.0
        ax.plot(k / fps, mp, color="#1c7ed6", lw=1.0, label="MediaPipe brow height (mm)")
        ax.plot(k / fps, trk, color="#d9822b", lw=1.3, label="tracker raise (mm)")
        ax.axvspan(a / fps, b / fps, color="green", alpha=0.08, label="calibration (neutral)")
        ax.set_ylabel(f"brow {s}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"r={summary[s]['r_raise']:.2f}  slope={summary[s]['slope']:.2f}  "
                     f"rmse(after linear fit)={summary[s]['rmse_after_fit_mm']:.2f} mm", fontsize=9)
    axes[-1].set_xlabel("s")
    fig.suptitle(f"{args.video.name}: tracker on aligned real frames vs MediaPipe")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=90)
    plt.close(fig)
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))

    # Overlay sheet at the extremes.
    picks = [a]
    for s in "RL":
        R = res[s][1]
        picks += [int(R[np.argmax(R[:, 1]), 0]), int(R[np.argmin(R[:, 1]), 0])]
    tiles = []
    for fi in picks:
        img = cv2.cvtColor(frames[fi], cv2.COLOR_GRAY2BGR)
        for s in "RL":
            t, R = res[s]
            row = R[R[:, 0] == fi]
            cv2.polylines(img, [np.round(gts[fi][BROW[s]["upper"]]).astype(np.int32)], False, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (t.x0, t.y0), (t.x1 - 1, t.y1 - 1), (90, 90, 90), 1)
        tiles.append(img)
    cv2.imwrite(str(out_dir / f"{stem}_frames.png"), np.hstack(tiles))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
