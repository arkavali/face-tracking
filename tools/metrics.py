"""Summarise evaluate.py results: how well the tracker follows MediaPipe's brows.

Per actor and brow:
  r_gt      Pearson r, tracker raise vs MediaPipe upper contour projected into the
            mask camera (both px, relative to neutral) - same viewpoint, same units
  rmse_mm   RMS difference of the same two signals, per strip, converted to mm
  r_mp      Pearson r, tracker raise vs MediaPipe brow height in the head frame (mm)
  noise_mm  std of the tracker raise during neutral clips (jitter at rest), mm
  span_mm   5..95 percentile span of the MediaPipe brow height (how much moved), mm
  snr       span_mm / noise_mm
  valid     share of frames with a valid tracker result
  vis       share of the upper contour visible (not hidden by holder / brow ridge)
Writes out/eval/<config>/summary.json and prints a table.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import OUT


def pearson(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def brow_stats(d, s: str) -> dict:
    ppm = float(d[f"{s}_px_per_mm"])
    raise_px = d[f"{s}_raise"]
    gt = d[f"{s}_gt"]                      # (N, strips) px, +up
    off = -d[f"{s}_off"]                   # (N, strips) px, +up
    gt_mean = np.nanmean(gt, 1)
    neutral = d["emotion"] == 1
    err = (off - gt)
    rmse = float(np.sqrt(np.nanmean(err ** 2)) / ppm)
    noise = float(np.std(raise_px[neutral]) / ppm) if neutral.any() else float("nan")
    mp = d[f"{s}_mp"]
    span = float(np.percentile(mp, 95) - np.percentile(mp, 5))
    return {
        "r_gt": pearson(raise_px, gt_mean),
        "rmse_mm": rmse,
        "r_mp": pearson(raise_px, mp),
        "r_inner": pearson(d[f"{s}_inner"], d[f"{s}_mp_inner"]),
        "r_outer": pearson(d[f"{s}_outer"], d[f"{s}_mp_outer"]),
        "noise_mm": noise,
        "span_mm": span,
        "snr": span / noise if noise and noise > 0 else float("nan"),
        "valid": float(np.mean(d[f"{s}_valid"])),
        "vis": float(np.mean(d[f"{s}_vis"])),
        "px_per_mm": ppm,
    }


def summarise(config: str) -> dict:
    root = OUT / "eval" / config
    res = {}
    for f in sorted(root.glob("actor_*.npz")):
        d = np.load(f)
        res[f.stem] = {s: brow_stats(d, s) for s in "RL"}
    keys = ["r_gt", "rmse_mm", "r_mp", "r_inner", "r_outer", "noise_mm", "span_mm", "snr", "valid", "vis"]
    agg = {}
    for k in keys:
        vals = [res[a][s][k] for a in res for s in "RL"]
        agg[k] = float(np.nanmedian(vals)) if vals else float("nan")
    out = {"config": config, "median": agg, "per_actor": res}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    cols = ["r_gt", "rmse_mm", "r_mp", "r_inner", "r_outer", "noise_mm", "span_mm", "snr", "valid", "vis"]
    print(f"{'config':28s} " + " ".join(f"{c:>8s}" for c in cols))
    for c in args.configs:
        s = summarise(c)
        print(f"{c:28s} " + " ".join(f"{s['median'][k]:8.2f}" for k in cols))
        for a, v in s["per_actor"].items():
            for side in "RL":
                print(f"   {a} {side:22s} " + " ".join(f"{v[side][k]:8.2f}" for k in cols))


if __name__ == "__main__":
    main()
