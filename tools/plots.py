"""Time-series plots: tracker raise vs MediaPipe, per brow, for selected emotions."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, RAVDESS_EMOTIONS


def plot(config: str, actor: int, emotions=(1, 8, 5, 4, 6), max_clips_per_emotion: int = 2) -> str:
    d = np.load(OUT / "eval" / config / f"actor_{actor:02d}.npz")
    clips = []
    for e in emotions:
        cs = sorted({c for c, em in zip(d["clip"], d["emotion"]) if em == e})
        clips += [c for c in cs[:max_clips_per_emotion] if c not in clips]
    idx = np.concatenate([np.flatnonzero(d["clip"] == c) for c in clips])
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    t = np.arange(len(idx))
    for ax, s in zip(axes, "RL"):
        ppm = float(d[f"{s}_px_per_mm"])
        ax.plot(t, d[f"{s}_raise"][idx] / ppm, color="#d9822b", lw=1.4, label="tracker raise (mm, camera view)")
        ax.plot(t, np.nanmean(d[f"{s}_gt"][idx], 1) / ppm, color="#2b8a3e", lw=1.0, label="MediaPipe contour in camera (mm)")
        ax.plot(t, d[f"{s}_mp"][idx], color="#1c7ed6", lw=1.0, ls="--", label="MediaPipe head-frame height (mm)")
        bad = ~d[f"{s}_valid"][idx].astype(bool)
        ax.fill_between(t, -6, 10, where=bad, color="red", alpha=0.12, step="mid", label="tracker invalid")
        ax.set_ylabel(f"brow {s}")
        ax.set_ylim(-6, 10)
        ax.grid(alpha=0.3)
    edges = np.flatnonzero(np.r_[True, d["clip"][idx][1:] != d["clip"][idx][:-1]])
    for ax in axes:
        for e in edges:
            ax.axvline(e, color="k", lw=0.5, alpha=0.4)
    for e in edges:
        em = int(d["emotion"][idx][e])
        axes[0].text(e + 2, 9, RAVDESS_EMOTIONS.get(em, "own video"), fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_xlabel("frame (clips concatenated)")
    fig.suptitle(f"{config} - actor {actor}")
    fig.tight_layout()
    dst = OUT / "eval" / config / f"plot_actor_{actor:02d}.png"
    fig.savefig(dst, dpi=90)
    plt.close(fig)
    return str(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--actors", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    for a in args.actors:
        print(plot(args.config, a))


if __name__ == "__main__":
    main()
