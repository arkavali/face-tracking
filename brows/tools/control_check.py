"""Control-oriented metric: can the tracker tell a deliberate brow move from rest?

Using MediaPipe's head-frame brow height as the label (per brow, relative to neutral):
  raised  = height > +RAISE_MM, lowered = height < -LOWER_MM, rest = |height| < REST_MM
AUC (probability that a random raised frame gets a higher tracker value than a random
rest frame; same for lowered vs rest with the sign flipped). 0.5 = chance, 1.0 = perfect.
Also the accuracy of a simple threshold halfway between the medians of the two groups.
"""
from __future__ import annotations

import sys

import numpy as np

from common import OUT

RAISE_MM, LOWER_MM, REST_MM = 3.0, 2.0, 1.0


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 10 or len(neg) < 10:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def thr_acc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 10 or len(neg) < 10:
        return float("nan")
    t = (np.median(pos) + np.median(neg)) / 2
    return float(((pos > t).sum() + (neg <= t).sum()) / (len(pos) + len(neg)))


def main() -> None:
    print(f"{'config':22s} {'AUC up':>7s} {'acc up':>7s} {'AUC down':>9s} {'acc down':>9s}  brows(up/down)")
    for cfg in sys.argv[1:]:
        up_a, up_c, dn_a, dn_c = [], [], [], []
        for f in sorted((OUT / "eval" / cfg).glob("actor_*.npz")):
            d = np.load(f)
            for s in "RL":
                mp, tr = d[f"{s}_mp"], d[f"{s}_raise"]
                rest = tr[np.abs(mp) < REST_MM]
                up = tr[mp > RAISE_MM]
                dn = tr[mp < -LOWER_MM]
                up_a.append(auc(up, rest))
                up_c.append(thr_acc(up, rest))
                dn_a.append(auc(-dn, -rest))
                dn_c.append(thr_acc(-dn, -rest))
        n_up = int(np.isfinite(up_a).sum())
        n_dn = int(np.isfinite(dn_a).sum())
        print(f"{cfg:22s} {np.nanmedian(up_a):7.2f} {np.nanmedian(up_c):7.2f} {np.nanmedian(dn_a):9.2f} "
              f"{np.nanmedian(dn_c):9.2f}  {n_up}/{n_dn}")


if __name__ == "__main__":
    main()
