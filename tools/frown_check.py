"""How well do the tracker's outputs follow MediaPipe's brow blendshapes?

For each evaluated actor: Pearson r of
  -inner (inner end lowering)  vs browDown<Side>      (frown proxy)
  raise                         vs browOuterUp<Side>
  inner                         vs browInnerUp
MediaPipe's Left/Right naming is checked both ways (its convention vs the subject's side).
"""
from __future__ import annotations

import sys

import numpy as np

from common import BS_BROW, OUT


def r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or a[m].std() < 1e-9 or b[m].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    for cfg in sys.argv[1:]:
        rows = []
        for f in sorted((OUT / "eval" / cfg).glob("actor_*.npz")):
            d = np.load(f)
            bs = {n: d["bs"][:, i] for i, n in enumerate(BS_BROW)}
            for s, same, other in (("R", "Right", "Left"), ("L", "Left", "Right")):
                inner, rais = d[f"{s}_inner"], d[f"{s}_raise"]
                rows.append((f.stem, s,
                             r(-inner, bs[f"browDown{same}"]), r(-inner, bs[f"browDown{other}"]),
                             r(rais, bs[f"browOuterUp{same}"]), r(rais, bs[f"browOuterUp{other}"]),
                             r(inner, bs["browInnerUp"])))
        a = np.array([x[2:] for x in rows], float)
        print(f"{cfg}: median  down(same) {np.nanmedian(a[:, 0]):.2f}  down(other) {np.nanmedian(a[:, 1]):.2f}  "
              f"outerUp(same) {np.nanmedian(a[:, 2]):.2f}  outerUp(other) {np.nanmedian(a[:, 3]):.2f}  "
              f"innerUp {np.nanmedian(a[:, 4]):.2f}   ({len(rows)} brows)")


if __name__ == "__main__":
    main()
