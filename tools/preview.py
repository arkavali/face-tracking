"""Contact sheet: every placement x a few eye states.

python preview.py [out.png] [--screen 1.0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from eyesim import EyeState, Look, Renderer
from placements import NAMES, make

STATES = [
    ("centre", EyeState()),
    ("left 25", EyeState(yaw=-25)),
    ("nasal 25", EyeState(yaw=25)),
    ("up 20", EyeState(pitch=20)),
    ("down 20", EyeState(pitch=-20)),
    ("squint", EyeState(openness=0.55, squint=0.7)),
    ("blink", EyeState(openness=0.05)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="../out/preview.png")
    ap.add_argument("--screen", type=float, default=0.0)
    ap.add_argument("--names", nargs="*", default=NAMES)
    a = ap.parse_args()
    rows = []
    for name in a.names:
        sc, cam, info = make(name, a.screen)
        r = Renderer(sc, cam)
        tiles = []
        for label, st in STATES:
            img, gt = r.render(st, look=Look())
            t = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if "pc_full" in gt:
                cv2.circle(t, tuple(int(round(v)) for v in gt["pc_full"]), 2, (0, 0, 255), -1)
            cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(t)
        row = np.hstack(tiles)
        head = np.zeros((22, row.shape[1], 3), np.uint8)
        cv2.putText(head, f"{name}: {info['dist_mm']:.0f} mm, {info['elev_deg']:.0f} deg below",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([head, row]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main()
