"""Pupil detector on the user's phone clips (visible light, red channel).

python own_video.py <clip.mp4> <out.png> [--rot180] [--crop x y w h] [--every N]
Visible light is not the real case (IR makes the pupil much darker than the iris); this
only checks that the seed / starburst / ellipse chain survives real lids, lashes and brows.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from pupil import PupilDetector, PupilParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--rot180", action="store_true")
    ap.add_argument("--crop", type=int, nargs=4, default=None)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip)
    det = PupilDetector(PupilParams(above_seed=10, grad=8, max_inside=30))
    tiles, n, hit, k = [], 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        k += 1
        if k % a.every:
            continue
        if a.rot180:
            fr = cv2.rotate(fr, cv2.ROTATE_180)
        if a.crop:
            x, y, w, h = a.crop
            fr = fr[y:y + h, x:x + w]
        g = cv2.resize(fr[:, :, 2], (320, 240), interpolation=cv2.INTER_AREA)
        e = det.detect(g)
        n += 1
        hit += e is not None
        if len(tiles) < 24 and k % (a.every * 20) == 0:
            t = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            if e is not None:
                cv2.ellipse(t, ((e.cx, e.cy), (2 * e.a, 2 * e.b), np.degrees(e.angle)), (0, 0, 255), 1)
            tiles.append(t)
    print(f"{a.clip}: detected in {hit}/{n} frames ({100 * hit / max(n, 1):.0f}%)")
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


if __name__ == "__main__":
    main()
