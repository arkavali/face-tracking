"""Render the IR camera view from given placements (GPU ray casting, LED at the camera).

  python preview.py 120:130:300 66:130:340 ...     lens:y:z (x = 0, aimed at the mouth ROI)
  -> out/preview.png  (ROI points drawn: green = lips, yellow = open-jaw lower lip, red = chin)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

import sweep
from scene import Scene, to_np, xp

OUT = Path(__file__).resolve().parent.parent / "out"


def rays(cam, fwd, model, f, w=sweep.W, h=sweep.H):
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, h / 2 - (np.arange(h) + 0.5))
    r = np.hypot(u, v)
    th = r / f if model == "fisheye" else np.arctan(r / f)
    ph = np.arctan2(v, u)
    d = (np.sin(th) * np.cos(ph))[..., None] * right + (np.sin(th) * np.sin(ph))[..., None] * up \
        + np.cos(th)[..., None] * fwd
    valid = th < np.pi / 2 * 0.999
    return d.reshape(-1, 3), valid.ravel()


def render(sc, cam, fwd, model, f):
    d, valid = rays(cam, fwd, model, f)
    t, i = sc.cast(cam[None], d)
    t, i = to_np(t), to_np(i)
    hit = np.isfinite(t) & valid
    n = sc.normal[np.maximum(i, 0)]
    cos = np.abs((n * d).sum(1))
    # LED next to the lens: Lambert * 1/r^2, skin brighter than the (dark) mask lining
    alb = np.where(sc.obj[np.maximum(i, 0)] == 0, 1.0, 0.35)
    tt = np.where(hit, t, 1e9)
    img = np.where(hit, alb * cos * (60.0 / tt) ** 2, 0)
    img = np.clip(img / np.percentile(img[hit], 99) * 230, 0, 255) if hit.any() else img
    return img.reshape(sweep.H, sweep.W).astype(np.uint8)


def main():
    specs = sys.argv[1:] or ["120:125:300", "120:140:290", "66:125:340", "160:120:280"]
    sc = Scene(box=(np.array([-130, -20, 120.0]), np.array([130, 230, 420.0])))
    roi = sweep.mouth_roi(sc)
    pts = np.concatenate([roi["neutral"], roi["jaw"], roi["chin"]])
    col = [(0, 255, 0)] * len(roi["neutral"]) + [(0, 255, 255)] * len(roi["jaw"]) + [(0, 0, 255)] * len(roi["chin"])
    aim = pts.mean(0)
    tiles = []
    for s in specs:
        name, y, z = s.split(":")
        model, fov = sweep.LENSES[name]
        f = sweep.lens_f(model, fov)
        cam = np.array([0.0, float(y), float(z)])
        img = cv2.cvtColor(render(sc, cam, aim - cam, model, f), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for p, c in zip(sweep.project(pts, cam, aim - cam, model, f), col):
            cv2.circle(img, (int(p[0] * 2), int(p[1] * 2)), 2, c, -1)
        dz, dy = cam[2] - roi["slit"][2], cam[1] - roi["slit"][1]
        cv2.putText(img, f"{name} deg  {dz:.0f} mm fwd, {dy:+.0f} mm up", (8, 22), 0, 0.6, (255, 200, 0), 2)
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)])
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "preview.png"), sheet)
    print("out/preview.png")


if __name__ == "__main__":
    main()
