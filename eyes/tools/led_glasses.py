"""Where do the LEDs' reflections in a spectacle lens land, relative to the eye?

For a camera placement and a set of LED-pair candidates, over a gaze grid: how often
both corneal glints are visible, and how close the nearest lens reflection comes to the
pupil (in pupil-free image distance, px). A lens reflection within ~3 iris radii of the
pupil confuses the glint search and can cover the pupil.
python led_glasses.py [placement]   (default thru_ledside)
"""
from __future__ import annotations

import sys

import numpy as np

import eyesim as E
from placements import make

CANDIDATES = {
    "sides, eye level, z45": [(-26, 0, 45), (26, 0, 45)],
    "sides, eye level, z35": [(-26, 0, 35), (26, 0, 35)],
    "sides, wide, z40": [(-32, 0, 40), (32, 0, 40)],
    "sides, low, z45": [(-26, -12, 45), (26, -12, 45)],
    "sides, high, z45": [(-26, 12, 45), (26, 12, 45)],
    "top corners, z45": [(-24, 22, 45), (24, 22, 45)],
    "bottom corners, z45": [(-24, -24, 45), (24, -24, 45)],
    "next to camera (behind mirror)": [(-8, -14, 60), (14, -14, 60)],
    "inside the lens gap, sides z20": [(-22, 0, 20), (22, 0, 20)],
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "thru_ledside"
    grid = [(y, p) for y in range(-25, 26, 10) for p in range(-15, 16, 10)]
    print(f"{name} with glasses (4% per surface); {len(grid)} gaze directions")
    print(f"{'LEDs':34s} 2 corneal glints  lens refl. visible  min dist to pupil (median, px)")
    for label, leds in CANDIDATES.items():
        sc, cam, _ = make(name)
        sc.lids.up_open, sc.lids.lo_open = 19.0, 10.0
        sc.glasses = E.Glasses()
        sc.leds = [(np.array(p, float), 0.5) for p in leds]
        sc.lash_up = sc.lash_lo = 0
        r = E.Renderer(sc, cam, ss=1)
        two, vis, dists = 0, 0, []
        for y, p in grid:
            st = E.EyeState(y, p)
            _, gt = r.render(st)
            two += len(gt["glints"]) >= 2
            lens = r._lens(r.O, r.D, cam)
            m = lens["spec"] > 0
            if m.any() and "pc_full" in gt:
                vis += 1
                ys, xs = np.nonzero(m)
                dists.append(float(np.min(np.hypot(xs - gt["pc_full"][0], ys - gt["pc_full"][1]))))
        md = f"{np.median(dists):6.0f}" if dists else "     -"
        print(f"{label:34s} {two / len(grid):16.0%}  {vis / len(grid):17.0%}  {md}")


if __name__ == "__main__":
    main()
