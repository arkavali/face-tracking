"""Camera / LED placements for one (right) eye, in the eye frame of eyesim.py (mm).

Mask geometry used (from brow_tracker's Rig, the user's CAD sections of 2026-10-09):
the AR mirror frame is ~50 mm wide, 58 mm tall, its top ~29 mm above the eye-corner line
and its face ~31.7 mm in front of the brow skin. The brow skin is taken ~20 mm in front
of the eyeball centre (~6.5 mm ahead of the corneal apex), so the mirror plane is at
z ~ 52 and its bottom edge at y ~ -29. These are estimates: the real AR combiner pose
(tilt, coating side) still has to be confirmed against the CAD.

Placements
* under_*: camera below the mirror frame, between the frame and the face, looking up.
* thru_*: camera behind the semi-transparent combiner, looking almost straight at the eye.
"""
from __future__ import annotations

import numpy as np

from eyesim import Camera, Plate, Scene

PLATE_Z = 52.0
APEX = np.array([0.0, 0.0, 13.5])


def _target():
    # aim a bit behind the apex so the whole opening is centred
    return np.array([0.0, 0.0, 10.0])


def make(name: str, screen_level: float = 0.0, seed: int = 1):
    """(Scene, Camera, info) for a placement name."""
    sc = Scene(seed=seed, screen_level=screen_level)
    info = {}
    if name == "under_steep":          # tucked under the frame, close to the face
        cam = Camera.looking_at([2.0, -27.0, 34.0], _target())
        sc.leds = [((-12.0, -24.0, 30.0), 0.5), ((14.0, -24.0, 30.0), 0.5)]
    elif name == "under":              # under the frame bottom edge, ~40 deg up
        cam = Camera.looking_at([2.0, -27.0, 42.0], _target())
        sc.leds = [((-12.0, -25.0, 38.0), 0.5), ((14.0, -25.0, 38.0), 0.5)]
    elif name == "under_far":          # at the plate, bottom edge
        cam = Camera.looking_at([2.0, -26.0, 50.0], _target())
        sc.leds = [((-12.0, -25.0, 46.0), 0.5), ((14.0, -25.0, 46.0), 0.5)]
    elif name.startswith("thru"):
        # behind the combiner; LEDs either in front of it (at the frame bottom) or next to
        # the camera behind it (light passes the plate twice + a ghost of the LED)
        cam = Camera.looking_at([4.0, -8.0, 62.0], _target())
        sc.plate = Plate(np.array([0.0, 0.0, PLATE_Z]), np.array([0.0, 0.0, 1.0]))
        if name == "thru_ledbehind":
            sc.plate_leds_behind = True
            sc.leds = [((-8.0, -14.0, 60.0), 0.5), ((14.0, -14.0, 60.0), 0.5)]
        else:
            sc.leds = [((-12.0, -26.0, 46.0), 0.5), ((14.0, -26.0, 46.0), 0.5)]
    else:
        raise ValueError(name)
    v = APEX - cam.pos
    info["dist_mm"] = float(np.linalg.norm(v))
    info["elev_deg"] = float(np.degrees(np.arctan2(-v[1], np.hypot(v[0], v[2]))))
    return sc, cam, info


NAMES = ["under_steep", "under", "under_far", "thru", "thru_ledbehind"]
