"""Simple mouth shapes on the Fusion head mesh: smooth vertex displacements around the lips.

  shapes = make_shapes()      -> {name: (V, 3) vertices}, faces, lip weight per vertex
Not anatomy, just the visible geometry the camera and the LEDs care about: lips pushed forward
and gathered (pucker), opened into an "o" (funnel), corners pulled up and back (smile), lower face
swung down about the jaw joint (jaw open, leaving a dark gap), cheeks pushed out (puff).
Mouth slit, from the mesh (sweep.mouth_roi): x = 0, y = 99, z = 246 mm.
"""
from __future__ import annotations

import numpy as np

from scene import load_indexed

SLIT = np.array([0.0, 99.0, 246.0])
PIVOT = np.array([0.0, SLIT[1] + 30, SLIT[2] - 85])   # approximate jaw joint axis (x)


def smooth(t):
    """1 at t <= 0, 0 at t >= 1, smoothstep in between."""
    t = np.clip(t, 0, 1)
    return 1 - t * t * (3 - 2 * t)


def front(v):
    return v[:, 2] > SLIT[2] - 45


def lip_weight(v, half_w=27.0, half_h=13.0):
    dx, dy = v[:, 0] - SLIT[0], v[:, 1] - SLIT[1]
    return smooth((np.abs(dx) - half_w * 0.6) / (half_w * 0.6)) * smooth((np.abs(dy) - half_h * 0.5) / (half_h * 0.6)) * front(v)


def rot_x(p, ang, piv):
    c, s = np.cos(ang), np.sin(ang)
    q = p - piv
    return np.stack([q[:, 0], c * q[:, 1] - s * q[:, 2], s * q[:, 1] + c * q[:, 2]], 1) + piv


def jaw_open(v, deg):
    """Lower face (below the slit) swung down; full in the middle, fading towards the ears."""
    dy = v[:, 1] - SLIT[1]
    w = (smooth((np.abs(v[:, 0]) - 45) / 25) * (1 - smooth(-dy / 3)) * front(v))[:, None]
    return v + w * (rot_x(v, np.radians(deg), PIVOT) - v)   # +angle: chin down and back


def pucker(v, fwd=10.0, gather=0.4, up=0.0):
    w = lip_weight(v)
    out = v.copy()
    out[:, 0] -= gather * (v[:, 0] - SLIT[0]) * w
    out[:, 1] -= 0.25 * gather * (v[:, 1] - SLIT[1]) * w - up * w
    out[:, 2] += fwd * w
    return out


def lips_part(v, gap, half_w=22.0):
    """Upper lip up, lower lip down by gap/2 each (an opening between them)."""
    w = lip_weight(v, half_w=half_w, half_h=10)
    out = v.copy()
    out[:, 1] += np.sign(v[:, 1] - SLIT[1]) * gap / 2 * w
    return out


def smile(v, amount=1.0):
    out = v.copy()
    for side in (-1, 1):
        c = np.array([side * 25.0, SLIT[1], SLIT[2] - 6])
        d = np.linalg.norm((v - c) * [1, 1.2, 1], axis=1)
        w = smooth((d - 6) / 18) * front(v) * amount
        out[:, 0] += side * 5 * w
        out[:, 1] += 5 * w
        out[:, 2] -= 4 * w
    return out


def cheeks(v, amount=7.0):
    out = v.copy()
    for side in (-1, 1):
        c = np.array([side * 45.0, SLIT[1] + 8, SLIT[2] - 25])
        d = np.linalg.norm(v - c, axis=1)
        w = smooth((d - 8) / 22) * (v[:, 0] * side > 15)
        n = (v - c) / np.maximum(d[:, None], 1e-6)
        out += amount * w[:, None] * n * [1, 0.3, 1]
    return out


def make_shapes():
    v, f = load_indexed("head")
    shapes = {
        "neutral": v,
        "jaw": jaw_open(v, 16),
        "smile": smile(v),
        "pucker": pucker(v),
        "funnel": jaw_open(lips_part(pucker(v, fwd=7, gather=0.3), 8, half_w=15), 5),
        "pucker_up": pucker(v, up=5),
        "cheeks": cheeks(v),
    }
    return shapes, f, lip_weight(v)


def vertex_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
