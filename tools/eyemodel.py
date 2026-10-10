"""Eye-sphere model from pupil ellipses (JEOresearch 3DTracker idea, made incremental).

* Every pupil ellipse's minor axis, extended, passes (nearly) through the image of the
  eyeball rotation centre. The centre is the least-squares intersection of those lines,
  each weighted by the ellipse's eccentricity (a round ellipse has no reliable minor axis).
* The sums decay slowly (forgetting factor), so the model keeps re-estimating: that is
  what follows mask slip. JEOresearch freezes the model on a key press instead.
* Radius (px): the pupil centre sits R * sin(theta) from the centre, where
  cos(theta) = minor / major; R is a decaying least-squares fit of |p - c|^2 = R^2 sin^2.
* Gaze: (p - c) / R gives the gaze components across the line of sight to the eye;
  the third component points towards the camera. Rotated to the camera frame by the
  direction of the ray through c, then to the head frame by the camera's mounting pose.

All of it is 2x2 sums and a short sorted buffer per frame - trivial for the ESP32.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class EyeModel:
    def __init__(self, f: float, w: int, h: int, half_life_s: float = 20.0, fps: float = 25.0,
                 min_ecc: float = 0.15, r_buffer: int = 300):
        self.f, self.w, self.h = f, w, h
        self.decay = 0.5 ** (1.0 / (half_life_s * fps))
        self.min_ecc = min_ecc
        self.A = np.zeros((2, 2))
        self.bv = np.zeros(2)
        self.wsum = 0.0
        self.c = None
        self.rs = deque(maxlen=r_buffer)
        self.R = None
        self.r_num = 0.0
        self.r_den = 0.0

    def update(self, e) -> None:
        ecc = 1.0 - e.b / e.a
        self.A *= self.decay
        self.bv *= self.decay
        self.wsum *= self.decay
        if ecc > self.min_ecc:
            major = np.array([np.cos(e.angle), np.sin(e.angle)])   # normal of the minor-axis line
            N = np.outer(major, major) * ecc
            p = np.array([e.cx, e.cy])
            self.A += N
            self.bv += N @ p
            self.wsum += ecc
        if self.wsum > 3.0 and np.linalg.cond(self.A) < 1e3:
            self.c = np.linalg.solve(self.A, self.bv)
        if self.c is not None:
            # |p - c|^2 = R^2 sin^2(theta): decaying least squares over frames with a clear tilt
            s2 = max(1 - (e.b / e.a) ** 2, 0)
            self.r_num *= self.decay
            self.r_den *= self.decay
            if s2 > 0.25:
                d2 = (e.cx - self.c[0]) ** 2 + (e.cy - self.c[1]) ** 2
                self.r_num += d2 * s2
                self.r_den += s2 * s2
                self.rs.append(1)
            if len(self.rs) >= 20 and self.r_den > 0:
                self.R = float(np.sqrt(self.r_num / self.r_den))

    @property
    def ready(self) -> bool:
        return self.c is not None and self.R is not None

    def gaze_cam(self, e) -> np.ndarray | None:
        """Unit gaze vector in camera axes (x right, y down, z forward), or None."""
        if not self.ready:
            return None
        v = (np.array([e.cx, e.cy]) - self.c) / self.R
        n = np.linalg.norm(v)
        if n > 0.98:
            v *= 0.98 / n
        local = np.array([v[0], v[1], -np.sqrt(1 - v @ v)])
        ray = np.array([(self.c[0] - self.w / 2 + 0.5) / self.f, (self.c[1] - self.h / 2 + 0.5) / self.f, 1.0])
        ray /= np.linalg.norm(ray)
        return _rot_z_to(ray) @ local


def _rot_z_to(d):
    """Rotation taking +z to d (minimal rotation)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d)
    s, c = np.linalg.norm(v), float(z @ d)
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def yaw_pitch(g) -> tuple:
    """Head-frame gaze vector -> (yaw, pitch) degrees, same convention as eyesim.gaze_vec."""
    return float(np.degrees(np.arctan2(g[0], g[2]))), float(np.degrees(np.arcsin(np.clip(g[1], -1, 1))))


def poly2(X):
    x, y = X[:, 0], X[:, 1]
    return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], 1)


class Calib:
    """2nd-order polynomial from 2 features (pupil px, or model yaw/pitch) to yaw/pitch."""

    def __init__(self, order: int = 2):
        self.order = order
        self.W = None

    def fit(self, X, Y, ridge: float = 1e-3):
        P = self._feat(np.asarray(X, float))
        self.mu = P[:, 1:].mean(0)
        self.sd = P[:, 1:].std(0) + 1e-9
        P = self._norm(P)
        self.W = np.linalg.solve(P.T @ P + ridge * np.eye(P.shape[1]), P.T @ np.asarray(Y, float))
        return self

    def _feat(self, X):
        x, y = X[:, 0], X[:, 1]
        if self.order == 1:
            return np.stack([np.ones_like(x), x, y], 1)
        return poly2(X)

    def _norm(self, P):
        P = P.copy()
        P[:, 1:] = (P[:, 1:] - self.mu) / self.sd
        return P

    def __call__(self, X):
        return self._norm(self._feat(np.atleast_2d(np.asarray(X, float)))) @ self.W
