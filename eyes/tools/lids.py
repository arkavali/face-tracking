"""Eyelid opening (blink / squint) from a small eye patch, calibrated per wearer.

The patch is anchored to the eye-sphere model's centre (so it follows mask slip), shifted
by the mean pupil offset seen during the gaze calibration (the centre itself projects
below or behind the opening, depending on the camera), and
sampled on a coarse grid (default 24 x 12 points, bilinear). Each patch is normalised
(zero mean, unit std) and a ridge regression maps it to the opening: 1 = open,
~0.5 = squint, 0 = closed. Training data come from three short prompts the wearer
follows in the AR view: "eyes open, look around", "squint, look around", "close".

On the ESP32 this is 288 bilinear samples + one 288-long dot product per frame.
"""
from __future__ import annotations

import numpy as np

LABELS = {"lidcal_open": 1.0, "lidcal_squint": 0.5, "lidcal_closed": 0.0}


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


class LidEstimator:
    def __init__(self, nx: int = 24, ny: int = 12, width_r: float = 1.3, height_r: float = 0.9,
                 ridge: float = 30.0):
        """Patch: width_r x height_r times the eye radius R (px), centred at c + offset."""
        self.nx, self.ny = nx, ny
        self.width_r, self.height_r = width_r, height_r
        self.offset = np.zeros(2)     # mean pupil position relative to the eyeball centre
        self.ridge = ridge
        self.w = None

    def patch(self, img, c, R) -> np.ndarray:
        xs = c[0] + self.offset[0] + (np.linspace(-0.5, 0.5, self.nx) * self.width_r * R)
        ys = c[1] + self.offset[1] + (np.linspace(-0.5, 0.5, self.ny) * self.height_r * R)
        X, Y = np.meshgrid(xs, ys)
        v = _bilinear(img.astype(np.float32), X, Y).ravel()
        v = v - v.mean()
        return v / (v.std() + 1e-6)

    def fit(self, patches, labels):
        X = np.asarray(patches)
        y = np.asarray(labels, float)
        self.mu = X.mean(0)
        self.ym = y.mean()
        Xc = X - self.mu
        self.w = np.linalg.solve(Xc.T @ Xc + self.ridge * np.eye(X.shape[1]), Xc.T @ (y - self.ym))
        return self

    def __call__(self, p) -> float:
        return float(np.clip((p - self.mu) @ self.w + self.ym, -0.2, 1.3))
