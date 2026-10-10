"""Pupil ellipse detector written with ESP32 porting in mind (no OpenCV contours).

1. Seed: darkest box on a 2x-downsampled frame (integral image), like JEOresearch's
   darkest-area search, but the box size is ~ the smallest expected pupil.
2. Starburst: rays from the seed; along each ray the first dark->bright step whose
   bright side is clearly above the pupil level is an edge point. A short bright run
   followed by dark again is a glint inside the pupil and is skipped. The seed moves to
   the mean of the edge points and the rays are cast again (2-3 rounds).
3. Ellipse: RANSAC over 5-point subsets with the Fitzgibbon direct least-squares fit,
   then a refit on the inliers. Lid edges crossing the pupil are the usual outliers.

Everything is per-ray 1-D sampling plus a 6x6 eigen-problem: cheap on an ESP32-S3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PupilParams:
    box: int = 8                  # seed box (px at full resolution), ~ min pupil diameter
    rays: int = 36
    r_min: float = 2.0
    r_max: float = 45.0           # px
    step: float = 1.0
    grad: float = 14.0            # dark->bright step over 2 px (grey levels)
    above_seed: float = 18.0      # bright side must be this much above the pupil level
    glint_level: int = 235
    rounds: int = 3
    ransac_iter: int = 40
    inlier_px: float = 1.2
    min_points: int = 10
    max_axis_ratio: float = 4.0
    min_support: float = 0.4      # inlier rays / all rays
    max_inside: float = 25.0      # mean grey inside the ellipse minus the seed level


@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float                      # semi-major
    b: float                      # semi-minor
    angle: float                  # rad, major axis direction
    support: float                # inlier fraction of the rays
    n: int
    contrast: float = 0.0         # ring around the ellipse minus the pupil level (grey levels)
    top_vis: float = 1.0          # visible part of the pupil's upward extent (1 = lid clear of it)

    @property
    def minor_dir(self):
        return np.array([-np.sin(self.angle), np.cos(self.angle)])


def darkest_seed(img: np.ndarray, box: int):
    """Centre (x, y) and mean of the darkest box, searched on a 2x-downsampled frame."""
    h, w = img.shape
    s = img[: h // 2 * 2, : w // 2 * 2].astype(np.float64)
    s = (s[0::2, 0::2] + s[1::2, 0::2] + s[0::2, 1::2] + s[1::2, 1::2]) / 4
    # glints would pull a box mean up; clip bright pixels to keep a pupil with a glint dark
    s = np.minimum(s, 120)
    k = max(box // 2, 2)
    ii = np.pad(s.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    m = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)
    # ignore a border of the frame (vignetting, mask edge)
    bd = 3
    m[:bd] = m[-bd:] = 1e9
    m[:, :bd] = m[:, -bd:] = 1e9
    y, x = np.unravel_index(np.argmin(m), m.shape)
    return (2 * x + k, 2 * y + k), float(m[y, x])


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = x.astype(int)
    y0 = y.astype(int)
    fx, fy = x - x0, y - y0
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def starburst(img: np.ndarray, seed, level: float, p: PupilParams):
    """Edge points around the seed (N, 2)."""
    h, w = img.shape
    th = np.linspace(0, 2 * np.pi, p.rays, endpoint=False)
    r = np.arange(p.r_min, p.r_max, p.step)
    xs = seed[0] + np.cos(th)[:, None] * r[None]
    ys = seed[1] + np.sin(th)[:, None] * r[None]
    inside = (xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)
    prof = _bilinear(img.astype(np.float64), xs, ys)
    pts = []
    for i in range(p.rays):
        v = prof[i]
        ok = inside[i]
        j = 3
        while j < len(r) and ok[j]:
            if v[j] >= p.glint_level:
                # glint: skip the bright run if it returns to dark within a few px
                k = j
                while k < len(r) and v[k] >= level + p.above_seed and k - j < 6:
                    k += 1
                if k < len(r) and v[k] < level + p.above_seed:
                    j = k + 1
                    continue
            if v[j] - v[j - 2] > p.grad and v[j] > level + p.above_seed:
                # sub-pixel: position of the steepest step around j
                if j + 2 > len(v):
                    pts.append((seed[0] + np.cos(th[i]) * r[j], seed[1] + np.sin(th[i]) * r[j]))
                    break
                g = v[j - 1: j + 2] - v[j - 3: j]
                k = int(np.argmax(g)) - 1  # -1..1 around j - 1
                rr = r[j - 1 + k] if 0 <= j - 1 + k < len(r) else r[j]
                pts.append((seed[0] + np.cos(th[i]) * rr, seed[1] + np.sin(th[i]) * rr))
                break
            j += 1
    return np.array(pts, float).reshape(-1, 2)


def fit_ellipse(pts: np.ndarray):
    """Fitzgibbon direct least squares; conic (A..F) or None."""
    x, y = pts[:, 0], pts[:, 1]
    mx, my = x.mean(), y.mean()
    sx = max(np.abs(x - mx).max(), 1e-6)
    sy = max(np.abs(y - my).max(), 1e-6)
    xn, yn = (x - mx) / sx, (y - my) / sy
    D1 = np.stack([xn * xn, xn * yn, yn * yn], 1)
    D2 = np.stack([xn, yn, np.ones_like(xn)], 1)
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    w, v = np.linalg.eig(M)
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    idx = np.flatnonzero(cond > 0)
    if len(idx) == 0:
        return None
    a1 = v[:, idx[0]]
    a2 = T @ a1
    A, B, C, Dd, E, F = np.concatenate([a1, a2])
    # undo the normalisation: x = (X - mx) / sx
    A2, B2, C2 = A / sx ** 2, B / (sx * sy), C / sy ** 2
    D2_, E2 = Dd / sx, E / sy
    Dn = -2 * A2 * mx - B2 * my + D2_
    En = -2 * C2 * my - B2 * mx + E2
    Fn = A2 * mx ** 2 + B2 * mx * my + C2 * my ** 2 - D2_ * mx - E2 * my + F
    return np.array([A2, B2, C2, Dn, En, Fn])


def conic_to_params(c):
    """Conic A x^2 + B xy + C y^2 + D x + E y + F = 0 -> (cx, cy, a, b, angle) or None."""
    A, B, C, D, E, F = c
    M = np.array([[A, B / 2], [B / 2, C]])
    try:
        cx, cy = np.linalg.solve(2 * M, [-D, -E])
    except np.linalg.LinAlgError:
        return None
    k = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F
    lam, vec = np.linalg.eigh(M)
    q = -k / lam
    if not (q > 0).all():
        return None
    ax = np.sqrt(q)                       # semi-axis along each eigenvector
    i = int(np.argmax(ax))
    ang = float(np.arctan2(vec[1, i], vec[0, i]))
    return float(cx), float(cy), float(ax[i]), float(ax[1 - i]), ang


def _dist(pts, prm):
    """Approximate point-to-ellipse distance (px): radial distance in the ellipse frame."""
    cx, cy, a, b, ang = prm
    ca, sa = np.cos(ang), np.sin(ang)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca
    rr = np.hypot(u / a, v / b)
    return np.abs(rr - 1) * np.hypot(u, v) / np.maximum(rr, 1e-9)


class XorShift32:
    """Tiny deterministic generator shared with the C++ port (same subsets in both)."""

    def __init__(self, seed: int = 2463534242):
        self.s = seed & 0xFFFFFFFF or 1

    def next(self) -> int:
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.s = x
        return x

    def subset(self, n: int, k: int) -> list:
        out = []
        while len(out) < k:
            i = self.next() % n
            if i not in out:
                out.append(i)
        return out


def ransac_ellipse(pts: np.ndarray, p: PupilParams, rng):
    if len(pts) < p.min_points:
        return None, None
    best, best_in = None, None
    n = len(pts)
    for _ in range(p.ransac_iter):
        sub = pts[rng.subset(n, 5)]
        c = fit_ellipse(sub)
        if c is None:
            continue
        prm = conic_to_params(c)
        if prm is None or prm[3] < 1 or prm[2] / prm[3] > p.max_axis_ratio:
            continue
        inl = _dist(pts, prm) < p.inlier_px
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = prm, inl
    if best_in is None or best_in.sum() < p.min_points * 0.7:
        return None, None
    c = fit_ellipse(pts[best_in])
    prm = conic_to_params(c) if c is not None else None
    if prm is None:
        prm = best
    return prm, best_in


class PupilDetector:
    def __init__(self, p: PupilParams | None = None, seed: int = 0):
        self.p = p or PupilParams()
        self.rng = XorShift32(2463534242 + seed)

    def detect(self, img: np.ndarray) -> Ellipse | None:
        p = self.p
        # 3x3 box blur twice (cheap, separable): tames sensor noise before the ray search
        img = img.astype(np.float64)
        for _ in range(2):
            img = (np.pad(img, ((1, 1), (0, 0)), mode="edge")[:-2] + img
                   + np.pad(img, ((1, 1), (0, 0)), mode="edge")[2:]) / 3
            img = (np.pad(img, ((0, 0), (1, 1)), mode="edge")[:, :-2] + img
                   + np.pad(img, ((0, 0), (1, 1)), mode="edge")[:, 2:]) / 3
        (sx, sy), level = darkest_seed(img, p.box)
        seed = (float(sx), float(sy))
        pts = np.zeros((0, 2))
        for _ in range(p.rounds):
            pts = starburst(img, seed, level, p)
            if len(pts) < p.min_points:
                return None
            new = tuple(np.median(pts, 0))
            if np.hypot(new[0] - seed[0], new[1] - seed[1]) < 0.5:
                seed = new
                break
            seed = new
        prm, inl = ransac_ellipse(pts, p, self.rng)
        if prm is None:
            return None
        cx, cy, a, b, ang = prm
        if a > p.r_max * 1.2 or inl.sum() < p.min_support * p.rays:
            return None
        # the inside must be dark: rejects closed-lid / lash fits
        yy, xx = np.mgrid[max(int(cy - b), 0):int(cy + b) + 1, max(int(cx - b), 0):int(cx + b) + 1]
        ok = (xx < img.shape[1]) & (yy < img.shape[0])
        if not ok.any() or img[yy[ok], xx[ok]].mean() - level > p.max_inside:
            return None
        # ring just outside the pupil (iris): a real pupil is much darker than its ring
        t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)
        u, v = 1.6 * a * np.cos(t), 1.6 * b * np.sin(t)
        ring = _bilinear(img, cx + u * ca - v * sa, cy + u * sa + v * ca)
        # upper lid over the pupil: along rays pointing up (+-35 deg), edge distance from the
        # centre relative to the ellipse's own extent in that direction
        d = pts - [cx, cy]
        up = np.degrees(np.arctan2(-d[:, 1], d[:, 0]))
        sel = np.abs(up - 90) < 35
        top = 1.0
        if sel.sum() >= 2:
            u = d[sel, 0] * ca + d[sel, 1] * sa
            v = -d[sel, 0] * sa + d[sel, 1] * ca
            rr = np.hypot(u / a, v / b)          # 1 on the ellipse, < 1 inside it
            top = float(np.clip(np.median(rr), 0, 1))
        return Ellipse(cx, cy, a, b, ang, float(inl.sum()) / p.rays, int(inl.sum()),
                       float(np.median(ring) - level), top)


def find_glints(img: np.ndarray, e: Ellipse, level: int = 230, reach: float = 3.0, max_n: int = 4):
    """Centroids of small saturated blobs within reach * major axis of the pupil centre."""
    h, w = img.shape
    r = int(reach * e.a) + 2
    x0, x1 = max(int(e.cx) - r, 0), min(int(e.cx) + r + 1, w)
    y0, y1 = max(int(e.cy) - r, 0), min(int(e.cy) + r + 1, h)
    m = img[y0:y1, x0:x1] >= level
    if not m.any():
        return []
    # tiny connected components (4-neighbour flood fill, fine for a handful of pixels)
    seen = np.zeros_like(m)
    out = []
    for yy, xx in zip(*np.nonzero(m)):
        if seen[yy, xx]:
            continue
        stack, px = [(yy, xx)], []
        seen[yy, xx] = True
        while stack:
            a, b = stack.pop()
            px.append((a, b))
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c, d = a + da, b + db
                if 0 <= c < m.shape[0] and 0 <= d < m.shape[1] and m[c, d] and not seen[c, d]:
                    seen[c, d] = True
                    stack.append((c, d))
        if len(px) <= 40:
            p = np.array(px, float)
            out.append((p[:, 1].mean() + x0, p[:, 0].mean() + y0, len(px)))
    out.sort(key=lambda g: np.hypot(g[0] - e.cx, g[1] - e.cy))
    return out[:max_n]


class PupilGate:
    """Rejects pupils that are much weaker than the recent accepted ones (lash clumps and
    the dark slit of a closed eye): ring contrast below a share of the running median, or
    too few rays on the ellipse."""

    def __init__(self, contrast_share: float = 0.45, min_support: float = 0.55, n: int = 100):
        self.share, self.min_support = contrast_share, min_support
        self.hist = []
        self.n = n

    def __call__(self, e: Ellipse | None) -> bool:
        if e is None or e.support < self.min_support:
            return False
        ref = float(np.median(self.hist)) if len(self.hist) >= 10 else 0.0
        if e.contrast < self.share * ref:
            return False
        self.hist.append(e.contrast)
        if len(self.hist) > self.n:
            self.hist.pop(0)
        return True
