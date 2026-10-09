"""Brow tracker reference implementation (numpy), written to port 1:1 to the ESP32.

Input: one 8-bit grey frame from the mask camera and a fixed per-brow ROI (the
user draws it once; the simulator places it from the neutral brow landmarks).

Calibration (neutral face, a handful of frames):
  * In each image column of the ROI find the strongest "bright above / dark below"
    step - seen from above, that is the upper edge of the brow hair against the
    forehead (the lower half of the brow turns away under the brow ridge).
  * Fit a quadratic y = c0 + c1*x + c2*x^2 to those points (robust: 2 passes,
    drop outliers). This curve "unrolls" the arched brow.

Tracking (every frame), for each of N strips along the brow:
  * Profile p[k] = mean over the strip's columns of grey[curve(x) + k, x], k in
    [-K, K]: the brow edge becomes a horizontal step at k ~ 0 when neutral.
  * Method "edge": edge strength g[k] = mean(p[k-m..k-1]) - mean(p[k..k+m-1]);
    take the maximum within +-S of the previous position, refine with a parabola.
  * Method "ncc": shift that best aligns p with the neutral profile (NCC), also
    limited to +-S around the previous shift.
  * Low contrast / low correlation -> keep the previous value (flag invalid).
Outputs per brow: offset of every strip (px, negative = up), the mean ("raise"),
inner/outer strip means, and a validity flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerConfig:
    method: str = "edge"     # "edge" | "ncc"
    strips: int = 5
    # Window sizes in rows. 0 = derive from the ROI height at calibration: the ROI
    # covers ~30 mm vertically (brow + 14 mm above + 8 mm below), so ROI/30 ~ px/mm.
    half_range: int = 0      # K: profile spans curve-K .. curve+K rows (~12 mm)
    edge_half: int = 0       # m: edge filter half width (~1.2 mm)
    search: int = 0          # S: max change per frame (~2.5 mm)
    ncc_half: int = 0        # neutral template rows each side of the edge (~4 mm)
    smooth: int = 1          # box radius for profile smoothing
    min_edge: float = 4.0    # minimum edge strength (grey levels)
    min_ncc: float = 0.5
    range_mm: float = 15.0   # profile half-range when half_range is derived from the ROI
    cand_frac: float = 0.25  # calibration: edge candidates down to this share of the column's strongest
    lost_frames: int = 15    # invalid this many frames in a row -> search the whole range again


def _box(p: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return p
    k = np.ones(2 * r + 1) / (2 * r + 1)
    pad = np.pad(p, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def _edge_strength(p: np.ndarray, m: int) -> np.ndarray:
    """g[k] = mean(p[k-m:k]) - mean(p[k:k+m]); positive where it gets darker going down."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    n = len(p)
    g = np.full(n, -np.inf)
    k = np.arange(m, n - m + 1)
    above = (c[k] - c[k - m]) / m
    below = (c[k + m] - c[k]) / m
    g[k] = above - below
    return g


def _parabola(g: np.ndarray, i: int) -> float:
    if 0 < i < len(g) - 1 and np.isfinite(g[i - 1]) and np.isfinite(g[i + 1]):
        den = g[i - 1] - 2 * g[i] + g[i + 1]
        if den < 0:
            return i + 0.5 * (g[i - 1] - g[i + 1]) / den
    return float(i)


class BrowTracker:
    def __init__(self, roi: tuple, cfg: TrackerConfig = TrackerConfig(), inner_is_right: bool = True):
        from dataclasses import replace
        self.x0, self.y0, self.x1, self.y1 = (int(v) for v in roi)
        ppm = max(self.y1 - self.y0, 10) / 30.0
        self.cfg = replace(
            cfg,
            half_range=cfg.half_range or max(12, int(round(cfg.range_mm * ppm))),
            edge_half=cfg.edge_half or max(2, int(round(1.2 * ppm))),
            search=cfg.search or max(4, int(round(2.5 * ppm))),
            ncc_half=cfg.ncc_half or max(5, int(round(4 * ppm))),
        )
        self.inner_is_right = inner_is_right   # which ROI side is towards the nose
        self.curve = None
        self.prev = None
        self.templates = None
        self.lost = 0

    # ------------------------------------------------------------------ calibration
    def _topmost_brow_edge(self, p: np.ndarray, g: np.ndarray):
        """Index of the TOPMOST bright->dark step that is at least cand_frac of the strongest
        one and has a brow-thick (~3.6 mm) dark band below it, or None.

        The brow is the first dark band below the forehead; the glasses rim, the eyelid
        crease and the eye are all below it and often have MORE contrast (a black rim
        can be twice as strong), so "strongest edge" picks the wrong thing. Thin forehead
        wrinkles fail the band test. The ROI must start below the hairline.
        """
        band = 3 * self.cfg.edge_half
        fin = np.isfinite(g)
        if not fin.any():
            return None
        gmax = g[fin].max()
        if gmax < self.cfg.min_edge:
            return None
        thr = max(self.cfg.min_edge, self.cfg.cand_frac * gmax)
        for i in range(1, len(g) - 1):
            if not (fin[i] and g[i] >= thr and g[i] >= g[i - 1] and g[i] >= g[i + 1]):
                continue
            if i - band < 0 or i + band > len(p):
                continue
            if p[i - band:i].mean() - p[i:i + band].mean() >= 0.6 * g[i]:
                return i
        return None

    def _column_edges(self, img: np.ndarray):
        """Per column: the topmost brow-like step (see _topmost_brow_edge)."""
        roi = img[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        xs, ys, ws = [], [], []
        m = self.cfg.edge_half
        for j in range(roi.shape[1]):
            p = _box(roi[:, j], self.cfg.smooth)
            g = _edge_strength(p, m)
            pick = self._topmost_brow_edge(p, g)
            if pick is not None:
                xs.append(self.x0 + j)
                ys.append(self.y0 + _parabola(g, pick))
                ws.append(g[pick])
        return np.array(xs, float), np.array(ys, float), np.array(ws, float)

    def calibrate(self, frames: list) -> bool:
        """Fit the brow curve on neutral frames (median image)."""
        img = np.median(np.stack(frames), axis=0)
        xs, ys, ws = self._column_edges(img)
        if len(xs) < 8:
            return False
        keep = np.ones(len(xs), bool)
        for _ in range(2):
            c = np.polyfit(xs[keep], ys[keep], 2, w=ws[keep])
            res = np.abs(np.polyval(c, xs) - ys)
            keep = res < max(2.0, 2.5 * np.median(res[keep]))
        self.curve = c
        self.cx0, self.cx1 = int(xs[keep].min()), int(xs[keep].max()) + 1
        edges = np.linspace(self.cx0, self.cx1, self.cfg.strips + 1).round().astype(int)
        self.strip_x = [(edges[i], edges[i + 1]) for i in range(self.cfg.strips)]
        self.prev = np.zeros(self.cfg.strips)
        self.templates = [self._profile(img, s) for s in range(self.cfg.strips)]
        self.ref_strength = np.array([
            max(_edge_strength(_box(t, self.cfg.smooth), self.cfg.edge_half)[self.cfg.half_range], 1.0)
            for t in self.templates])
        return True

    # ------------------------------------------------------------------ tracking
    def _profile(self, img: np.ndarray, s: int) -> np.ndarray:
        """Mean grey along the brow curve, rows curve(x)-K .. curve(x)+K, for strip s."""
        K = self.cfg.half_range
        a, b = self.strip_x[s]
        xs = np.arange(a, b)
        yc = np.polyval(self.curve, xs)
        rows = yc[None, :] + np.arange(-K, K + 1)[:, None]          # (2K+1, w)
        r0 = np.floor(rows).astype(int)
        f = rows - r0
        h = img.shape[0]
        r0c = np.clip(r0, 0, h - 1)
        r1c = np.clip(r0 + 1, 0, h - 1)
        im = img.astype(np.float32)
        v = im[r0c, xs[None, :]] * (1 - f) + im[r1c, xs[None, :]] * f   # bilinear in y
        return v.mean(1)

    def update(self, img: np.ndarray) -> dict:
        cfg = self.cfg
        K, S = cfg.half_range, cfg.search
        off = np.zeros(cfg.strips)
        ok = np.zeros(cfg.strips, bool)
        score = np.zeros(cfg.strips)
        for s in range(cfg.strips):
            p = _box(self._profile(img, s), cfg.smooth)
            margin = max(cfg.edge_half, cfg.ncc_half if cfg.method == "ncc" else 0)
            c = int(np.clip(round(self.prev[s]) + K, margin, 2 * K - margin))
            lo = max(margin, c - S)
            hi = min(2 * K + 1 - margin, c + S + 1)
            if cfg.method == "edge":
                g = _edge_strength(p, cfg.edge_half)
                i = lo + int(np.argmax(g[lo:hi]))
                pos = _parabola(g, i) - K
                score[s] = g[i]
                ok[s] = g[i] > max(cfg.min_edge, 0.35 * self.ref_strength[s])
            else:
                t = self.templates[s][K - cfg.ncc_half:K + cfg.ncc_half + 1]
                t = (t - t.mean()) / (t.std() + 1e-6)
                best, bi, vals = -2.0, lo, {}
                for cc in range(lo, hi):
                    w = p[cc - cfg.ncc_half:cc + cfg.ncc_half + 1]
                    v = float(np.mean(t * (w - w.mean()) / (w.std() + 1e-6)))
                    vals[cc] = v
                    if v > best:
                        best, bi = v, cc
                g = np.array([vals.get(c, -np.inf) for c in range(2 * K + 1)])
                pos = _parabola(g, bi) - K
                score[s] = best
                ok[s] = best > cfg.min_ncc
            if ok[s]:
                off[s] = pos
            else:
                off[s] = self.prev[s]
        # The strips belong to one brow: a strip far from the others has jumped to some
        # other edge (glasses rim, wrinkle, eyelid) - pull it back to the group.
        if ok.sum() >= 3:
            med = np.median(off[ok])
            far = np.abs(off - med) > max(3.0, 0.5 * K)
            ok &= ~far
            off[far] = med
        # Lost for a while (occlusion, blink, mask shifted): look for the brow again over
        # the whole range with the calibration rule instead of trusting the old position.
        self.lost = self.lost + 1 if ok.mean() < 0.5 else 0
        if cfg.lost_frames and self.lost >= cfg.lost_frames:
            for s in range(cfg.strips):
                p = _box(self._profile(img, s), cfg.smooth)
                g = _edge_strength(p, cfg.edge_half)
                i = self._topmost_brow_edge(p, g)
                if i is not None:
                    off[s] = _parabola(g, i) - K
                    ok[s] = True
            self.lost = 0
        self.prev = off.copy()
        n = cfg.strips
        inner = slice(n - 2, n) if self.inner_is_right else slice(0, 2)
        outer = slice(0, 2) if self.inner_is_right else slice(n - 2, n)
        w = ok.astype(float) + 1e-3
        return {
            "off": off, "ok": ok, "score": score,
            "raise": float(-(off * w).sum() / w.sum()),          # px, positive = up
            "inner": float(-off[inner].mean()),
            "outer": float(-off[outer].mean()),
            "valid": bool(ok.mean() >= 0.5),
        }

    def curve_points(self, offset: np.ndarray | None = None):
        """Polyline of the tracked edge for drawing (per strip, shifted by offset)."""
        pts = []
        for s, (a, b) in enumerate(self.strip_x):
            xs = np.arange(a, b, max(1, (b - a) // 4))
            ys = np.polyval(self.curve, xs) + (0 if offset is None else offset[s])
            pts.append(np.stack([xs, ys], 1))
        return pts
