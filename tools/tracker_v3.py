"""Brow tracker v3 (experimental): wider range and re-acquisition after lock-on.

Changes over tracker.BrowTracker ("ncc" method only):
  * Larger profile half-range (16 mm instead of 12) so a full raise (~10 mm) stays
    inside the NCC search range.
  * Every frame the NCC is evaluated over the whole profile, not just +-S around the
    previous position. Candidates = local NCC maxima above a threshold whose edge has a
    brow-thick dark band below it (a glasses rim or an eyelid crease is thin).
  * Anatomy prior: the brow is the TOPMOST such dark band in the ROI (the glasses rim,
    the eyelid and the eye are all below it). If the previous position is still a
    candidate the tracker keeps following it locally (smooth); otherwise, or if a
    clearly better candidate sits above it, it jumps to the topmost candidate.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from tracker import BrowTracker, TrackerConfig, _box, _parabola


class BrowTrackerV3(BrowTracker):
    def __init__(self, roi: tuple, cfg: TrackerConfig = TrackerConfig(method="ncc"), inner_is_right: bool = True,
                 reacquire: str = "lost", range_mm: float = 16.0, jump_margin: float = 0.15):
        ppm = max(int(roi[3]) - int(roi[1]), 10) / 30.0
        if not cfg.half_range:
            cfg = replace(cfg, half_range=max(12, int(round(range_mm * ppm))))
        super().__init__(roi, replace(cfg, method="ncc"), inner_is_right)
        self.reacquire = reacquire
        self.jump_margin = jump_margin

    def calibrate(self, frames: list) -> bool:
        if not super().calibrate(frames):
            return False
        K, h = self.cfg.half_range, self.cfg.ncc_half
        self._tz = []
        self.ref_contrast = []
        band = 3 * self.cfg.edge_half
        for t in self.templates:
            tt = t[K - h:K + h + 1]
            self._tz.append((tt - tt.mean()) / (tt.std() + 1e-6))
            self.ref_contrast.append(max(t[K - band:K].mean() - t[K:K + band].mean(), 1.0))
        return True

    def _ncc_all(self, p: np.ndarray, s: int) -> np.ndarray:
        """NCC of the neutral template against every window position of profile p."""
        h = self.cfg.ncc_half
        n = len(p)
        win = np.lib.stride_tricks.sliding_window_view(p, 2 * h + 1)          # (n-2h, 2h+1)
        wz = (win - win.mean(1, keepdims=True)) / (win.std(1, keepdims=True) + 1e-6)
        c = np.full(n, -1.0)
        c[h:n - h] = wz @ self._tz[s] / (2 * h + 1)
        return c

    def _band_ok(self, p: np.ndarray, k: int, s: int) -> bool:
        band = 3 * self.cfg.edge_half
        if k - band < 0 or k + band > len(p):
            return False
        return p[k - band:k].mean() - p[k:k + band].mean() >= 0.4 * self.ref_contrast[s]

    def update(self, img: np.ndarray) -> dict:
        cfg = self.cfg
        K, S = cfg.half_range, cfg.search
        off = np.zeros(cfg.strips)
        ok = np.zeros(cfg.strips, bool)
        score = np.zeros(cfg.strips)
        for s in range(cfg.strips):
            p = _box(self._profile(img, s), cfg.smooth)
            c = self._ncc_all(p, s)
            thr = cfg.min_ncc
            peaks = [k for k in range(1, len(c) - 1)
                     if c[k] >= thr and c[k] >= c[k - 1] and c[k] >= c[k + 1] and self._band_ok(p, k, s)]
            prev_k = int(round(self.prev[s])) + K
            lo, hi = max(0, prev_k - S), min(len(c), prev_k + S + 1)
            loc = lo + int(np.argmax(c[lo:hi]))
            if not peaks:
                off[s], ok[s], score[s] = self.prev[s], False, c[loc]
                continue
            best = max(peaks, key=lambda k: c[k])
            local_ok = c[loc] >= thr and self._band_ok(p, loc, s)
            if self.reacquire == "topmost":
                top = min(peaks)
                if local_ok and (loc <= top + S or c[loc] >= c[top] + 0.1):
                    k = loc
                elif c[top] >= 0.8 * c[best]:
                    k = top
                else:
                    k = best
            else:
                # "lost": follow locally while the local match is still brow-like and
                # not clearly beaten; otherwise jump to the best brow-like match.
                if local_ok and c[loc] >= c[best] - self.jump_margin:
                    k = loc
                else:
                    k = best
            off[s] = _parabola(c, k) - K
            ok[s] = True
            score[s] = c[k]
        if ok.sum() >= 3:
            med = np.median(off[ok])
            far = np.abs(off - med) > max(3.0, 0.5 * K)
            ok &= ~far
            off[far] = med
        self.prev = off.copy()
        n = cfg.strips
        inner = slice(n - 2, n) if self.inner_is_right else slice(0, 2)
        outer = slice(0, 2) if self.inner_is_right else slice(n - 2, n)
        w = ok.astype(float) + 1e-3
        return {"off": off, "ok": ok, "score": score,
                "raise": float(-(off * w).sum() / w.sum()),
                "inner": float(-off[inner].mean()), "outer": float(-off[outer].mean()),
                "valid": bool(ok.mean() >= 0.5)}
