"""Head + mask meshes (exported from the Fusion design, mm) and GPU ray casting.

Frame (Fusion): x = wearer's left/right, y = up, z = forward (out of the face).
data/head.bin, data/mask.bin come from fusion_mcp.py + the dump script in README.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from gpu import cp

xp = cp if cp is not None else np
DATA = Path(__file__).resolve().parent.parent / "data"


def load_mesh(name: str) -> np.ndarray:
    """Triangles (N, 3, 3) in mm."""
    raw = (DATA / f"{name}.bin").read_bytes()
    nv, nt = struct.unpack_from("<ii", raw)
    v = np.frombuffer(raw, "<f8", nv * 3, 8).reshape(-1, 3) * 10.0   # Fusion cm -> mm
    i = np.frombuffer(raw, "<i4", nt * 3, 8 + nv * 24).reshape(-1, 3)
    return v[i]


class Scene:
    def __init__(self, names=("head", "mask"), box=None):
        """box = (lo, hi) xyz corners: keep only triangles with the centroid inside."""
        tris, ids = [], []
        for k, n in enumerate(names):
            t = load_mesh(n)
            if box is not None:
                c = t.mean(1)
                t = t[np.all((c > box[0]) & (c < box[1]), axis=1)]
            tris.append(t)
            ids.append(np.full(len(t), k, np.int32))
        self.names = names
        self.tris = np.concatenate(tris)
        self.obj = np.concatenate(ids)
        t = xp.asarray(self.tris, dtype=xp.float32)
        self.v0 = t[:, 0]
        self.e1 = t[:, 1] - t[:, 0]
        self.e2 = t[:, 2] - t[:, 0]
        n = np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])
        self.normal = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def cast(self, o, d, budget=4e6):
        """Nearest hit per ray (Moller-Trumbore). Returns t (inf = miss), triangle index."""
        o = xp.asarray(o, dtype=xp.float32).reshape(-1, 3)
        d = xp.asarray(d, dtype=xp.float32).reshape(-1, 3)
        if o.shape[0] == 1:
            o = xp.broadcast_to(o, d.shape)
        T = xp.full(d.shape[0], xp.inf, xp.float32)
        I = xp.full(d.shape[0], -1, xp.int32)
        step = max(1, int(budget / len(self.v0)))
        for a in range(0, d.shape[0], step):
            oo, dd = o[a:a + step, None, :], d[a:a + step, None, :]
            p = xp.cross(dd, self.e2[None])
            det = (self.e1[None] * p).sum(-1)
            inv = 1.0 / xp.where(xp.abs(det) < 1e-9, xp.float32(1e-9), det)
            s = oo - self.v0[None]
            u = (s * p).sum(-1) * inv
            q = xp.cross(s, self.e1[None])
            v = (dd * q).sum(-1) * inv
            t = (self.e2[None] * q).sum(-1) * inv
            ok = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-3) & (xp.abs(det) >= 1e-9)
            t = xp.where(ok, t, xp.inf)
            j = xp.argmin(t, axis=1)
            T[a:a + step] = t[xp.arange(t.shape[0]), j]
            I[a:a + step] = xp.where(xp.isfinite(T[a:a + step]), j, -1)
        return T, I


def to_np(a):
    return cp.asnumpy(a) if cp is not None and isinstance(a, cp.ndarray) else np.asarray(a)
