"""IR LED layout around the mouth: render the camera view of several mouth shapes for many LED
positions (shadows, wet-lip highlights, LED beam), then score each layout by how well the
shapes stay apart in the 80x60 net input compared with the spread caused by the mask sitting a
few mm differently.

  python ledsim.py            -> out/led/report.txt, out/led/best.png, out/led/leds.npz
Light adds linearly, so every candidate LED is rendered alone once per (shape, mask shift) and
layouts are sums of those images, then auto-exposed (98th percentile -> white, clipped).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import cv2
import numpy as np

import lipshapes
import rt
import sweep
from gpu import cp
from scene import Scene, to_np

OUT = Path(__file__).resolve().parent.parent / "out" / "led"
W, H = 160, 120
CAM = np.array([0.0, 129.0, 301.0])            # placement sweep: ~55 mm in front, 30 mm above
AIM = np.array([0.0, 85.0, 236.0])
LENS = sweep.LENSES["120"]
SHIFTS = [np.zeros(3)] + [s * 5.0 * e for e in np.eye(3) for s in (1, -1)]
PAIRS = [("neutral", "pucker"), ("pucker", "funnel"), ("jaw", "pucker"), ("jaw", "funnel"),
         ("neutral", "smile"), ("neutral", "cheeks"), ("pucker", "pucker_up"), ("smile", "pucker")]
SKIN, LINING, SPEC, SHININESS, BEAM_M = 0.6, 0.25, 0.35, 40.0, 2.0
BOX = (np.array([-120, -10, 150.0]), np.array([120, 220, 420.0]))


def camera_rays(cam, aim):
    model, fov = LENS
    f = sweep.lens_f(model, fov) * W / sweep.W
    fwd = (aim - cam) / np.linalg.norm(aim - cam)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, H / 2 - (np.arange(H) + 0.5))
    r = np.hypot(u, v)
    th = r / f
    ph = np.arctan2(v, u)
    d = (np.sin(th) * np.cos(ph))[..., None] * right + (np.sin(th) * np.sin(ph))[..., None] * up \
        + np.cos(th)[..., None] * fwd
    return d.reshape(-1, 3)


class ShapeScene:
    """Deformed head + static mask on the GPU, with smooth head normals."""

    def __init__(self, verts, faces, lipw, mask_tris, base=None, fast_shadows=False):
        c = verts[faces].mean(1)
        keep = np.all((c > BOX[0]) & (c < BOX[1]), axis=1)
        if base is not None:
            # an opened mouth: triangles stretched across the gap would cover it like a curtain
            def edges(v):
                t = v[faces]
                return np.stack([np.linalg.norm(t[:, a] - t[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1).max(1)
            e0, e1 = edges(base), edges(verts)
            keep &= ~((e1 > 2.5 * e0) & (e1 > 4.0))
        f = faces[keep]
        vn = lipshapes.vertex_normals(verts, faces)
        mt = mask_tris[np.all((mask_tris.mean(1) > BOX[0]) & (mask_tris.mean(1) < BOX[1]), axis=1)]
        tri = np.concatenate([verts[f], mt]).astype(np.float32)
        self.n_head = len(f)
        self.v0 = cp.asarray(tri[:, 0]); self.e1 = cp.asarray(tri[:, 1] - tri[:, 0]); self.e2 = cp.asarray(tri[:, 2] - tri[:, 0])
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        # per-corner normals: smooth on the head, flat on the mask
        cn = np.concatenate([vn[f], np.repeat(fn[self.n_head:, None], 3, 1)]).astype(np.float32)
        self.cn = cp.asarray(cn)
        self.fn = cp.asarray(fn.astype(np.float32))
        lw = np.concatenate([lipw[f].mean(1), np.zeros(len(mt))]).astype(np.float32)
        self.lipw = cp.asarray(lw)
        self.is_head = cp.asarray(np.r_[np.ones(self.n_head, bool), np.zeros(len(mt), bool)])
        # shadow rays: by default all triangles; fast_shadows keeps only the front of the face (the
        # LEDs sit in the beak cavity, so nose / lips / cheeks are the only possible occluders)
        if fast_shadows:
            hc = tri[:self.n_head].mean(1)
            occ = tri[:self.n_head][(hc[:, 2] > lipshapes.SLIT[2] - 35) & (np.abs(hc[:, 0]) < 85)]
            self.s0 = cp.asarray(occ[:, 0]); self.s1 = cp.asarray(occ[:, 1] - occ[:, 0]); self.s2 = cp.asarray(occ[:, 2] - occ[:, 0])
        else:
            self.s0, self.s1, self.s2 = self.v0, self.e1, self.e2

    def primary(self, cam, dirs):
        t, i, u, v = rt.cast(self.v0, self.e1, self.e2, cam[None], dirs)
        hit = i >= 0
        j = cp.maximum(i, 0)
        d = cp.asarray(dirs, dtype=cp.float32)
        p = cp.asarray(cam, dtype=cp.float32)[None] + d * cp.where(hit, t, 0)[:, None]
        w = cp.stack([1 - u - v, u, v], 1)
        n = (self.cn[j] * w[:, :, None]).sum(1)
        n /= cp.maximum(cp.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        inside = (self.fn[j] * d).sum(1) > 0          # looking at a back face: inside the mouth / head
        alb = cp.where(self.is_head[j], SKIN, LINING) * (~inside) * hit
        n = cp.where(((n * d).sum(1) > 0)[:, None], -n, n)
        return dict(p=p, n=n, d=d, alb=alb, spec=self.lipw[j] * (~inside) * hit, hit=hit)

    def shade(self, g, led, led_aim):
        """Radiance from one LED (point source, cos^m beam towards led_aim), with shadows."""
        L = cp.asarray(led, dtype=cp.float32)[None] - g["p"]
        dist = cp.linalg.norm(L, axis=1)
        l = L / cp.maximum(dist[:, None], 1e-6)
        ndl = cp.clip((g["n"] * l).sum(1), 0, None)
        o = g["p"] + g["n"] * 0.15
        t, i, _, _ = rt.cast(self.s0, self.s1, self.s2, o, l, tmax=dist - 0.5)
        lit = (i < 0) & g["hit"] & (ndl > 0)
        ax = led_aim - led; ax = ax / np.linalg.norm(ax)
        beam = cp.clip((-l * cp.asarray(ax, dtype=cp.float32)[None]).sum(1), 0, None) ** BEAM_M
        h = l - g["d"]
        h /= cp.maximum(cp.linalg.norm(h, axis=1, keepdims=True), 1e-6)
        spec = SPEC * g["spec"] * cp.clip((g["n"] * h).sum(1), 0, None) ** SHININESS
        e = beam * lit / cp.maximum(dist, 5.0) ** 2 * 1e4
        return (g["alb"] * ndl + spec) * e


def candidates(mask_sc, face):
    """LED positions in the free space between face and beak (x >= 0; mirrored for pairs)."""
    xs_, ys_, fz = face
    pts = []
    for x in (0.0, 15.0, 30.0, 45.0):
        for y in (70.0, 90.0, 110.0, 130.0, 150.0):
            for z in (270.0, 290.0, 310.0, 330.0):
                fzv = fz[np.argmin(abs(ys_ - y)), np.argmin(abs(xs_ - x))]
                if np.isfinite(fzv) and z < fzv + 8:
                    continue
                t, _ = mask_sc.cast(np.array([[x, y, z]]), np.array([[0, 0, 1.0]]))
                t = float(to_np(t)[0])
                if not np.isfinite(t) or t < 6:          # outside the beak or against its wall
                    continue
                if np.linalg.norm(np.array([x, y, z]) - CAM) < 8:
                    continue                             # where the camera board is
                pts.append((x, y, z))
    return np.array(pts)


def auto_expose(img, roi):
    s = np.percentile(img[roi], 98)
    return np.clip(img / max(s, 1e-9), 0, 1)


def score(imgs, roi):
    """imgs: {shape: (n_shift, h, w)} exposed images -> (mean log separation, min, dark share)."""
    small = {k: v.reshape(len(v), 60, 2, 80, 2).mean((2, 4)) for k, v in imgs.items()}
    m = {k: v.mean(0) for k, v in small.items()}
    var = {k: ((v - m[k]) ** 2).sum((1, 2)).mean() for k, v in small.items()}
    seps = []
    for a, b in PAIRS:
        d2 = ((m[a] - m[b]) ** 2).sum()
        seps.append(np.log(d2 / (0.5 * (var[a] + var[b]) + 1e-6)))
    dark = np.mean([(v[:, roi] < 0.06).mean() for v in imgs.values()])
    return float(np.mean(seps)), float(np.min(seps)), float(dark), seps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shapes, faces, lipw = lipshapes.make_shapes()
    mask_tris = Scene(("mask",)).tris
    mask_sc = Scene(("mask",))
    face = sweep.face_map_head_only(None)
    cand = candidates(mask_sc, face)
    # mirrored list: index k -> (x, y, z); mirror of k -> (-x, y, z)
    full = [tuple(p) for p in cand] + [(-p[0], p[1], p[2]) for p in cand if p[0] > 0]
    full = np.array(full)
    mirror = {}
    for k, p in enumerate(full):
        q = (-p[0], p[1], p[2])
        mirror[k] = int(np.argmin(np.linalg.norm(full - q, axis=1)))
    cam_led = len(full)                              # extra: LED right next to the lens
    full = np.vstack([full, CAM + [0, -6, 0]])
    print(f"{len(full)} LED positions, {len(shapes)} shapes x {len(SHIFTS)} shifts", flush=True)

    dirs = None
    roi = None
    per = {}                                         # shape -> (n_led, n_shift, H, W)
    for name, verts in shapes.items():
        ss = ShapeScene(verts, faces, lipw, mask_tris, base=shapes["neutral"])
        stack = np.zeros((len(full), len(SHIFTS), H, W), np.float32)
        for si, sh in enumerate(SHIFTS):
            cam = CAM + sh
            d = camera_rays(cam, AIM + sh)
            g = ss.primary(cam, d)
            if roi is None and si == 0 and name == "neutral":
                roi = to_np(g["hit"]).reshape(H, W) & (to_np(g["p"][:, 1]).reshape(H, W) < 125)
            for k, led in enumerate(full):
                stack[k, si] = to_np(ss.shade(g, led + sh, AIM + sh)).reshape(H, W)
        per[name] = stack
        print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    names = list(shapes)

    def layout_images(leds):
        return {n: np.stack([auto_expose(per[n][leds, s].sum(0), roi) for s in range(len(SHIFTS))]) for n in names}

    results = []
    singles = [k for k in range(len(full)) if full[k][0] == 0]
    pairs = sorted({tuple(sorted((k, mirror[k]))) for k in range(len(full) - 1) if full[k][0] > 0})
    layouts = [("cam", (cam_led,))] + [("center", (k,)) for k in singles if k != cam_led]
    layouts += [("pair", p) for p in pairs] + [("pair+cam", p + (cam_led,)) for p in pairs]
    for kind, leds in layouts:
        mean, mn, dark, seps = score(layout_images(list(leds)), roi)
        results.append((mean, mn, dark, kind, leds, seps))
    # best pairs combined with each other (two pairs = 4 LEDs)
    top_pairs = [r[4] for r in sorted([r for r in results if r[3] == "pair"], key=lambda r: -r[0])[:8]]
    for a, b in itertools.combinations(top_pairs, 2):
        leds = tuple(sorted(set(a + b)))
        mean, mn, dark, seps = score(layout_images(list(leds)), roi)
        results.append((mean, mn, dark, "2 pairs", leds, seps))
    results.sort(key=lambda r: -(r[0] - 2 * r[2]))

    def desc(leds):
        return " + ".join(f"({full[k][0]:+.0f},{full[k][1] - lipshapes.SLIT[1]:+.0f},{full[k][2] - lipshapes.SLIT[2]:.0f})"
                          for k in leds)
    base = next(r for r in results if r[3] == "cam")
    lines = [f"camera at {CAM - lipshapes.SLIT} from the mouth slit (x, up, forward), 120 deg, 80x60 net input",
             "LED positions as (x, mm above the slit, mm in front of the slit); score = mean log separation "
             "of shape pairs vs the +-5 mm mask-shift spread (higher = shapes easier to tell apart); dark = "
             "share of mouth-region pixels below 6 % after auto exposure",
             "pairs: " + ", ".join(f"{a}/{b}" for a, b in PAIRS), "",
             f"baseline, one LED at the lens: score {base[0]:.2f}, min {base[1]:.2f}, dark {base[2]:.2f}", ""]
    for mean, mn, dark, kind, leds, seps in results[:15]:
        lines.append(f"{mean:5.2f} (min {mn:5.2f}, dark {dark:.2f})  {kind:8s} {desc(leds)}   "
                     + " ".join(f"{s:.1f}" for s in seps))
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # preview: shapes (columns) for the baseline and the 3 best layouts (rows)
    rows = []
    for mean, mn, dark, kind, leds, _ in [base] + results[:3]:
        imgs = layout_images(list(leds))
        row = [cv2.resize((imgs[n][0] * 255).astype(np.uint8), (W * 2, H * 2), interpolation=cv2.INTER_NEAREST) for n in names]
        row = [cv2.cvtColor(r, cv2.COLOR_GRAY2BGR) for r in row]
        for r, n in zip(row, names):
            cv2.putText(r, n, (4, 16), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(row[0], f"{kind} {mean:.2f}", (4, H * 2 - 8), 0, 0.5, (0, 255, 0), 1)
        rows.append(np.hstack(row))
    cv2.imwrite(str(OUT / "best.png"), np.vstack(rows))
    np.savez(OUT / "leds.npz", full=full, cam=CAM, aim=AIM)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
