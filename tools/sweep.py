"""Mouth camera placement sweep inside the beak (GPU ray casting on the Fusion meshes).

For every camera position on the midline (y = height, z = forward) the camera aims at the
centre of the mouth region of interest (ROI) and is scored by:
  cover  - share of ROI points inside the image (5 % border) and not hidden by nose or mask,
           worst case over the camera shifted +-SHIFT mm in x, y, z (mask sitting differently)
  px/mm  - image scale on the lips at 320x240
ROI = lips with smile / pucker margins + the lower lip and chin swung down by an open jaw.

  python sweep.py            -> out/sweep/*.png, out/sweep/report.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scene import Scene, to_np

OUT = Path(__file__).resolve().parent.parent / "out" / "sweep"
W, H = 320, 240
SHIFT = 10.0
FAN_Z = 340.0          # keep the beak tip (z > this) free for the fan and ESP boards

# name: (model, diagonal FOV deg)
LENSES = {"120": ("fisheye", 120.0), "160": ("fisheye", 160.0), "66": ("rect", 66.0)}


def lens_f(model, fov):
    half = np.radians(fov / 2)
    r = np.hypot(W, H) / 2
    return r / half if model == "fisheye" else r / np.tan(half)


def project(pts, cam, fwd, model, f):
    """Pixel coords (N, 2) for points seen from cam looking along fwd (y-up roll)."""
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    v = pts - cam
    xc, yc, zc = v @ right, v @ up, v @ fwd
    th = np.arctan2(np.hypot(xc, yc), zc)
    r = f * th if model == "fisheye" else np.where(zc > 0, f * np.tan(np.minimum(th, 1.5)), 1e9)
    ph = np.arctan2(yc, xc)
    return np.stack([W / 2 + r * np.cos(ph), H / 2 - r * np.sin(ph)], 1)


def face_map(sc):
    """Front height map of the head surface: z(x, y) on a 1 mm grid."""
    xs, ys = np.arange(-60, 61, 1.0), np.arange(20, 181, 1.0)
    X, Y = np.meshgrid(xs, ys)
    o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 600.0)], 1)
    t, i = sc.cast(o, np.tile([0, 0, -1.0], (X.size, 1)))
    t, i = to_np(t), to_np(i)
    head = (i >= 0) & (sc.obj[np.maximum(i, 0)] == 0)
    z = np.where(head, 600.0 - t, np.nan).reshape(X.shape)
    # the mask is in front of the face: cast again from inside the mask for points it hid
    return xs, ys, z


def mouth_roi(sc):
    xs, ys, z = face_map_head_only(sc)
    mid = z[:, xs == 0].ravel()
    # lower lip bump, upper lip bump, slit = the dip between them (searched below the nose)
    sel = (ys >= 80) & (ys <= 120)
    yy, zz = ys[sel], mid[sel]
    lo = np.argmax(np.where(yy < 97, zz, -1e9))
    up = np.argmax(np.where(yy > yy[lo] + 6, zz, -1e9))
    slit_i = lo + np.argmin(zz[lo:up + 1])
    slit_y, slit_z = yy[slit_i], zz[slit_i]
    pts = []
    for x in np.arange(-34, 35, 4.0):
        for y in np.arange(slit_y - 14, slit_y + 15, 3.0):
            if abs(y - slit_y) < 3:
                continue      # the crease itself is hidden from most angles
            zi = z[np.argmin(abs(ys - y)), np.argmin(abs(xs - x))]
            if np.isfinite(zi):
                pts.append((x, y, zi))
    pts = np.array(pts)
    # open jaw: lower face swung 22 deg about the TMJ axis (approximate position)
    piv = np.array([0, slit_y + 30, slit_z - 85])
    a = -np.radians(22)     # chin goes down and back
    rot = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
    low = pts[pts[:, 1] < slit_y]
    jaw = (low - piv) @ rot.T + piv
    chin = []
    for x in np.arange(-20, 21, 5.0):
        for y in (slit_y - 22, slit_y - 28):
            zi = z[np.argmin(abs(ys - y)), np.argmin(abs(xs - x))]
            if np.isfinite(zi):
                chin.append((x, y, zi))
    chin = (np.array(chin) - piv) @ rot.T + piv
    return dict(slit=np.array([0, slit_y, slit_z]), neutral=pts, jaw=jaw, chin=chin)


_HEAD = None


def face_map_head_only(sc):
    global _HEAD
    if _HEAD is None:
        _HEAD = face_map(Scene(("head",)))
    return _HEAD


def score(sc, cam, aim, roi_pts, model, f, surf, slit_y):
    fwd = aim - cam
    pix = project(roi_pts, cam, fwd, model, f)
    m = 0.05
    inside = (pix[:, 0] > W * m) & (pix[:, 0] < W * (1 - m)) & (pix[:, 1] > H * m) & (pix[:, 1] < H * (1 - m))
    d = roi_pts - cam
    dist = np.linalg.norm(d, axis=1)
    t, i = sc.cast(cam[None], d / dist[:, None])
    t, i = to_np(t), to_np(i)
    hit_y = sc.tris[np.maximum(i, 0)].mean(1)[:, 1]
    is_head = sc.obj[np.maximum(i, 0)] == 0
    # moved jaw points: the neutral lower face is not an occluder, only nose/upper lip and mask
    blocks = np.isfinite(t) & (t < dist - 4.0) & (surf | ~is_head | (hit_y > slit_y + 3))
    return inside & ~blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=5.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    box = (np.array([-130, -20, 120.0]), np.array([130, 230, 420.0]))
    sc = Scene(box=box)
    roi = mouth_roi(sc)
    pts = np.concatenate([roi["neutral"], roi["jaw"], roi["chin"]])
    surf = np.r_[np.ones(len(roi["neutral"]), bool), np.zeros(len(roi["jaw"]) + len(roi["chin"]), bool)]
    aim = pts.mean(0)
    lips = roi["neutral"][np.abs(roi["neutral"][:, 1] - roi["slit"][1]) < 8]
    print("slit", roi["slit"].round(1), "aim", aim.round(1), "roi pts", len(pts))

    # inner mask wall on the midline, per height
    mask = Scene(("mask",))
    ys = np.arange(90, 171, args.step)
    zs = np.arange(roi["slit"][2] + 15, 380, args.step)
    o = np.stack([np.zeros_like(ys), ys, np.full_like(ys, roi["slit"][2])], 1)
    tw, _ = mask.cast(o, np.tile([0, 0, 1.0], (len(ys), 1)))
    wall = roi["slit"][2] + to_np(tw)
    xs_, ys_, fz = face_map_head_only(sc)
    face_mid = fz[:, xs_ == 0].ravel()

    shifts = [np.zeros(3)] + [s * SHIFT * e for e in np.eye(3) for s in (1, -1)]
    lenses = {k: (m, lens_f(m, fov)) for k, (m, fov) in LENSES.items()}
    cov = {k: np.full((len(ys), len(zs)), np.nan) for k in lenses}
    ppm = {k: np.full((len(ys), len(zs)), np.nan) for k in lenses}
    slit_y = roi["slit"][1]
    for a, y in enumerate(ys):
        fy = face_mid[np.argmin(abs(ys_ - y))]
        for b, zc in enumerate(zs):
            if zc > wall[a] - 12 or (np.isfinite(fy) and zc < fy + 12):
                continue          # outside the free space (board/lens needs ~12 mm)
            cam = np.array([0.0, y, zc])
            fwd = aim - cam
            cams = np.array([cam + s for s in shifts])
            d = (pts[None] - cams[:, None]).reshape(-1, 3)
            o = np.repeat(cams, len(pts), 0)
            dist = np.linalg.norm(d, axis=1)
            t, i = sc.cast(o, d / dist[:, None])
            t, i = to_np(t), to_np(i)
            j = np.maximum(i, 0)
            hit_y = sc.tris[j].mean(1)[:, 1]
            is_head = sc.obj[j] == 0
            sf = np.tile(surf, len(shifts))
            # moved jaw points: the neutral lower face is not an occluder, only nose/upper lip and mask
            vis = ~(np.isfinite(t) & (t < dist - 4.0) & (sf | ~is_head | (hit_y > slit_y + 3)))
            vis = vis.reshape(len(shifts), -1)
            for k, (model, f) in lenses.items():
                worst = 1.0
                for n, c in enumerate(cams):
                    pix = project(pts, c, fwd, model, f)
                    m = 0.05
                    inside = (pix[:, 0] > W * m) & (pix[:, 0] < W * (1 - m)) & (pix[:, 1] > H * m) & (pix[:, 1] < H * (1 - m))
                    worst = min(worst, (inside & vis[n]).mean())
                cov[k][a, b] = worst
                p = project(lips, cam, fwd, model, f)
                ppm[k][a, b] = np.ptp(p[:, 0]) / max(np.ptp(lips[:, 0]), 1e-6)
    res = {k: (cov[k], ppm[k]) for k in lenses}
    lines = [f"slit (mouth centre) {roi['slit'].round(1)} mm, aim {aim.round(1)}, shift +-{SHIFT} mm, fan zone z > {FAN_Z}"]
    for name, (cv, pm) in res.items():
        good = np.where(cv >= 0.98, pm, -1)
        if np.nanmax(good) > 0:
            a, b = np.unravel_index(np.nanargmax(good), good.shape)
            lines.append(f"{name:>4} deg: best full-cover spot y={ys[a]:.0f} z={zs[b]:.0f} "
                         f"({zs[b]-roi['slit'][2]:.0f} mm in front of the mouth, {ys[a]-slit_y:+.0f} mm above), "
                         f"{pm[a, b]:.1f} px/mm; full cover in {np.nansum(cv >= 0.98)} of {np.isfinite(cv).sum()} spots, "
                         f"{np.nansum((cv >= 0.98) & (zs[None] <= FAN_Z))} with z <= {FAN_Z:.0f}")
        else:
            a, b = np.unravel_index(np.nanargmax(cv), cv.shape)
            lines.append(f"{name:>4} deg: no full-cover spot; best cover {cv[a, b]:.2f} at y={ys[a]:.0f} z={zs[b]:.0f}")
    np.savez(OUT / "sweep.npz", ys=ys, zs=zs, wall=wall, slit=roi["slit"], aim=aim,
             **{f"cov_{k}": v[0] for k, v in res.items()}, **{f"ppm_{k}": v[1] for k, v in res.items()})

    fig, axs = plt.subplots(2, len(LENSES), figsize=(5 * len(LENSES), 8), squeeze=False)
    ext = [zs[0] - roi["slit"][2], zs[-1] - roi["slit"][2], ys[0] - roi["slit"][1], ys[-1] - roi["slit"][1]]
    for c, (name, (cov, ppm)) in enumerate(res.items()):
        im = axs[0, c].imshow(cov, origin="lower", extent=ext, vmin=0.6, vmax=1, aspect="equal", cmap="viridis")
        axs[0, c].set_title(f"{name} deg: worst-case cover (+-{SHIFT:.0f} mm)")
        plt.colorbar(im, ax=axs[0, c])
        im = axs[1, c].imshow(np.where(cov >= 0.98, ppm, np.nan), origin="lower", extent=ext, aspect="equal", cmap="magma")
        axs[1, c].set_title(f"{name} deg: px/mm on lips (full cover only)")
        plt.colorbar(im, ax=axs[1, c])
        for ax in axs[:, c]:
            ax.axvline(FAN_Z - roi["slit"][2], color="r", ls="--", lw=1)
            ax.set_xlabel("mm in front of the mouth"); ax.set_ylabel("mm above the mouth")
    plt.tight_layout(); plt.savefig(OUT / "sweep.png", dpi=90)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
