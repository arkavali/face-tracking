"""Mask-camera simulator: re-render a frontal face video frame as seen by a small
wide-angle camera rigidly mounted inside the mask, above the AR mirror holder,
looking down at the brows.

How it works
------------
* MediaPipe landmarks (x, y in pixels, z ~ x scale) give a 3D face mesh whose
  texture is the frontal video frame (weak perspective: the studio camera is ~1 m
  away, so texture coordinates are just the landmark pixel positions).
* The facial transformation matrix gives the head rotation; landmarks are moved
  into a metric HEAD frame (mm): origin = midpoint of the four eye corners,
  +x = subject's left, +y = up, +z = forward (out of the face).
  Scale comes from an assumed 63 mm distance between the eye centres.
* A virtual camera with an equidistant fisheye lens is placed in that head frame,
  so it follows the head exactly like a camera mounted in the mask.
* Every mesh triangle is projected into the virtual camera; a painter-ordered
  triangle-id map plus per-triangle affine maps give a dense remap from the
  virtual image back into the frontal frame (one cv2.remap per frame).
* Lighting: an IR LED next to the lens (inverse square + Lambert), applied on top
  of an "IR-like" texture (mostly red channel, reduced local contrast, because
  skin is flatter and hair less dark in near-IR than in visible light).
* Occluders (mirror holder top flange, optional glasses rim) are 3D quads in the
  head frame, drawn over the face.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from common import BROW, EYE_CORNERS, mesh_triangles

EYE4 = [33, 133, 362, 263]
IPD_MM = 63.0


# ----------------------------------------------------------------------------- head frame

def to_cam_px(lm: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalised landmarks -> pixel units with axes x right, y up, z towards the viewer."""
    return lm * np.array([w, -h, -w], np.float32)


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    pad = k // 2
    ap = np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)], 0)
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 0, ap)


def _fill_nan_frames(a: np.ndarray) -> np.ndarray:
    """Replace NaN frames (face lost) with the nearest valid frame."""
    bad = np.isnan(a.reshape(len(a), -1)).any(1)
    if not bad.any():
        return a
    good = np.flatnonzero(~bad)
    if len(good) == 0:
        return a
    idx = np.searchsorted(good, np.arange(len(a))).clip(0, len(good) - 1)
    prev = good[(idx - 1).clip(0)]
    nxt = good[idx]
    near = np.where(np.abs(prev - np.arange(len(a))) <= np.abs(nxt - np.arange(len(a))), prev, nxt)
    out = a.copy()
    out[bad] = a[near[bad]]
    return out


@dataclass
class HeadTrack:
    """Per-frame landmarks expressed in the metric head frame."""
    head: np.ndarray      # (N, 478, 3) mm, head frame
    src_px: np.ndarray    # (N, 478, 2) texture coordinates in the frontal frame
    fwd: np.ndarray       # (N, 3) direction to the frontal camera in the head frame
    valid: np.ndarray     # (N,) bool, face was detected
    mm_per_px: float


def head_track(lm: np.ndarray, mat: np.ndarray, w: int, h: int, smooth: int = 5) -> HeadTrack:
    valid = ~np.isnan(lm[:, 0, 0])
    lm = _fill_nan_frames(lm)
    mat = _fill_nan_frames(mat)
    c = to_cam_px(lm, w, h)
    eyes_r = c[:, list(EYE_CORNERS["R"])].mean(1)
    eyes_l = c[:, list(EYE_CORNERS["L"])].mean(1)
    mm_per_px = IPD_MM / float(np.nanmedian(np.linalg.norm(eyes_l - eyes_r, axis=1)))
    # Head pose: rotation of MediaPipe's canonical face (y up, z towards the viewer)
    # into camera axes. Lightly smoothed over time, then re-orthonormalised.
    R = _smooth(mat[:, :3, :3].astype(np.float64), smooth)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    o = _smooth(c[:, EYE4].mean(1).astype(np.float64), smooth)
    head = np.einsum("npa,nab->npb", c - o[:, None, :], R) * mm_per_px  # R^T (c - o)
    fwd = R[:, 2, :]  # camera +z (towards viewer) expressed in head coords: row 2 of R
    return HeadTrack(head.astype(np.float32), (lm[:, :, :2] * [w, h]).astype(np.float32),
                     fwd.astype(np.float32), valid, mm_per_px)


@dataclass
class Anchors:
    """Per-subject brow geometry in the head frame, taken from a neutral clip."""
    brow_center: dict          # side -> (3,) mm, centre of the brow (upper+lower contour mean)
    brow_pts: dict             # side -> (10, 3) mm, contour points (upper 5 then lower 5)
    eye_center: dict           # side -> (3,) mm
    skin_z: float              # forward position of the brow skin (mean contour z)
    ref_head: np.ndarray       # (478, 3) neutral face shape, used for the lighting
    ref_fwd: np.ndarray        # (3,) frontal camera direction for that shape

    @staticmethod
    def from_track(tr: HeadTrack) -> "Anchors":
        ref = np.nanmedian(tr.head[tr.valid], axis=0)
        fwd = np.nanmedian(tr.fwd[tr.valid], axis=0)
        bc, bp, ec = {}, {}, {}
        for s in "RL":
            idx = BROW[s]["upper"] + BROW[s]["lower"]
            bp[s] = ref[idx]
            bc[s] = ref[idx].mean(0)
            ec[s] = ref[list(EYE_CORNERS[s])].mean(0)
        skin_z = float(np.mean([bc["R"][2], bc["L"][2]]))
        return Anchors(bc, bp, ec, skin_z, ref, fwd / np.linalg.norm(fwd))


# ----------------------------------------------------------------------------- camera

class FisheyeCamera:
    """Equidistant fisheye (r = f * theta), OpenCV image axes (x right, y down)."""

    def __init__(self, pos, target, width=320, height=240, dfov_deg=120.0, up=(0.0, 1.0, 0.0)):
        self.pos = np.asarray(pos, np.float64)
        self.target = np.asarray(target, np.float64)
        f = self.target - self.pos
        f /= np.linalg.norm(f)
        x = np.cross(f, up)
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        self.R = np.stack([x, y, f])          # head -> camera rows
        self.w, self.h = width, height
        self.focal = (np.hypot(width, height) / 2) / np.radians(dfov_deg / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2
        self.max_theta = np.radians(100.0)    # nothing beyond 100 deg off-axis

    def project(self, P: np.ndarray):
        """Head-frame points (..., 3) -> pixels (..., 2), distance (...), valid (...)."""
        v = (np.asarray(P, np.float64) - self.pos) @ self.R.T
        rho = np.hypot(v[..., 0], v[..., 1])
        theta = np.arctan2(rho, v[..., 2])
        r = self.focal * theta
        k = np.where(rho > 1e-9, r / np.maximum(rho, 1e-9), self.focal / np.maximum(v[..., 2], 1e-9))
        uv = np.stack([self.cx + v[..., 0] * k, self.cy + v[..., 1] * k], -1)
        dist = np.linalg.norm(v, axis=-1)
        return uv, dist, theta < self.max_theta

    def px_per_mm_at(self, P: np.ndarray) -> float:
        """Image displacement (px) of a 1 mm vertical (head y) move at point P."""
        a, _, _ = self.project(P)
        b, _, _ = self.project(np.asarray(P) + [0, 1.0, 0])
        return float(np.linalg.norm(b - a))


@dataclass
class Rig:
    """Camera placement in the head frame plus the AR mirror holder geometry.

    Holder (per eye, from the user's CAD sections, 2026-10-09): a vertical mirror frame
    ~50 mm wide, 58 mm tall, its top ~29 mm above the eye centre and its face ~31.7 mm in
    front of the brow; from the frame's outer top corner a narrow arm (~5 mm) runs
    diagonally towards the face and ends ~14 mm in front of the brow, just inside of
    the eye centre. The camera sits above the holder: cam_y above the eye-corner line,
    cam_z in front of the brow skin. The red dot in the CAD side view was ~(57, 39).
    """
    mode: str = "side"           # "side": one camera per brow, "center": one camera for both
    cam_y: float = 57.0
    cam_z: float = 39.0
    lateral_mm: float = 0.0      # side mode: shift the camera outwards (+) from above the eye
    aim_down_mm: float = 0.0     # aim below the brow centre
    flange_top: float = 29.0     # holder top above the eye-corner line
    plate_gap: float = 31.7      # mirror frame face in front of the brow skin
    arm_gap: float = 14.0        # arm tip in front of the brow skin
    arm: bool = True
    width: int = 320
    height: int = 240
    dfov_deg: float = 120.0

    def cameras(self, an: Anchors) -> dict:
        y = self.cam_y
        z = an.skin_z + self.cam_z
        cams = {}
        if self.mode == "side":
            for s in "RL":
                tgt = an.brow_center[s] - [0, self.aim_down_mm, 0]
                out = -1.0 if s == "R" else 1.0  # subject's right is head -x
                pos = np.array([an.eye_center[s][0] + out * self.lateral_mm, y, z])
                cams[s] = FisheyeCamera(pos, tgt, self.width, self.height, self.dfov_deg)
        else:
            tgt = (an.brow_center["R"] + an.brow_center["L"]) / 2 - [0, self.aim_down_mm, 0]
            cams["C"] = FisheyeCamera([0.0, y, z], tgt, self.width, self.height, self.dfov_deg)
        return cams

    # Lights: "cam" = LED next to the brow camera; "below" = LED under the AR mirror
    # (shared with a future pupil camera), lighting the eye socket and the lower brow.
    lights: tuple = ("cam",)
    below_y: float = -12.0       # "below" LED height relative to the eye-corner line
    below_fwd: float = 30.0      # "below" LED distance in front of the brow skin
    below_weight: float = 1.0

    def occluders(self, an: Anchors, key: str) -> list:
        quads = []
        for s in ("RL" if key == "C" else key):
            quads += holder(an, s, self.flange_top, self.plate_gap, self.arm_gap, self.arm)
        return quads

    def light_list(self, an: Anchors, key: str, cam: FisheyeCamera) -> list:
        out = []
        for spec in self.lights:
            if spec == "cam":
                out.append((cam.pos, 1.0))
            elif spec == "below":
                for s in ("RL" if key == "C" else key):
                    out.append((np.array([an.eye_center[s][0], self.below_y,
                                          an.skin_z + self.below_fwd]), self.below_weight))
        return out


# ----------------------------------------------------------------------------- appearance

@dataclass
class Look:
    ir_red_weight: float = 0.85     # near-IR looks most like the red channel
    ir_contrast: float = 0.7        # local contrast kept vs visible light (1 = unchanged)
    ambient: float = 0.25           # share of light not coming from the LED next to the lens
    blur_px: float = 0.7            # defocus, Gaussian sigma in output pixels
    noise_read: float = 2.0         # additive noise, grey levels
    noise_shot: float = 0.35        # shot-noise factor (sigma = k * sqrt(level))
    target_p90: float = 200.0       # auto-exposure target for the 90th percentile
    bg_level: float = 40.0          # where no face mesh is visible (inside of the mask)
    seed: int = 0
    rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)


def ir_texture(frame_bgr: np.ndarray, look: Look, mm_per_px: float) -> np.ndarray:
    b, g, r = cv2.split(frame_bgr.astype(np.float32))
    grey = look.ir_red_weight * r + (1 - look.ir_red_weight) * g
    if look.ir_contrast != 1.0:
        sigma = 8.0 / mm_per_px  # ~8 mm: keeps skin shading, squashes brow/skin contrast
        base = cv2.GaussianBlur(grey, (0, 0), sigma)
        grey = base + look.ir_contrast * (grey - base)
    return grey


@dataclass
class Quad:
    pts: np.ndarray        # (4, 3) head-frame mm
    level: float = 25.0    # grey level before lighting


def holder(an: Anchors, side: str, top_y: float = 29.0, plate_gap: float = 31.7,
           arm_gap: float = 14.0, arm: bool = True, half_width: float = 25.0,
           height: float = 58.0, arm_width: float = 5.0) -> list:
    """AR mirror holder of one eye: the vertical mirror frame plus its diagonal top arm."""
    ex = an.eye_center[side][0]
    out = -1.0 if side == "R" else 1.0            # lateral direction in head x
    zp = an.skin_z + plate_gap
    quads = [Quad(np.array([[ex - half_width, top_y, zp], [ex + half_width, top_y, zp],
                            [ex + half_width, top_y - height, zp], [ex - half_width, top_y - height, zp]]), 25.0)]
    if arm:
        a = np.array([ex + out * (half_width + 6.0), top_y, zp])   # outer top corner of the frame
        b = np.array([ex - out * 3.0, top_y, an.skin_z + arm_gap])  # tip, just inside the eye centre
        d = b - a
        n = np.array([-d[2], 0.0, d[0]])
        n *= arm_width / 2 / np.linalg.norm(n)
        quads.append(Quad(np.array([a + n, b + n, b - n, a - n]), 25.0))          # top face
        quads.append(Quad(np.array([b + n, b - n, b - n - [0, 6, 0], b + n - [0, 6, 0]]), 25.0))  # tip clip
    return quads


def glasses_rim(an: Anchors, rim_y: float = 15.0, lens_gap: float = 14.0,
                thick: float = 4.0, half_width: float = 25.0) -> list:
    """Top rims of a pair of glasses (one quad each) plus the bridge, as dark strips."""
    quads = []
    for s in "RL":
        ec = an.eye_center[s]
        z = ec[2] + lens_gap
        xs = np.linspace(ec[0] - half_width, ec[0] + half_width, 7)
        for a, b in zip(xs[:-1], xs[1:]):
            # slight arc: rim highest above the pupil
            ya = rim_y - 0.004 * (a - ec[0]) ** 2
            yb = rim_y - 0.004 * (b - ec[0]) ** 2
            quads.append(Quad(np.array([[a, ya, z], [b, yb, z], [b, yb - thick, z], [a, ya - thick, z]]), 30.0))
    zb = np.mean([an.eye_center["R"][2], an.eye_center["L"][2]]) + 14.0 + 2.0
    xa = an.eye_center["R"][0] + half_width
    xb = an.eye_center["L"][0] - half_width
    yb = rim_y - 4.0
    quads.append(Quad(np.array([[xa, yb, zb], [xb, yb, zb], [xb, yb - 3, zb], [xa, yb - 3, zb]]), 30.0))
    return quads


# ----------------------------------------------------------------------------- renderer

_TRI = None


def _triangles() -> np.ndarray:
    global _TRI
    if _TRI is None:
        _TRI = mesh_triangles()
    return _TRI


def _raster(head: np.ndarray, fwd: np.ndarray, cam: FisheyeCamera) -> dict:
    """Rasterise the face mesh into the virtual camera.

    Returns the oriented triangles, their normals, the destination vertex positions,
    per-pixel triangle ids (painter's order, far first) and per-pixel barycentrics.
    """
    tri = _triangles()
    W, H = cam.w, cam.h
    P = head[:468].astype(np.float64)
    uv, dist, ok = cam.project(P)

    # Orient triangles counter-clockwise as seen by the frontal camera, so the
    # outward normal points towards it (positive along fwd).
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    n = np.cross(b - a, c - a)
    flip = (n @ fwd) < 0
    tri = tri.copy()
    tri[flip, 1], tri[flip, 2] = tri[flip, 2], tri[flip, 1].copy()
    n[flip] *= -1
    cent = (a + b + c) / 3
    keep = (np.einsum("ij,ij->i", n, cam.pos - cent) > 0) & ok[tri].all(1)   # back-face cull
    d2 = uv[tri]                                                              # (T, 3, 2)
    keep &= ((d2[..., 0].max(1) >= 0) & (d2[..., 0].min(1) < W) &
             (d2[..., 1].max(1) >= 0) & (d2[..., 1].min(1) < H))
    area = (d2[:, 1, 0] - d2[:, 0, 0]) * (d2[:, 2, 1] - d2[:, 0, 1]) - \
           (d2[:, 2, 0] - d2[:, 0, 0]) * (d2[:, 1, 1] - d2[:, 0, 1])
    keep &= np.abs(area) > 1e-3
    tid = np.flatnonzero(keep)
    order = tid[np.argsort(-dist[tri[tid]].mean(1))]

    idmap = np.full((H, W), -1, np.int32)
    for t in order:
        cv2.fillConvexPoly(idmap, np.round(d2[t] * 16).astype(np.int32), int(t), lineType=cv2.LINE_8, shift=4)

    Dh = np.concatenate([d2, np.ones((len(tri), 3, 1))], 2)
    good = np.abs(np.linalg.det(Dh)) > 1e-9
    Dinv = np.zeros_like(Dh)
    Dinv[good] = np.linalg.inv(Dh[good])            # [x y 1] @ Dinv -> barycentric

    ys, xs = np.nonzero(idmap >= 0)
    t = idmap[ys, xs]
    X = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
    bar = np.einsum("ni,nij->nj", X, Dinv[t])
    return {"tri": tri, "n": n, "P": P, "dist": dist, "Dinv": Dinv,
            "idmap": idmap, "ys": ys, "xs": xs, "t": t, "X": X, "bar": bar}


def led_shading(head: np.ndarray, fwd: np.ndarray, cam: FisheyeCamera, ambient: float,
                lights: list | None = None) -> np.ndarray:
    """Irradiance image from IR LEDs (inverse square + Lambert, no cast shadows).

    lights: [(position_mm, weight), ...]; default is one LED next to the lens.
    Each light is normalised to give `weight` at the camera's aim point (head-on).
    Evaluated on a fixed (neutral) face shape: the bone under the brows does not move,
    only skin and hair slide over it, so the lighting must not follow the brow
    landmarks (MediaPipe's mesh would drag the brow-ridge shading along with them).
    """
    if lights is None:
        lights = [(cam.pos, 1.0)]
    r = _raster(head, fwd, cam)
    P, tri, n = r["P"], r["tri"], r["n"]
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, tri[:, k], n)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
    vshade = np.zeros(len(P))
    wsum = sum(w for _, w in lights)
    for pos, w in lights:
        L = np.asarray(pos, np.float64) - P
        dl = np.linalg.norm(L, axis=1)
        d_ref = np.linalg.norm(cam.target - np.asarray(pos, np.float64))
        lam = np.clip(np.einsum("ij,ij->i", vn, L) / dl, 0, 1)
        vshade += w * (d_ref / dl) ** 2 * (ambient + (1 - ambient) * lam)
    vshade /= wsum
    shade = np.full((cam.h, cam.w), np.nan, np.float32)
    shade[r["ys"], r["xs"]] = np.einsum("nj,nj->n", r["bar"], vshade[tri[r["t"]]])
    # Fill pixels the neutral mesh does not cover (e.g. a raised brow above the
    # neutral silhouette) from the nearest covered pixel.
    hole = np.isnan(shade)
    if hole.any() and (~hole).any():
        _, lab = cv2.distanceTransformWithLabels(hole.astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        src_y, src_x = np.nonzero(~hole)
        # labels index the zero pixels in scan order
        lut = np.zeros(lab.max() + 1, np.float32)
        order = np.argsort(lab[src_y, src_x])
        lut[lab[src_y, src_x][order]] = shade[src_y, src_x][order]
        shade[hole] = lut[lab[hole]]
    return np.nan_to_num(shade, nan=ambient)


def render(tex: np.ndarray, src_px: np.ndarray, head: np.ndarray, fwd: np.ndarray,
           cam: FisheyeCamera, look: Look, quads: list = (), shade: np.ndarray | None = None) -> dict:
    """Render one virtual-camera frame.

    tex     float32 texture (frontal frame, IR-like grey)
    src_px  (478, 2) texture coordinates; head (478, 3) head-frame mm; fwd (3,) frontal view dir
    shade   irradiance image from led_shading() on the neutral face (computed from the
            current mesh if omitted)
    Returns dict(img=uint8, depth=float32 mm (inf where empty), face/occ masks).
    """
    W, H = cam.w, cam.h
    r = _raster(head, fwd, cam)
    tri, t, X = r["tri"], r["t"], r["X"]
    Msrc = r["Dinv"] @ src_px[tri].astype(np.float64)   # (T, 3, 2): [x y 1] @ M -> texture px
    src = np.einsum("ni,nij->nj", X, Msrc[t])
    mapx = np.full((H, W), -1, np.float32)
    mapy = np.full((H, W), -1, np.float32)
    mapx[r["ys"], r["xs"]] = src[:, 0]
    mapy[r["ys"], r["xs"]] = src[:, 1]
    img = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    depth = np.full((H, W), np.inf, np.float32)
    depth[r["ys"], r["xs"]] = np.einsum("nj,nj->n", r["bar"], r["dist"][tri[t]])
    if shade is None:
        shade = led_shading(head, fwd, cam, look.ambient)
    d_ref = np.linalg.norm(cam.target - cam.pos)

    face = r["idmap"] >= 0
    out = np.where(face, img * shade, look.bg_level * 0.5)

    # Occluders (always in front of the face here).
    occ = np.zeros((H, W), bool)
    for q in quads:
        # Subdivide so that quads partly outside the fisheye's field (or behind the
        # lens, e.g. a holder right next to the camera) still draw their visible part.
        a, b, c4, d4 = (np.asarray(p, np.float64) for p in q.pts)
        n = 12
        s = np.linspace(0, 1, n + 1)
        grid = ((1 - s)[:, None, None] * ((1 - s)[None, :, None] * a + s[None, :, None] * b) +
                s[:, None, None] * ((1 - s)[None, :, None] * d4 + s[None, :, None] * c4))
        guv, gd, gok = cam.project(grid)
        m = np.zeros((H, W), bool)
        cell = np.zeros((H, W), np.uint8)
        for i in range(n):
            for j in range(n):
                if gok[i, j] and gok[i, j + 1] and gok[i + 1, j + 1] and gok[i + 1, j]:
                    poly = np.array([guv[i, j], guv[i, j + 1], guv[i + 1, j + 1], guv[i + 1, j]])
                    cell[:] = 0
                    cv2.fillConvexPoly(cell, np.round(poly * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
                    cd = gd[i:i + 2, j:j + 2].mean()
                    m |= cell.astype(bool) & (depth > cd)      # only where it is in front of the face
        if not m.any():
            continue
        lvl = q.level * (d_ref / max(float(gd[gok].mean()), 1.0)) ** 2
        out[m] = lvl
        occ |= m

    # Auto exposure on the visible face, then optics and sensor noise.
    vis = face & ~occ
    if vis.any():
        out = out * (look.target_p90 / max(np.percentile(out[vis], 90), 1e-3))
    if look.blur_px > 0:
        out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), look.blur_px)
    rng = look.rng
    out = out + rng.normal(0, look.noise_read, out.shape) + \
        rng.normal(0, 1, out.shape) * look.noise_shot * np.sqrt(np.clip(out, 0, None))
    return {"img": np.clip(out, 0, 255).astype(np.uint8), "depth": depth, "face": face, "occ": occ}


def brow_gt(head_f: np.ndarray, cam: FisheyeCamera, side: str) -> dict:
    """Ground-truth brow contours projected into the virtual camera (outer -> inner)."""
    up, _, _ = cam.project(head_f[BROW[side]["upper"]])
    lo, _, _ = cam.project(head_f[BROW[side]["lower"]])
    return {"upper": up, "lower": lo, "mid": (up + lo) / 2}
