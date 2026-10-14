"""Synthetic near-IR eye camera: ray-traced eye seen by a small camera inside the mask.

Why not brow_tracker's camsim: that re-projects a frontal video (~2 px/mm) onto a face
mesh. A pupil tracker needs ~10 px/mm, a refracting cornea, corneal glints and lids that
move with the gaze, so this module ray-traces a parametric eye instead.

Frame (mm), fixed to the head, one per eye: origin = eyeball rotation centre,
+x = subject's left, +y = up, +z = forward (primary gaze). The default eye is the RIGHT
eye, so +x points towards the nose.

Model (textbook values, Le Grand / Gullstrand simplified):
* eyeball (sclera) sphere r = 12 mm; cornea sphere r = 7.8 mm whose centre sits 5.7 mm
  along the gaze, so the apex is 13.5 mm from the rotation centre; single refracting
  surface n = 1.3375; iris + pupil in a plane 10.0 mm along the gaze, iris r = 6 mm.
* lids: a skin sphere r = 12.6 mm fixed to the head with an almond-shaped opening in
  (azimuth, elevation); the upper lid follows the vertical gaze; "openness" lowers it
  towards the lower lid (blink), "squint" also raises the lower lid.
* surroundings: face plane, cheek and brow-ridge spheres (only for occlusion / background).
* lashes: 3D strands rooted on the lid margins, drawn with a depth test.
* light: point LEDs (inverse square + Lambert) plus ambient; corneal glints (Purkinje I)
  solved analytically per LED; optional reflection of the AR screen (visible light, seen
  only when the lens has no IR-pass filter); optional semi-transparent combiner plate
  between eye and camera (transmission, LED ghost reflection, veil).
* glasses (optional): a thin spectacle lens ~12 mm in front of the cornea with a frame
  rim; rays through it are bent by the lens power (thin-lens deflection), and both lens
  surfaces mirror the LEDs (meniscus spheres, 4% per uncoated surface). With glasses the
  corneal glints are traced per pixel too (LEDs as 5 mm discs), so they move with the
  lens like the rest of the eye.
* camera: pinhole (OV2640 66 deg diagonal ~ 54.9 deg horizontal at 4:3), defocus blur,
  shot + read noise after an auto-exposure gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

R_EYE = 12.0
R_CORNEA = 7.8
D_CORNEA = 13.5 - R_CORNEA        # cornea sphere centre along the gaze
N_CORNEA = 1.3375
Z_IRIS = 10.0                     # iris / pupil plane along the gaze
R_IRIS = 6.0
R_SKIN = 12.6
LIMBUS_DOT = (R_EYE ** 2 - R_CORNEA ** 2 + D_CORNEA ** 2) / (2 * D_CORNEA)  # p.g at the limbus

# Labels of the ray-traced surfaces
SKIN, SCLERA, IRIS, PUPIL, FACE, NONE = 0, 1, 2, 3, 4, 5

# Near-IR (850 nm) reflectance guesses, 0..1
ALB = {SKIN: 0.55, SCLERA: 0.62, IRIS: 0.30, PUPIL: 0.03, FACE: 0.5, NONE: 0.0}
ALB_LASH = 0.10


def gaze_vec(yaw_deg, pitch_deg) -> np.ndarray:
    """Gaze unit vector; yaw > 0 = towards +x (right eye: towards the nose), pitch > 0 = up."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])


def look_rot(fwd, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Rows = x (image right), y (image down), z (forward) of a camera looking along fwd."""
    f = np.asarray(fwd, float)
    f = f / np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    return np.stack([x, y, f])


def rot_axis(axis, deg) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return cv2.Rodrigues((a * np.radians(deg)).reshape(3, 1))[0]


# ----------------------------------------------------------------------------- camera

@dataclass
class Camera:
    pos: np.ndarray
    R: np.ndarray                 # rows: camera axes in the eye frame
    w: int = 320
    h: int = 240
    dfov_deg: float = 66.0        # diagonal field of view

    @property
    def f(self) -> float:
        return (np.hypot(self.w, self.h) / 2) / np.tan(np.radians(self.dfov_deg / 2))

    @staticmethod
    def looking_at(pos, target, **kw) -> "Camera":
        pos = np.asarray(pos, float)
        return Camera(pos, look_rot(np.asarray(target, float) - pos), **kw)

    def rays(self, ss: int = 1):
        """Unit ray directions (H*ss, W*ss, 3) in the eye frame."""
        f = self.f * ss
        u = (np.arange(self.w * ss) + 0.5) - self.w * ss / 2
        v = (np.arange(self.h * ss) + 0.5) - self.h * ss / 2
        uu, vv = np.meshgrid(u, v)
        d = np.stack([uu / f, vv / f, np.ones_like(uu)], -1) @ self.R
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def project(self, P):
        """Eye-frame points (..., 3) -> pixels (..., 2) and depth (...)."""
        v = (np.asarray(P, float) - self.pos) @ self.R.T
        z = v[..., 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        uv = np.stack([v[..., 0] / zs * self.f + self.w / 2 - 0.5,
                       v[..., 1] / zs * self.f + self.h / 2 - 0.5], -1)
        return uv, z

    def moved(self, dpos=(0, 0, 0), drot_deg=(0, 0, 0)) -> "Camera":
        """Rigidly moved copy (mask slip): translation in mm, small rotations about x, y, z."""
        R = self.R
        for ax, d in zip(np.eye(3), drot_deg):
            if d:
                R = R @ rot_axis(ax, d).T
        return Camera(self.pos + np.asarray(dpos, float), R, self.w, self.h, self.dfov_deg)


# ----------------------------------------------------------------------------- geometry

def _sphere_hit(o, d, c, r):
    """Nearest positive t of rays o + t d with a sphere (front and back t), NaN if missed."""
    oc = o - c
    b = np.einsum("...k,...k->...", d, oc)
    cc = np.einsum("...k,...k->...", oc, oc) - r * r
    disc = b * b - cc
    s = np.sqrt(np.where(disc > 0, disc, np.nan))
    t0, t1 = -b - s, -b + s
    return t0, t1


def _first_pos(t0, t1):
    t = np.where(t0 > 1e-6, t0, np.where(t1 > 1e-6, t1, np.nan))
    return t


@dataclass
class EyeState:
    yaw: float = 0.0              # deg
    pitch: float = 0.0            # deg
    pupil_r: float = 2.0          # mm
    openness: float = 1.0        # 1 = relaxed open, 0 = closed
    squint: float = 0.0          # 0..1, raises the lower lid

    @property
    def g(self) -> np.ndarray:
        return gaze_vec(self.yaw, self.pitch)


@dataclass
class Lids:
    """Opening of the skin sphere in (azimuth, elevation) degrees. Right eye: +az = nasal."""
    az_nasal: float = 56.0
    el_nasal: float = -4.0
    az_temp: float = -62.0
    el_temp: float = 2.0
    up_open: float = 24.0         # upper lid margin elevation above the canthal line, pitch 0
    up_follow: float = 0.8        # upper lid moves with the vertical gaze
    lo_open: float = 15.0         # lower lid depth below the canthal line
    lo_follow: float = 0.3
    squint_lift: float = 9.0      # lower lid lift at squint = 1

    def curves(self, az, st: EyeState):
        """Lower and upper margin elevation (deg) at azimuth az (deg); NaN outside canthi."""
        u = (az - self.az_temp) / (self.az_nasal - self.az_temp)
        inside = (u > 0) & (u < 1)
        uc = np.clip(u, 0, 1)
        base = self.el_temp + (self.el_nasal - self.el_temp) * uc
        shape_up = np.sin(np.pi * uc ** 0.85) ** 0.7       # peak a bit nasal of centre
        shape_lo = np.sin(np.pi * uc) ** 0.9
        lo = base - np.maximum(self.lo_open - self.lo_follow * min(st.pitch, 0.0)
                               - self.squint_lift * st.squint, 0.5) * shape_lo
        up_full = base + np.maximum(self.up_open + self.up_follow * st.pitch, 2.0) * shape_up
        closed = lo + 0.5 * shape_lo
        up = closed + np.clip(st.openness, 0, 1.3) * (up_full - closed)
        return np.where(inside, lo, np.nan), np.where(inside, up, np.nan)

    def aperture_deg(self, st: EyeState) -> float:
        """Opening height at the centre (deg) - ground truth for openness."""
        az = np.linspace(self.az_temp, self.az_nasal, 61)
        lo, up = self.curves(az, st)
        return float(np.nanmax(up - lo))


def ang(p):
    """Azimuth / elevation (deg) of eye-frame points as seen from the rotation centre."""
    az = np.degrees(np.arctan2(p[..., 0], p[..., 2]))
    el = np.degrees(np.arctan2(p[..., 1], np.hypot(p[..., 0], p[..., 2])))
    return az, el


@dataclass
class Plate:
    """Semi-transparent combiner (two-way mirror) as an infinite plane between eye and camera."""
    point: np.ndarray
    normal: np.ndarray
    T: float = 0.35               # transmission of one pass
    R: float = 0.55               # reflection on the camera side
    veil: float = 0.03            # light from the camera side reflected into the lens (dark mask)


@dataclass
class Glasses:
    """Spectacle lens in front of one eye (eye frame, mm). Thin lens for refraction,
    two spheres (meniscus, centres on the eye side) for the surface reflections."""
    center: tuple = (0.0, -1.0, 25.5)   # lens centre: 12 mm in front of the corneal apex
    tilt_deg: float = 8.0               # pantoscopic tilt: bottom edge closer to the face
    power_d: float = -2.0               # dioptres (minus = myopia)
    half_w: float = 25.0
    half_h: float = 16.0
    rim_mm: float = 3.0
    r_front: float = 120.0              # front (camera side) surface radius
    refl: float = 0.04                  # reflectance per surface (uncoated ~4%, AR coated ~0.5-1%)
    rim_albedo: float = 0.12

    def frame(self):
        t = np.radians(self.tilt_deg)
        nrm = np.array([0.0, -np.sin(t), np.cos(t)])   # towards the camera, top tilted forward
        up = np.array([0.0, np.cos(t), np.sin(t)])
        side = np.cross(up, nrm)
        return np.asarray(self.center, float), nrm, up, side

    def r_back(self):
        # lensmaker (n = 1.5, thin): P = 0.5 (1/R1 - 1/R2)
        inv = 1.0 / self.r_front - self.power_d / 1000.0 / 0.5
        return 1.0 / inv


@dataclass
class Scene:
    lids: Lids = field(default_factory=Lids)
    leds: list = field(default_factory=list)      # [(pos (3,), power)]
    ambient: float = 0.04
    face_z: float = 7.0                            # face plane around the eye
    cheek: tuple = ((-4.0, -34.0, -12.0), 30.0)    # sphere (centre, radius) or None
    brow: tuple = ((0.0, 30.0, -6.0), 28.0)
    plate: Plate | None = None
    plate_leds_behind: bool = False                # LEDs on the camera side of the plate
    screen_level: float = 0.0                      # AR screen corneal reflection (0 = IR filter)
    screen_fov: tuple = (40.0, 24.0)               # its angular size (deg), straight ahead
    glasses: Glasses | None = None
    led_radius: float = 2.5                        # LED emitter disc (glints traced per pixel)
    led_radiance: float = 240.0                    # x power; 2.5% corneal reflection saturates
    lash_up: int = 55
    lash_lo: int = 25
    seed: int = 1


# ----------------------------------------------------------------------------- tracing

def _trace_eye(o, d, g, pupil_r):
    """Trace rays against the eyeball + cornea. Returns t, label, shading normal, point."""
    tb0, tb1 = _sphere_hit(o, d, np.zeros(3), R_EYE)
    cc = g * D_CORNEA
    tc0, _ = _sphere_hit(o, d, cc, R_CORNEA)
    pb = o + tb0[..., None] * d
    pc = o + tc0[..., None] * d
    ball_ok = (tb0 > 0) & (np.einsum("...k,k->...", pb, g) < LIMBUS_DOT)
    corn_ok = (tc0 > 0) & (np.einsum("...k,k->...", pc, g) >= LIMBUS_DOT)
    tb = np.where(ball_ok, tb0, np.inf)
    tc = np.where(corn_ok, tc0, np.inf)
    use_c = tc < tb
    t = np.where(use_c, tc, tb)
    hit = np.isfinite(t)
    p = o + np.where(hit, t, 0)[..., None] * d
    lab = np.full(t.shape, NONE, np.int8)
    lab[hit & ~use_c] = SCLERA
    n = p / R_EYE
    # cornea: refract and hit the iris plane
    nc = (p - cc) / R_CORNEA
    cosi = -np.einsum("...k,...k->...", d, nc)
    eta = 1.0 / N_CORNEA
    k = 1 - eta ** 2 * (1 - cosi ** 2)
    dt = eta * d + (eta * cosi - np.sqrt(np.clip(k, 0, None)))[..., None] * nc
    denom = np.einsum("...k,k->...", dt, g)
    ti = (Z_IRIS - np.einsum("...k,k->...", p, g)) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    pi = p + ti[..., None] * dt
    rr = np.linalg.norm(pi - np.einsum("...k,k->...", pi, g)[..., None] * g, axis=-1)
    m = hit & use_c
    lab[m & (rr < pupil_r)] = PUPIL
    lab[m & (rr >= pupil_r) & (rr < R_IRIS)] = IRIS
    lab[m & (rr >= R_IRIS)] = SCLERA            # seen through the cornea edge
    n = np.where(m[..., None], g[None, None, :] * np.ones_like(n), n)
    p_shade = np.where(m[..., None], pi, p)
    return t, lab, n, p_shade, rr, p, m


def _iris_tex(pi, g, seed):
    """Radial-fibre iris texture factor around 1 from points on the iris plane."""
    gx = np.cross([0.0, 1.0, 0.0], g)
    gx /= np.linalg.norm(gx)
    gy = np.cross(g, gx)
    x = np.einsum("...k,k->...", pi, gx)
    y = np.einsum("...k,k->...", pi, gy)
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    rng = np.random.default_rng(seed)
    f = np.ones_like(r)
    for k, amp in zip(rng.integers(20, 90, 6), rng.uniform(0.04, 0.10, 6)):
        f += amp * np.sin(k * a + rng.uniform(0, 6.28) + 0.3 * r)
    f *= 1 - 0.45 * np.clip((r - 5.0) / 1.0, 0, 1)         # dark limbal ring
    f *= 1 + 0.25 * np.clip(1 - (r - 2.5) / 1.5, 0, 1)      # brighter collarette
    return f


class Renderer:
    def __init__(self, scene: Scene, cam: Camera, ss: int = 2):
        self.sc, self.cam, self.ss = scene, cam, ss
        self.D = cam.rays(ss)
        self.O = np.broadcast_to(cam.pos, self.D.shape)
        rng = np.random.default_rng(scene.seed)
        self.lash_up = (np.sort(rng.uniform(0.04, 0.96, scene.lash_up)),
                        rng.uniform(7.0, 10.0, scene.lash_up), rng.normal(0, 0.15, scene.lash_up))
        self.lash_lo = (np.sort(rng.uniform(0.15, 0.9, scene.lash_lo)),
                        rng.uniform(4.0, 6.5, scene.lash_lo), rng.normal(0, 0.15, scene.lash_lo))
        self.noise_rng = np.random.default_rng(scene.seed + 100)

    # ------------------------------------------------------------------ helpers
    def _light(self, p, n):
        out = np.full(p.shape[:-1], self.sc.ambient)
        for pos, pw in self.sc.leds:
            v = np.asarray(pos) - p
            d2 = np.einsum("...k,...k->...", v, v)
            lam = np.clip(np.einsum("...k,...k->...", v, n) / np.sqrt(d2), 0, None)
            out += pw * lam * (35.0 ** 2) / d2          # power 1 = full at 35 mm
        return out

    def _led_through_plate(self, pos) -> bool:
        pl = self.sc.plate
        return pl is not None and self.sc.plate_leds_behind

    # ------------------------------------------------------------------ main
    def render(self, st: EyeState, cam: Camera | None = None, look=None):
        """Grey uint8 image plus ground truth dict."""
        look = look or Look()
        cam = cam or self.cam
        if cam is not self.cam:
            D = cam.rays(self.ss)
            O = np.broadcast_to(cam.pos, D.shape)
        else:
            D, O = self.D, self.O
        sc, g = self.sc, st.g
        H, W = D.shape[:2]
        lens = None
        if sc.glasses is not None:
            lens = self._lens(O, D, cam)
            O, D = lens["O"], lens["D"]

        # surroundings
        t_face = (sc.face_z - O[..., 2]) / np.where(np.abs(D[..., 2]) < 1e-9, -1e-9, D[..., 2])
        t_face = np.where(t_face > 0, t_face, np.inf)
        t_skin0, _ = _sphere_hit(O, D, np.zeros(3), R_SKIN)
        t_skin = np.where(t_skin0 > 0, t_skin0, np.inf)
        extra = []
        for sph in (sc.cheek, sc.brow):
            if sph is None:
                extra.append(np.full((H, W), np.inf))
                continue
            t0, _ = _sphere_hit(O, D, np.asarray(sph[0], float), sph[1])
            extra.append(np.where(t0 > 0, t0, np.inf))

        # eye composite (also traced without lids for the ground-truth pupil)
        te, lab_e, n_e, ps_e, rr_e, pe, use_c = _trace_eye(O, D, g, st.pupil_r)
        te = np.where(np.isfinite(te) & (lab_e != NONE), te, np.inf)

        # lid test at the first crossing of the skin sphere (or at the cornea if it is in front)
        t_lid = np.minimum(t_skin, te)
        p_lid = O + np.where(np.isfinite(t_lid), t_lid, 0)[..., None] * D
        az, el = ang(p_lid)
        lo, up = sc.lids.curves(az, st)
        inside = (el > lo) & (el < up)
        inside &= np.isfinite(t_lid)

        t_eye_s = np.where(inside, te, np.inf)                 # eye seen through the opening
        t_lid_s = np.where(~inside & np.isfinite(t_lid), t_lid, np.inf)
        cand = np.stack([t_lid_s, t_eye_s, t_face, extra[0], extra[1]])
        k = np.argmin(cand, 0)
        tmin = np.min(cand, 0)
        lab = np.full((H, W), NONE, np.int8)
        lab[(k == 0) & np.isfinite(tmin)] = SKIN
        lab[(k == 1) & np.isfinite(tmin)] = lab_e[(k == 1) & np.isfinite(tmin)]
        lab[(k >= 2) & np.isfinite(tmin)] = FACE

        p = O + np.where(np.isfinite(tmin), tmin, 0)[..., None] * D
        n = np.zeros_like(p)
        n[k == 0] = p[k == 0] / np.linalg.norm(p[k == 0], axis=-1, keepdims=True)
        n[k == 1] = n_e[k == 1]
        n[k == 2] = [0, 0, 1]
        for i, sph in ((3, sc.cheek), (4, sc.brow)):
            if sph is not None:
                m = k == i
                q = p[m] - np.asarray(sph[0])
                n[m] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        ps = np.where((k == 1)[..., None], ps_e, p)

        alb = np.vectorize(ALB.get, otypes=[float])(lab)
        iris = lab == IRIS
        if iris.any():
            alb[iris] *= _iris_tex(ps[iris], g, sc.seed)
        # skin detail: lid crease above the upper lid, darker lid margin rim
        skin = lab == SKIN
        if skin.any():
            a_s, e_s = ang(p[skin])
            lo_s, up_s = sc.lids.curves(a_s, st)
            du = e_s - up_s
            dl = lo_s - e_s
            f = np.ones_like(e_s)
            f -= 0.35 * np.exp(-((du - 1.2) / 1.0) ** 2) * np.isfinite(du)   # margin / lash line
            f -= 0.25 * np.exp(-((dl - 0.8) / 0.8) ** 2) * np.isfinite(dl)
            crease = 14.0 * max(st.openness, 0.3)
            f -= 0.15 * np.exp(-((du - crease) / 3.0) ** 2) * np.isfinite(du)
            alb[skin] *= np.nan_to_num(f, nan=1.0)
        img = alb * self._light(ps, n)
        # sclera / iris: slightly darker towards the lid edges (lid shadow, crude)
        eyem = (lab == SCLERA) | (lab == IRIS) | (lab == PUPIL)
        if eyem.any():
            a_e, e_e = ang(p_lid[eyem])
            lo_e, up_e = sc.lids.curves(a_e, st)
            sh = 1 - 0.35 * np.exp(-(up_e - e_e) / 3.0)
            img[eyem] *= np.nan_to_num(np.clip(sh, 0.4, 1), nan=1.0)
        zbuf = np.where(np.isfinite(tmin), tmin, 1e9)
        spec_ss = None
        if lens is not None:
            zbuf = zbuf + lens["t"]                         # distances from the camera again
            spec_ss, gl_ss = self._corneal_spec(O, D, pe, use_c & (k == 1), st)

        # lashes (dark strands with a depth test), drawn at the supersampled resolution
        cover = self._lashes(st, cam, zbuf, H, W)
        img = img * (1 - cover) + cover * ALB_LASH * self._light_pts_mean(p)

        if lens is not None:
            img = np.where(lens["ap"], img * (1 - 2 * sc.glasses.refl) ** 2, img)
            img = img + spec_ss * np.where(lens["ap"], (1 - 2 * sc.glasses.refl), 1.0)
            img = img + lens["spec"]
            rimv = sc.glasses.rim_albedo * self._light(lens["q"], np.broadcast_to(lens["nrm"], lens["q"].shape))
            img = np.where(lens["rim"], rimv, img)

        # plate: transmission and camera-side light
        if sc.plate is not None:
            T = sc.plate.T * (sc.plate.T if sc.plate_leds_behind else 1.0)
            img = img * T + sc.plate.veil

        # ground-truth pupil (ignoring lids) at output resolution
        img = cv2.resize(img.astype(np.float32), (cam.w, cam.h), interpolation=cv2.INTER_AREA)
        vis_pupil = cv2.resize((lab == PUPIL).astype(np.float32), (cam.w, cam.h),
                               interpolation=cv2.INTER_AREA)
        full_pupil = cv2.resize(((lab_e == PUPIL)).astype(np.float32), (cam.w, cam.h),
                                interpolation=cv2.INTER_AREA)

        # glints (corneal reflections of the LEDs and of the AR screen)
        spec = np.zeros((cam.h, cam.w), np.float32)
        gl_px = []
        for pos, pw in (sc.leds if lens is None else []):
            q = self._glint(np.asarray(pos, float), cam, st, zbuf)
            if q is not None:
                gl_px.append(q[0])
                self._splat(spec, q[0], 0.6, 6.0 * pw * (sc.plate.T ** 2 if (sc.plate is not None and sc.plate_leds_behind) else (sc.plate.T if sc.plate else 1.0)))
        if sc.screen_level > 0:
            self._screen(spec, cam, st, zbuf)
        img = img + spec
        if sc.plate is not None and sc.plate_leds_behind:
            for pos, pw in sc.leds:     # LED mirrored by the plate straight into the lens
                pl = sc.plate
                nn = pl.normal / np.linalg.norm(pl.normal)
                mpos = np.asarray(pos) - 2 * np.dot(np.asarray(pos) - pl.point, nn) * nn
                uv, z = cam.project(mpos)
                if z > 0:
                    self._splat(img, uv, 2.5, 3.0 * pl.R * pw)

        if lens is not None:                    # ground-truth glints: centroids of the traced ones
            for m in gl_ss:
                if m.any():
                    ys, xs = np.nonzero(m)
                    gl_px.append(((xs.mean() + 0.5) / self.ss - 0.5, (ys.mean() + 0.5) / self.ss - 0.5))
        out = look.finish(img, self.noise_rng)
        gt = dict(yaw=st.yaw, pitch=st.pitch, open=sc.lids.aperture_deg(st), pupil_r=st.pupil_r,
                  vis_px=float(vis_pupil.sum()), full_px=float(full_pupil.sum()),
                  glints=gl_px)
        if full_pupil.sum() > 1:
            ys, xs = np.mgrid[:cam.h, :cam.w]
            s = full_pupil.sum()
            gt["pc_full"] = (float((full_pupil * xs).sum() / s), float((full_pupil * ys).sum() / s))
        if vis_pupil.sum() > 1:
            ys, xs = np.mgrid[:cam.h, :cam.w]
            s = vis_pupil.sum()
            gt["pc_vis"] = (float((vis_pupil * xs).sum() / s), float((vis_pupil * ys).sum() / s))
        # projection of the eyeball centre and of the true pupil centre (no refraction)
        gt["eye_c"] = tuple(cam.project(np.zeros(3))[0])
        gt["pupil_3d"] = tuple(cam.project(g * Z_IRIS)[0])
        return out, gt

    def _lens(self, O, D, cam):
        """Rays through the spectacle lens: exit point/direction, rim mask, LED reflections."""
        gl = self.sc.glasses
        c, nrm, up, side = gl.frame()
        O = np.broadcast_to(O, D.shape).astype(float)
        dn = np.einsum("...k,k->...", D, nrm)
        t = np.einsum("...k,k->...", c - O, nrm) / np.where(np.abs(dn) < 1e-9, -1e-9, dn)
        q = O + t[..., None] * D
        sv = np.einsum("...k,k->...", q - c, side)
        su = np.einsum("...k,k->...", q - c, up)
        e_in = (sv / gl.half_w) ** 2 + (su / gl.half_h) ** 2
        e_out = (sv / (gl.half_w + gl.rim_mm)) ** 2 + (su / (gl.half_h + gl.rim_mm)) ** 2
        hit = (t > 0) & (dn < 0)
        ap = hit & (e_in < 1)
        rim = hit & ~ap & (e_out < 1)
        # thin lens: transverse slopes change by -P * offset
        P = gl.power_d / 1000.0
        a = -dn
        slv = np.einsum("...k,k->...", D, side) / np.maximum(a, 1e-9) - P * sv
        slu = np.einsum("...k,k->...", D, up) / np.maximum(a, 1e-9) - P * su
        Dn = -nrm[None, None, :] + slv[..., None] * side + slu[..., None] * up
        Dn /= np.linalg.norm(Dn, axis=-1, keepdims=True)
        D2 = np.where(ap[..., None], Dn, D)
        O2 = np.where(ap[..., None], q, O)
        tt = np.where(ap, t, 0.0)
        # surface reflections of the LEDs (front and back sphere normals at the plane hit)
        spec = np.zeros(D.shape[:2])
        for rc in (gl.r_front, gl.r_back()):
            cc = c - nrm * rc
            n = (q - cc) / abs(rc)
            if rc < 0:
                n = -n
            r = D - 2 * np.einsum("...k,...k->...", D, n)[..., None] * n
            for pos, pw in self.sc.leds:
                t0, _ = _sphere_hit(q, r, np.asarray(pos, float), self.sc.led_radius)
                spec += np.where(ap & (t0 > 0), gl.refl * self.sc.led_radiance * pw, 0.0)
        return dict(O=O2, D=D2, t=tt, ap=ap, rim=rim, q=q, nrm=nrm, spec=spec)

    def _corneal_spec(self, O, D, pe, vis, st):
        """Per-pixel corneal reflection of the LED discs (2.5%), and one mask per LED."""
        cc = st.g * D_CORNEA
        n = (pe - cc) / R_CORNEA
        r = D - 2 * np.einsum("...k,...k->...", D, n)[..., None] * n
        spec = np.zeros(D.shape[:2])
        masks = []
        for pos, pw in self.sc.leds:
            t0, _ = _sphere_hit(pe, r, np.asarray(pos, float), self.sc.led_radius)
            m = vis & (t0 > 0)
            spec += np.where(m, 0.025 * self.sc.led_radiance * pw, 0.0)
            masks.append(m)
        return spec, masks

    def _light_pts_mean(self, p):
        return self._light(p, np.broadcast_to(np.array([0.0, 0.0, 1.0]), p.shape))

    def _lashes(self, st, cam, zbuf, H, W):
        sc, ss = self.sc, self.ss
        canvas = np.zeros((H * 2, W * 2), np.float32)    # 2x more for anti-aliasing
        f2 = 2 * ss
        for (us, lens, jit), upper in ((self.lash_up, True), (self.lash_lo, False)):
            az = sc.lids.az_temp + us * (sc.lids.az_nasal - sc.lids.az_temp)
            lo, up = sc.lids.curves(az, st)
            el = up if upper else lo
            ok = np.isfinite(el)
            for a, e, L, j in zip(az[ok], el[ok], lens[ok], jit[ok]):
                ar, er = np.radians(a), np.radians(e)
                root = R_SKIN * np.array([np.sin(ar) * np.cos(er), np.sin(er), np.cos(ar) * np.cos(er)])
                nrm = root / R_SKIN
                upv = np.array([0.0, 1.0, 0.0]) - nrm[1] * nrm
                upv /= np.linalg.norm(upv) + 1e-9
                s = 1.0 if upper else -1.0
                # strands leave forward, then curl away from the opening
                pts = []
                for tt in np.linspace(0, 1, 6):
                    dirv = nrm * (1.0 - 0.3 * tt) + s * upv * (0.35 + 0.9 * tt + j)
                    pts.append(root + L * tt * dirv / np.linalg.norm(dirv) * (0.6 + 0.4 * (1 - tt)))
                pts = np.array(pts)
                uv, z = cam.project(pts)
                if (z <= 0).any():
                    continue
                for i in range(len(pts) - 1):
                    m = (uv[i] + uv[i + 1]) / 2 * ss
                    xi, yi = int(m[0]), int(m[1])
                    if not (0 <= xi < W and 0 <= yi < H):
                        continue
                    dist = np.linalg.norm((pts[i] + pts[i + 1]) / 2 - cam.pos)
                    if dist > zbuf[yi, xi] + 0.3:
                        continue
                    wpx = max(1, int(round(0.12 * cam.f * f2 / dist)))
                    a0 = tuple(int(c) for c in np.round(uv[i] * f2 + (f2 - 1) / 2))
                    a1 = tuple(int(c) for c in np.round(uv[i + 1] * f2 + (f2 - 1) / 2))
                    cv2.line(canvas, a0, a1, 0.6 * (1 - 0.5 * i / 5), wpx, cv2.LINE_AA)
        canvas = np.clip(canvas, 0, 0.9)
        return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

    def _glint(self, led, cam, st, zbuf):
        """Point on the cornea that mirrors the LED into the camera (pixel, or None)."""
        cc = st.g * D_CORNEA
        a = led - cc
        b = cam.pos - cc
        n = (a / np.linalg.norm(a) + b / np.linalg.norm(b))
        n /= np.linalg.norm(n)
        for _ in range(8):
            p = cc + R_CORNEA * n
            l = led - p
            c = cam.pos - p
            n = l / np.linalg.norm(l) + c / np.linalg.norm(c)
            n /= np.linalg.norm(n)
        p = cc + R_CORNEA * n
        if np.dot(p, st.g) < LIMBUS_DOT:
            return None
        az, el = ang(p)
        lo, up = self.sc.lids.curves(np.array([az]), st)
        if not (lo[0] < el < up[0]):
            return None
        uv, z = cam.project(p)
        if z <= 0 or not (0 <= uv[0] < cam.w and 0 <= uv[1] < cam.h):
            return None
        xi, yi = int(uv[0] * self.ss), int(uv[1] * self.ss)
        if np.linalg.norm(p - cam.pos) > zbuf[min(yi, zbuf.shape[0] - 1), min(xi, zbuf.shape[1] - 1)] + 0.5:
            return None
        return uv, p

    def _screen(self, spec, cam, st, zbuf):
        """Corneal reflection of the AR screen (visible light): a far rectangle straight ahead."""
        cc = st.g * D_CORNEA
        fx, fy = self.sc.screen_fov
        for ax in np.linspace(-fx / 2, fx / 2, 24):
            for ay in np.linspace(-fy / 2, fy / 2, 14):
                src = gaze_vec(ax, ay)
                n = src + (cam.pos - cc) / np.linalg.norm(cam.pos - cc)
                n /= np.linalg.norm(n)
                for _ in range(6):
                    p = cc + R_CORNEA * n
                    c = cam.pos - p
                    n = src + c / np.linalg.norm(c)
                    n /= np.linalg.norm(n)
                p = cc + R_CORNEA * n
                if np.dot(p, st.g) < LIMBUS_DOT:
                    continue
                az, el = ang(p)
                lo, up = self.sc.lids.curves(np.array([az]), st)
                if not (lo[0] < el < up[0]):
                    continue
                uv, z = cam.project(p)
                if z > 0:
                    self._splat(spec, uv, 0.9, self.sc.screen_level * 0.05)

    @staticmethod
    def _splat(img, uv, sigma, amp):
        x0, y0 = uv
        r = int(3 * sigma + 2)
        xi, yi = int(round(x0)), int(round(y0))
        ys = slice(max(yi - r, 0), min(yi + r + 1, img.shape[0]))
        xs = slice(max(xi - r, 0), min(xi + r + 1, img.shape[1]))
        if ys.start >= ys.stop or xs.start >= xs.stop:
            return
        yy, xx = np.mgrid[ys, xs]
        img[ys, xs] += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


@dataclass
class Look:
    blur_px: float = 0.6          # defocus, Gaussian sigma
    photons: float = 1500.0        # electrons at relative brightness 1 (sets the noise)
    read_e: float = 6.0
    target_p90: float = 190.0     # auto exposure target
    max_gain: float = 8.0         # beyond this the image just gets darker

    def finish(self, img, rng):
        img = cv2.GaussianBlur(img, (0, 0), self.blur_px) if self.blur_px > 0 else img
        e = np.clip(img, 0, None) * self.photons
        e = e + rng.normal(0, 1, e.shape) * np.sqrt(e + self.read_e ** 2)
        p90 = np.percentile(img, 90) * self.photons
        gain = min(self.target_p90 / max(p90, 1e-6), self.max_gain * self.target_p90 / (0.6 * self.photons))
        return np.clip(e * gain, 0, 255).astype(np.uint8)
