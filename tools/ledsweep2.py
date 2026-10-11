"""LED layout sweep, round 2: one bounce off the inner beak wall, a denser grid near the camera,
vertical layouts, and the blinking (2-channel) check on the best layouts.

  python ledsweep2.py [--rho 0.5]   -> out/led2/report.txt, layouts.png, map.png, blink.txt

Light model per LED: direct (point source, cos^2 beam at the mouth, shadows, wet-lip highlight,
as ledsim.py) + one diffuse bounce off the inner wall of the beak: the wall is sampled into
~2000 patches, each lit by the LED (with shadows) and re-emitting as a Lambertian surface of
albedo rho towards the face (no occlusion test on that second leg). Direct and bounce images are
stored separately per LED, so any wall albedo can be evaluated afterwards.
Layouts: LED at the lens, single centre LEDs, horizontal pairs (+-x), vertical pairs (two centre
LEDs), triples (vertical pair near the camera + one between camera and nose), crosses (best
horizontal pair + best vertical pair).
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ledsim
import lipshapes
import rt
import sweep
from gpu import cp
from scene import Scene, to_np

OUT = Path(__file__).resolve().parent.parent / "out" / "led2"
H, W = ledsim.H, ledsim.W
SLIT = lipshapes.SLIT
CAM, AIM = ledsim.CAM, ledsim.AIM
SHIFTS = ledsim.SHIFTS
REF = np.array([0.0, 110.0, 295.0])      # a point inside the beak cavity (wall orientation)
N_PATCH = 2000
FAN_Z = 340.0
SHOW = ["neutral", "pucker", "funnel", "jaw", "smile", "cheeks"]


def rel(p):
    """World mm -> (x, mm above the slit, mm in front of the slit)."""
    return np.array([p[0], p[1] - SLIT[1], p[2] - SLIT[2]])


def wall_patches(mask_tris, rng):
    c = mask_tris.mean(1)
    n = np.cross(mask_tris[:, 1] - mask_tris[:, 0], mask_tris[:, 2] - mask_tris[:, 0])
    a = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2 * a[:, None], 1e-12)
    sel = ((c[:, 2] > SLIT[2] - 10) & (np.abs(c[:, 0]) < 120) & (c[:, 1] > -10) & (c[:, 1] < 230)
           & ((n * (REF - c)).sum(1) > 0))           # inner surface (normals point into the cavity)
    idx = np.where(sel)[0]
    pick = rng.choice(idx, size=min(N_PATCH, len(idx)), replace=False, p=a[idx] / a[idx].sum())
    return c[pick], n[pick], a[idx].sum() / len(pick)


def candidates(mask_sc, face):
    xs_, ys_, fz = face
    pts = []
    for x in (0.0, 10.0, 20.0, 30.0, 45.0):
        for y in np.arange(SLIT[1] - 40, SLIT[1] + 61, 10.0):
            for z in np.arange(265.0, FAN_Z + 1, 10.0):
                fzv = fz[np.argmin(abs(ys_ - y)), np.argmin(abs(xs_ - x))]
                if np.isfinite(fzv) and z < fzv + 8:
                    continue
                t, _ = mask_sc.cast(np.array([[x, y, z]]), np.array([[0, 0, 1.0]]))
                t = float(to_np(t)[0])
                if not np.isfinite(t) or t < 6:
                    continue
                if np.linalg.norm(np.array([x, y, z]) - CAM) < 8:
                    continue
                pts.append((x, y, z))
    return np.array(pts)


def bounce(ss, g, leds, sh, pc, pn, pa):
    """(N_pixels, n_led) bounce radiance for unit wall albedo."""
    pcs, pns = cp.asarray(pc, dtype=cp.float32), cp.asarray(pn, dtype=cp.float32)
    E = cp.zeros((len(pc), len(leds)), cp.float32)
    for k, led in enumerate(leds):
        lp = led + sh
        L = cp.asarray(lp, dtype=cp.float32)[None] - pcs
        d = cp.linalg.norm(L, axis=1)
        l = L / d[:, None]
        cosm = cp.clip((pns * l).sum(1), 0, None)
        _, i, _, _ = rt.cast(ss.v0, ss.e1, ss.e2, lp[None], -l, tmax=d - 1.0)
        ax = (AIM + sh) - lp; ax = ax / np.linalg.norm(ax)
        beam = cp.clip((-l * cp.asarray(ax, dtype=cp.float32)[None]).sum(1), 0, None) ** ledsim.BEAM_M
        E[:, k] = beam * cosm * (i < 0) / cp.maximum(d, 5.0) ** 2 * 1e4
    B = E * (pa / np.pi)                                   # patch radiance x area (rho = 1)
    P, n_p, alb = g["p"], g["n"], g["alb"]
    out = cp.zeros((P.shape[0], len(leds)), cp.float32)
    for a in range(0, P.shape[0], 4096):
        w = pcs[None] - P[a:a + 4096, None]
        r2 = cp.maximum((w * w).sum(-1), 100.0)
        r = cp.sqrt(r2)
        cp_ = cp.clip((n_p[a:a + 4096, None] * w).sum(-1) / r, 0, None)
        cm = cp.clip(-(pns[None] * w).sum(-1) / r, 0, None)
        G = cp_ * cm / r2
        out[a:a + 4096] = (G @ B) * alb[a:a + 4096, None]
    return out


def sep_score(feat):
    """feat: {shape: (n_shift, F)} -> mean, min log separation over ledsim.PAIRS, per-pair list."""
    m = {k: v.mean(0) for k, v in feat.items()}
    var = {k: ((v - m[k]) ** 2).sum(1).mean() for k, v in feat.items()}
    seps = [float(np.log(((m[a] - m[b]) ** 2).sum() / (0.5 * (var[a] + var[b]) + 1e-9))) for a, b in ledsim.PAIRS]
    return float(np.mean(seps)), float(np.min(seps)), seps


def small(img):
    return img.reshape(*img.shape[:-2], 60, 2, 80, 2).mean((-3, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.5, help="inner wall albedo in NIR")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(3)
    shapes, faces, lipw = lipshapes.make_shapes()
    names = list(shapes)
    mask_tris = Scene(("mask",)).tris
    mask_sc = Scene(("mask",))
    face = sweep.face_map_head_only(None)
    cand = candidates(mask_sc, face)
    full = [tuple(p) for p in cand] + [(-p[0], p[1], p[2]) for p in cand if p[0] > 0]
    full = np.array(full + [tuple(CAM + [0, -6, 0])])
    cam_led = len(full) - 1
    mirror = {k: int(np.argmin(np.linalg.norm(full - [-p[0], p[1], p[2]], axis=1))) for k, p in enumerate(full)}
    pc, pn, pa = wall_patches(mask_tris, rng)
    print(f"{len(full)} LED positions, {len(pc)} wall patches, {len(names)} shapes x {len(SHIFTS)} shifts", flush=True)

    direct = {n: np.zeros((len(full), len(SHIFTS), H, W), np.float16) for n in names}
    indir = {n: np.zeros((len(full), len(SHIFTS), H, W), np.float16) for n in names}
    roi = None
    for name, verts in shapes.items():
        ss = ledsim.ShapeScene(verts, faces, lipw, mask_tris, base=shapes["neutral"])
        for si, sh in enumerate(SHIFTS):
            g = ss.primary(CAM + sh, ledsim.camera_rays(CAM + sh, AIM + sh))
            if roi is None:
                roi = to_np(g["hit"]).reshape(H, W) & (to_np(g["p"][:, 1]).reshape(H, W) < 125)
            for k, led in enumerate(full):
                direct[name][k, si] = to_np(ss.shade(g, led + sh, AIM + sh)).reshape(H, W)
            ind = to_np(bounce(ss, g, full, sh, pc, pn, pa))
            indir[name][:, si] = ind.T.reshape(len(full), H, W)
        print(f"  {name}: {time.time() - t0:.0f} s", flush=True)

    def raw(n, leds, rho=args.rho):
        """(n_shift, H, W) linear image for a set of LEDs."""
        return (direct[n][leds].astype(np.float32) + rho * indir[n][leds].astype(np.float32)).sum(0)

    def exposed(leds, rho=args.rho):
        return {n: np.stack([ledsim.auto_expose(im, roi) for im in raw(n, leds, rho)]) for n in names}

    def evaluate(leds, rho=args.rho):
        imgs = exposed(list(leds), rho)
        mean, mn, seps = sep_score({n: small(v).reshape(len(SHIFTS), -1) for n, v in imgs.items()})
        dark = float(np.mean([(v[:, roi] < 0.06).mean() for v in imgs.values()]))
        return mean, mn, dark, seps

    centres = [k for k in range(len(full)) if full[k][0] == 0]
    hpairs = sorted({tuple(sorted((k, mirror[k]))) for k in range(len(full)) if full[k][0] > 0})
    layouts = [("lens", (cam_led,))] + [("centre", (k,)) for k in centres if k != cam_led]
    layouts += [("h-pair", p) for p in hpairs]
    vpairs = [(a, b) for a, b in itertools.combinations(centres, 2) if abs(full[a][1] - full[b][1]) >= 20]
    layouts += [("v-pair", p) for p in vpairs]
    near_cam = [k for k in centres if np.linalg.norm(full[k] - CAM) < 30 and k != cam_led] + [cam_led]
    nose_gap = [k for k in centres if 25 <= rel(full[k])[1] <= 50 and 20 <= rel(full[k])[2] <= 50]
    for a, b in itertools.combinations(near_cam, 2):
        if abs(full[a][1] - full[b][1]) >= 12:
            layouts += [("triple", (a, b, c)) for c in nose_gap if c not in (a, b)]
    print(f"{len(layouts)} layouts to score", flush=True)
    res = []
    for kind, leds in layouts:
        mean, mn, dark, seps = evaluate(leds)
        res.append(dict(kind=kind, leds=tuple(leds), mean=mean, min=mn, dark=dark, seps=seps, score=mean - 2 * dark))
    best_h = max((r for r in res if r["kind"] == "h-pair"), key=lambda r: r["score"])
    best_v = max((r for r in res if r["kind"] == "v-pair"), key=lambda r: r["score"])
    for hp in sorted((r for r in res if r["kind"] == "h-pair"), key=lambda r: -r["score"])[:5]:
        for vp in sorted((r for r in res if r["kind"] == "v-pair"), key=lambda r: -r["score"])[:5]:
            leds = tuple(sorted(set(hp["leds"] + vp["leds"])))
            mean, mn, dark, seps = evaluate(leds)
            res.append(dict(kind="cross", leds=leds, mean=mean, min=mn, dark=dark, seps=seps, score=mean - 2 * dark))
    res.sort(key=lambda r: -r["score"])
    base = next(r for r in res if r["kind"] == "lens")

    def desc(leds):
        return " + ".join("({:+.0f},{:+.0f},{:.0f})".format(*rel(full[k])) for k in leds)

    def desc_cam(leds):
        return " + ".join("({:+.0f},{:+.0f},{:+.0f})".format(*(full[k] - CAM)) for k in leds)

    lines = [f"wall albedo rho = {args.rho}; camera at {rel(CAM)} (x, up, forward from the mouth slit); score = "
             "mean log separation - 2 x dark share (see ledsim.py)", "positions: from the slit | from the camera", "",
             f"baseline LED at the lens: score {base['score']:.2f} (sep {base['mean']:.2f}, dark {base['dark']:.2f})", ""]
    top_by_kind = {}
    for kind in ("h-pair", "v-pair", "triple", "cross", "centre"):
        rk = [r for r in res if r["kind"] == kind][:5]
        if not rk:
            continue
        top_by_kind[kind] = rk[0]
        lines.append(f"== best {kind}")
        for r in rk:
            lines.append(f"  {r['score']:5.2f} (sep {r['mean']:5.2f}, min {r['min']:5.2f}, dark {r['dark']:.2f})  "
                         f"{desc(r['leds'])}  |  {desc_cam(r['leds'])}")
        lines.append("")
    # does the wall bounce change the ranking? same top layouts without it
    lines.append("== same layouts without the wall bounce (rho = 0) / with a light wall (rho = 0.8)")
    for kind, r in top_by_kind.items():
        s0 = evaluate(r["leds"], 0.0)
        s8 = evaluate(r["leds"], 0.8)
        lines.append(f"  {kind:7s} rho 0: {s0[0] - 2 * s0[2]:5.2f} (dark {s0[2]:.2f})   rho {args.rho}: {r['score']:5.2f} "
                     f"(dark {r['dark']:.2f})   rho 0.8: {s8[0] - 2 * s8[2]:5.2f} (dark {s8[2]:.2f})")
    b0 = evaluate(base["leds"], 0.0)
    lines.append(f"  lens    rho 0: {b0[0] - 2 * b0[2]:5.2f} (dark {b0[2]:.2f})   rho {args.rho}: {base['score']:5.2f} (dark {base['dark']:.2f})")
    lines.append("")

    # blinking: the two LEDs of the best pairs as 2 channels vs both on
    lines.append("== blinking: LED A frame + LED B frame as 2 channels (locked exposure) vs both on, sep mean / min")
    blink_set = [r for r in res if r["kind"] in ("h-pair", "v-pair")][:6]
    for r in blink_set:
        a, b = r["leds"]
        f2 = {}
        for n in names:
            ia, ib = raw(n, [a]), raw(n, [b])
            sc = np.array([np.percentile(np.maximum(x, y)[roi], 98) for x, y in zip(ia, ib)])[:, None, None]
            f2[n] = np.concatenate([small(np.clip(ia / sc, 0, 1)).reshape(len(SHIFTS), -1),
                                    small(np.clip(ib / sc, 0, 1)).reshape(len(SHIFTS), -1)], 1)
        m2, mn2, _ = sep_score(f2)
        lines.append(f"  {r['kind']:6s} {desc(r['leds'])}: both {r['mean']:5.2f} / {r['min']:5.2f}   2ch {m2:5.2f} / {mn2:5.2f}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # picture: per layout one row = each LED alone (neutral) | all on: SHOW shapes; LED marks in the image
    def proj(p):
        model, fov = ledsim.LENS
        return sweep.project(p[None], CAM, AIM - CAM, model, sweep.lens_f(model, fov))[0] * W / sweep.W

    rows = []
    show = [("lens (baseline)", base)] + [(k, v) for k, v in top_by_kind.items() if k != "centre"]
    max_leds = max(len(r["leds"]) for _, r in show)
    for label, r in show:
        tiles = []
        for k in r["leds"]:
            im = ledsim.auto_expose(raw("neutral", [k])[0], roi)
            tiles.append(im)
        tiles += [np.zeros((H, W))] * (max_leds - len(r["leds"]))
        allim = exposed(list(r["leds"]))
        tiles += [allim[n][0] for n in SHOW]
        t = [cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for x in tiles]
        t = [cv2.resize(x, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST) for x in t]
        for j, k in enumerate(r["leds"]):
            cv2.putText(t[j], "LED ({:+.0f},{:+.0f},{:.0f})".format(*rel(full[k])), (3, 14), 0, 0.42, (0, 200, 255), 1)
        for j, n in enumerate(SHOW):
            img = t[max_leds + j]
            cv2.putText(img, n, (3, 14), 0, 0.45, (0, 255, 255), 1)
            for k in r["leds"]:
                u, v = proj(full[k]) * 2
                if 0 <= u < W * 2 and 0 <= v < H * 2:
                    cv2.circle(img, (int(u), int(v)), 4, (0, 140, 255), -1)
        cv2.putText(t[0], f"{label}: {r['score']:.2f}", (3, H * 2 - 8), 0, 0.45, (0, 255, 0), 1)
        rows.append(np.hstack(t))
    cv2.imwrite(str(OUT / "layouts.png"), np.vstack(rows))

    # map: side view (z, y) with face profile + beak wall, front view (x, y); best layouts marked
    xs_, ys_, fz = face
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    prof = fz[:, xs_ == 0].ravel()
    axs[0].plot(prof - SLIT[2], ys_ - SLIT[1], "k-", lw=1, label="face (midline)")
    tw, _ = mask_sc.cast(np.stack([np.zeros_like(ys_), ys_, np.full_like(ys_, SLIT[2])], 1), np.tile([0, 0, 1.0], (len(ys_), 1)))
    axs[0].plot(SLIT[2] + to_np(tw) - SLIT[2], ys_ - SLIT[1], color="0.5", lw=1, label="beak wall (midline)")
    axs[0].axvline(FAN_Z - SLIT[2], color="r", ls="--", lw=0.8, label="fan zone")
    for ax in axs:
        ax.plot(*([rel(CAM)[2], rel(CAM)[1]] if ax is axs[0] else [0, rel(CAM)[1]]), "ks", ms=9, label="camera")
    colours = {"h-pair": "tab:blue", "v-pair": "tab:green", "triple": "tab:purple", "cross": "tab:orange"}
    for kind, r in top_by_kind.items():
        if kind not in colours:
            continue
        P = np.array([rel(full[k]) for k in r["leds"]])
        axs[0].plot(P[:, 2], P[:, 1], "o", color=colours[kind], ms=8, alpha=0.8, label=kind)
        axs[1].plot(P[:, 0], P[:, 1], "o", color=colours[kind], ms=8, alpha=0.8, label=kind)
    C = np.array([rel(p) for p in full])
    axs[0].plot(C[:, 2], C[:, 1], ".", color="0.8", ms=3, zorder=0)
    axs[1].plot(C[:, 0], C[:, 1], ".", color="0.8", ms=3, zorder=0)
    axs[0].set_xlabel("mm in front of the mouth slit"); axs[0].set_ylabel("mm above the slit"); axs[0].set_title("side view")
    axs[1].set_xlabel("x, mm (wearer's left > 0)"); axs[1].set_ylabel("mm above the slit"); axs[1].set_title("front view")
    axs[1].plot([-25, 25], [0, 0], "k-", lw=3, label="mouth")
    for ax in axs:
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(OUT / "map.png", dpi=80); plt.close(fig)
    np.savez(OUT / "leds.npz", full=full)
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
