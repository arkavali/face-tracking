"""Synthetic training data for comparing LED lighting variants with a trained net.

  python synth.py render [--n 3000]    -> out/synth/data.npz (per-LED linear 80x60 images + labels)
  python synth.py train                -> out/synth/report.txt (val error per lighting variant)

Each sample: a random mix of mouth deformations (jaw open, smile L/R, pucker, "o" opening, lips up,
cheek puff L/R; 1-3 active, random strength, 15 % neutral) blended as displacement fields on the
Fusion head, a random mask shift (+-5 mm) and small aim change, a smooth random albedo pattern
and +-15 % gain per LED. Every candidate LED is rendered on its own (direct + one wall bounce,
rho 0.5), so the lighting variants are built afterwards from the same samples. The small mouth
CNN (train.MouthNet, width 8,12,16,24,32) is trained per variant on the same split; the
validation error shows which light lets the net read the mouth best (1 channel = LEDs on
together, 2 channels = blinking: LED A frame + LED B frame).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import ledsim
import ledsweep2
import lipshapes
import rt
from gpu import cp
from scene import Scene, to_np

OUT = Path(__file__).resolve().parent.parent / "out" / "synth"
SLIT = lipshapes.SLIT
LABELS = ["jawOpen", "smileL", "smileR", "pucker", "funnel", "lipsUp", "cheekL", "cheekR"]


def rel_to_world(x, up, fwd):
    return np.array([x, SLIT[1] + up, SLIT[2] + fwd], float)


# candidate LEDs (x, mm above the slit, mm in front); variants are built from these
LEDS = {
    "lens": ledsim.CAM + [0, -6, 0],
    "sideL_far": rel_to_world(45, 11, 64), "sideR_far": rel_to_world(-45, 11, 64),
    "sideL_near": rel_to_world(20, 21, 54), "sideR_near": rel_to_world(-20, 21, 54),
    "up_cam": rel_to_world(0, 45, 55), "down_lip": rel_to_world(0, -25, 50),
    "nose_gap": rel_to_world(0, 35, 38),
    "down_cam": rel_to_world(0, 15, 55),
    "sideL_mid": rel_to_world(30, 5, 55), "sideR_mid": rel_to_world(-30, 5, 55),
    "sideL_low": rel_to_world(45, -9, 64), "sideR_low": rel_to_world(-45, -9, 64),
    "v_front": rel_to_world(0, 0, 69), "v_above_nose": rel_to_world(0, 60, 29),   # best v-pair, ledsweep2
}
VARIANTS = {   # name: list of channels, each channel = list of LEDs summed
    "lens": [["lens"]],
    "side far": [["sideL_far", "sideR_far"]],
    "side near": [["sideL_near", "sideR_near"]],
    "vertical": [["up_cam", "down_lip"]],
    "cross": [["sideL_near", "sideR_near", "up_cam", "down_lip"]],
    "blink side far": [["sideL_far"], ["sideR_far"]],
    "blink side near": [["sideL_near"], ["sideR_near"]],
    "blink vertical": [["up_cam"], ["down_lip"]],
    "side mid": [["sideL_mid", "sideR_mid"]],
    "side low": [["sideL_low", "sideR_low"]],
    "cam vertical": [["up_cam", "down_cam"]],
    "triple cam+nose": [["up_cam", "down_cam", "nose_gap"]],
    "blink side mid": [["sideL_mid"], ["sideR_mid"]],
    "blink cam vertical": [["up_cam"], ["down_cam"]],
    "blink side 4ch": [["sideL_near"], ["sideR_near"], ["sideL_far"], ["sideR_far"]],
    "v-pair (sweep best)": [["v_front", "v_above_nose"]],
    "cross (sweep best)": [["v_front", "v_above_nose", "sideL_far", "sideR_far"]],
    "blink v-pair": [["v_front"], ["v_above_nose"]],
}


# second study: where exactly to put the upper / lower LED of the blinking vertical pair
LEDS_V = {
    "U1": rel_to_world(0, 45, 55), "U2": rel_to_world(0, 40, 70), "U3": rel_to_world(0, 38, 40),
    "U4": rel_to_world(0, 55, 60),
    "D1": rel_to_world(0, -25, 50), "D2": rel_to_world(0, -35, 65), "D3": rel_to_world(0, -10, 45),
    "D4": rel_to_world(0, -20, 75),
    "L": rel_to_world(45, 10, 60), "R": rel_to_world(-45, 10, 60),
}
VARIANTS_V = {f"blink {u}/{d}": [[u], [d]] for u in ("U1", "U2", "U3", "U4") for d in ("D1", "D2", "D3", "D4")}
VARIANTS_V.update({
    "both U1+D1": [["U1", "D1"]],
    "control: U1+D1 twice": [["U1", "D1"], ["U1", "D1"]],     # 2 channels, no extra light info
    "blink 3ch U1/D1/sides": [["U1"], ["D1"], ["L", "R"]],
    "blink 4ch U1/D1/L/R": [["U1"], ["D1"], ["L"], ["R"]],
})
SETS = {"main": (LEDS, VARIANTS, "data.npz"), "vertical": (LEDS_V, VARIANTS_V, "data_vertical.npz")}


def side_smile(v, side, amount):
    out = v.copy()
    c = np.array([side * 25.0, SLIT[1], SLIT[2] - 6])
    d = np.linalg.norm((v - c) * [1, 1.2, 1], axis=1)
    w = lipshapes.smooth((d - 6) / 18) * lipshapes.front(v) * amount
    out[:, 0] += side * 5 * w
    out[:, 1] += 5 * w
    out[:, 2] -= 4 * w
    return out


def side_cheek(v, side, amount):
    c = np.array([side * 45.0, SLIT[1] + 8, SLIT[2] - 25])
    d = np.linalg.norm(v - c, axis=1)
    w = lipshapes.smooth((d - 8) / 22) * (v[:, 0] * side > 15)
    n = (v - c) / np.maximum(d[:, None], 1e-6)
    return v + 7.0 * amount * w[:, None] * n * [1, 0.3, 1]


def bases(v):
    """Displacement fields at full strength, in LABELS order."""
    return [
        lipshapes.jaw_open(v, 18) - v,
        side_smile(v, 1, 1.0) - v,     # x > 0 = wearer's left
        side_smile(v, -1, 1.0) - v,
        lipshapes.pucker(v) - v,
        lipshapes.lips_part(v, 9, half_w=15) - v,
        lipshapes.pucker(v, fwd=0, gather=0, up=5) - v,
        side_cheek(v, 1, 1.0) - v,
        side_cheek(v, -1, 1.0) - v,
    ]


def sample_weights(rng):
    w = np.zeros(len(LABELS), np.float32)
    if rng.random() < 0.15:
        return w
    k = rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])
    for i in rng.choice(len(LABELS), size=k, replace=False):
        w[i] = rng.uniform(0.15, 1.0)
    return w


def camera_rays(cam, aim, w=80, h=60):
    """ledsim.camera_rays at the net's own 80x60 (4x fewer rays than the 160x120 sweep renders)."""
    import sweep
    model, fov = ledsim.LENS
    f = sweep.lens_f(model, fov) * w / sweep.W
    fwd = (aim - cam) / np.linalg.norm(aim - cam)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u, v = np.meshgrid(np.arange(w) + 0.5 - w / 2, h / 2 - (np.arange(h) + 0.5))
    th = np.hypot(u, v) / f
    ph = np.arctan2(v, u)
    d = (np.sin(th) * np.cos(ph))[..., None] * right + (np.sin(th) * np.sin(ph))[..., None] * up         + np.cos(th)[..., None] * fwd
    return d.reshape(-1, 3)


def shade_all(ss, g, leds, aim):
    """ledsim.ShapeScene.shade for all LEDs in one kernel launch -> (K, N) radiance."""
    K, N = len(leds), g["p"].shape[0]
    lp = cp.asarray(leds, dtype=cp.float32)
    L = lp[:, None] - g["p"][None]                                    # (K, N, 3)
    dist = cp.linalg.norm(L, axis=2)
    l = L / cp.maximum(dist[..., None], 1e-6)
    n = g["n"][None]
    ndl = cp.clip((n * l).sum(2), 0, None)
    o = cp.broadcast_to(g["p"] + g["n"] * 0.15, (K, N, 3)).reshape(-1, 3)
    _, i, _, _ = rt.cast(ss.s0, ss.s1, ss.s2, o, l.reshape(-1, 3), tmax=(dist - 0.5).reshape(-1))
    lit = (i.reshape(K, N) < 0) & g["hit"][None] & (ndl > 0)
    ax = cp.asarray(aim, dtype=cp.float32)[None] - lp
    ax /= cp.linalg.norm(ax, axis=1, keepdims=True)
    beam = cp.clip((-l * ax[:, None]).sum(2), 0, None) ** ledsim.BEAM_M
    h = l - g["d"][None]
    h /= cp.maximum(cp.linalg.norm(h, axis=2, keepdims=True), 1e-6)
    spec = ledsim.SPEC * g["spec"][None] * cp.clip((n * h).sum(2), 0, None) ** ledsim.SHININESS
    e = beam * lit / cp.maximum(dist, 5.0) ** 2 * 1e4
    return (g["alb"][None] * ndl + spec) * e


def bounce_fast(g, leds, sh, aim, pc, pn, pa):
    """ledsweep2.bounce without the LED -> wall shadow test, all LEDs at once (LEDs sit inside the
    beak and see almost all of its inner wall)."""
    pcs, pns = cp.asarray(pc, dtype=cp.float32), cp.asarray(pn, dtype=cp.float32)
    L = cp.asarray(leds + sh, dtype=cp.float32)[None] - pcs[:, None]           # (M, K, 3)
    d = cp.linalg.norm(L, axis=2)
    l = L / d[..., None]
    cosm = cp.clip((pns[:, None] * l).sum(2), 0, None)
    ax = cp.asarray(aim - (leds + sh), dtype=cp.float32)
    ax /= cp.linalg.norm(ax, axis=1, keepdims=True)
    beam = cp.clip((-l * ax[None]).sum(2), 0, None) ** ledsim.BEAM_M
    B = beam * cosm / cp.maximum(d, 5.0) ** 2 * 1e4 * (pa / np.pi)
    P, n_p = g["p"], g["n"]
    w = pcs[None] - P[:, None]
    r2 = cp.maximum((w * w).sum(-1), 100.0)
    r = cp.sqrt(r2)
    G = cp.clip((n_p[:, None] * w).sum(-1) / r, 0, None) * cp.clip(-(pns[None] * w).sum(-1) / r, 0, None) / r2
    return (G @ B) * g["alb"][:, None]


def render(n, set_name="main"):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base, faces = lipshapes.load_indexed("head")
    B = bases(base)
    lipw = lipshapes.lip_weight(base)
    mask_tris = Scene(("mask",)).tris
    ledsweep2.N_PATCH = 600            # the bounce is a small, smooth term: fewer wall patches suffice
    pc, pn, pa = ledsweep2.wall_patches(mask_tris, np.random.default_rng(3))
    led_dict, _, fname = SETS[set_name]
    names = list(led_dict)
    leds = np.array([led_dict[k] for k in names])
    X = np.zeros((n, len(names), 60, 80), np.float16)
    Y = np.zeros((n, len(LABELS)), np.float32)
    t0 = time.time()
    for s in range(n):
        w = sample_weights(rng)
        v = base + sum(wi * bi for wi, bi in zip(w, B) if wi > 0) if w.any() else base
        ss = ledsim.ShapeScene(v, faces, lipw, mask_tris, base=base, fast_shadows=True)
        sh = rng.uniform(-5, 5, 3)
        aim = ledsim.AIM + sh + rng.uniform(-3, 3, 3)
        cam = ledsim.CAM + sh
        g = ss.primary(cam, camera_rays(cam, aim))
        tex = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (80, 60), interpolation=cv2.INTER_CUBIC)
        tex = np.clip(1 + 0.25 * tex, 0.4, 1.8)
        img = shade_all(ss, g, leds + sh, aim) + 0.5 * bounce_fast(g, leds, sh, aim, pc, pn, pa).T
        img = to_np(img).reshape(len(leds), 60, 80)
        X[s] = img * tex[None] * rng.uniform(0.85, 1.15, (len(leds), 1, 1))
        Y[s] = w
        if s % 200 == 0:
            print(f"  {s}/{n}  {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT / fname, X=X, Y=Y, leds=np.array(names), labels=np.array(LABELS))
    print(f"rendered {n} samples in {time.time() - t0:.0f} s")


def build_inputs(X, names, variant, rng, variants=None):
    """Exposed 8-bit-like inputs (N, C, 60, 80) for a lighting variant."""
    idx = {k: i for i, k in enumerate(names)}
    chans = [X[:, [idx[k] for k in ch]].astype(np.float32).sum(1) for ch in (variants or VARIANTS)[variant]]
    stack = np.stack(chans, 1)
    # one locked exposure per sample over all channels (98th percentile of the brightest channel)
    scale = np.percentile(stack.max(1).reshape(len(X), -1), 98, axis=1)[:, None, None, None]
    img = np.clip(stack / np.maximum(scale, 1e-9), 0, 1)
    img = img + rng.normal(0, 0.01, img.shape)                     # sensor noise
    return np.round(np.clip(img, 0, 1) * 255) / 255


def train_one(X, Y, names, variant, trn, va, epochs, seed, dev, variants=None):
    """Train train.MouthNet on one lighting variant -> per-label val MAE (best of the last epochs)."""
    import torch
    import torch.nn.functional as F
    import train as tr
    torch.manual_seed(seed)
    Xi = build_inputs(X, names, variant, np.random.default_rng(1), variants)
    c_in = Xi.shape[1]
    model = tr.MouthNet(n_out=len(LABELS), width=(8, 12, 16, 24, 32)).to(dev)
    if c_in != 1:
        first = model.features[0]
        model.features[0] = torch.nn.Conv2d(c_in, first.out_channels, 3, padding=1, bias=False).to(dev)
    xt = torch.tensor(Xi[trn], dtype=torch.float32, device=dev)
    yt = torch.tensor(Y[trn], device=dev)
    xv = torch.tensor(Xi[va], dtype=torch.float32, device=dev)
    yv = Y[va]
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    steps = epochs * (len(xt) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    best = None
    for ep in range(epochs):
        model.train()
        p = torch.randperm(len(xt), device=dev)
        for a in range(0, len(xt) - 63, 64):
            i = p[a:a + 64]
            loss = F.mse_loss(model(xt[i]), yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep >= epochs - 10 or ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                pv = model(xv).cpu().numpy()
            mae = np.abs(pv - yv).mean(0)
            if best is None or mae.mean() < best.mean():
                best = mae
    return best


def train_variants(epochs, only=None, seeds=(0,), out_name="report.txt", set_name="main"):
    import torch
    _, variants, fname = SETS[set_name]
    d = np.load(OUT / fname)
    X, Y, names = d["X"], d["Y"], list(d["leds"])
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    n_val = n // 5
    va, trn = perm[:n_val], perm[n_val:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lines = [f"{n} synthetic samples ({len(trn)} train / {n_val} val), {epochs} epochs, net width 8,12,16,24,32, "
             f"seeds {list(seeds)} (mean; +- = std of the 'all' column over seeds)",
             "val MAE per label (0..1 scale); lower is better", "",
             f"{'variant':20s} {'all':>6s} {'+-':>6s} " + " ".join(f"{l:>8s}" for l in LABELS)]
    for variant in (only or variants):
        runs = np.array([train_one(X, Y, names, variant, trn, va, epochs, seed, dev, variants) for seed in seeds])
        mean = runs.mean(0)
        line = f"{variant:20s} {mean.mean():6.3f} {runs.mean(1).std():6.3f} " + " ".join(f"{m:8.3f}" for m in mean)
        lines.append(line)
        print(line, flush=True)
    (OUT / out_name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "train"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--variants", default=None, help="comma list of variant names (default all)")
    ap.add_argument("--seeds", default="0", help="comma list of training seeds")
    ap.add_argument("--report", default="report.txt")
    ap.add_argument("--set", default="main", choices=list(SETS))
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.n, args.set)
    else:
        only = args.variants.split(",") if args.variants else None
        train_variants(args.epochs, only, tuple(int(v) for v in args.seeds.split(",")), args.report, args.set)


if __name__ == "__main__":
    main()
