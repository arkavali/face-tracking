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
}


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


def render(n):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    base, faces = lipshapes.load_indexed("head")
    B = bases(base)
    lipw = lipshapes.lip_weight(base)
    mask_tris = Scene(("mask",)).tris
    pc, pn, pa = ledsweep2.wall_patches(mask_tris, np.random.default_rng(3))
    names = list(LEDS)
    leds = np.array([LEDS[k] for k in names])
    X = np.zeros((n, len(names), 60, 80), np.float16)
    Y = np.zeros((n, len(LABELS)), np.float32)
    t0 = time.time()
    for s in range(n):
        w = sample_weights(rng)
        v = base + sum(wi * bi for wi, bi in zip(w, B) if wi > 0) if w.any() else base
        ss = ledsim.ShapeScene(v, faces, lipw, mask_tris, base=base)
        sh = rng.uniform(-5, 5, 3)
        aim = ledsim.AIM + sh + rng.uniform(-3, 3, 3)
        cam = ledsim.CAM + sh
        g = ss.primary(cam, ledsim.camera_rays(cam, aim))
        tex = cv2.resize(rng.normal(0, 1, (6, 8)).astype(np.float32), (ledsim.W, ledsim.H), interpolation=cv2.INTER_CUBIC)
        tex = np.clip(1 + 0.25 * tex, 0.4, 1.8)
        ind = to_np(ledsweep2.bounce(ss, g, leds, sh, pc, pn, pa))
        for k, led in enumerate(leds):
            img = to_np(ss.shade(g, led + sh, aim)) + 0.5 * ind[:, k]
            img = img.reshape(ledsim.H, ledsim.W) * tex * rng.uniform(0.85, 1.15)
            X[s, k] = img.reshape(60, 2, 80, 2).mean((1, 3))
        Y[s] = w
        if s % 200 == 0:
            print(f"  {s}/{n}  {time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(OUT / "data.npz", X=X, Y=Y, leds=np.array(names), labels=np.array(LABELS))
    print(f"rendered {n} samples in {time.time() - t0:.0f} s")


def build_inputs(X, names, variant, rng):
    """Exposed 8-bit-like inputs (N, C, 60, 80) for a lighting variant."""
    idx = {k: i for i, k in enumerate(names)}
    chans = [X[:, [idx[k] for k in ch]].astype(np.float32).sum(1) for ch in VARIANTS[variant]]
    stack = np.stack(chans, 1)
    # one locked exposure per sample over all channels (98th percentile of the brightest channel)
    scale = np.percentile(stack.max(1).reshape(len(X), -1), 98, axis=1)[:, None, None, None]
    img = np.clip(stack / np.maximum(scale, 1e-9), 0, 1)
    img = img + rng.normal(0, 0.01, img.shape)                     # sensor noise
    return np.round(np.clip(img, 0, 1) * 255) / 255


def train_variants(epochs):
    import torch
    import torch.nn.functional as F
    import train as tr
    d = np.load(OUT / "data.npz")
    X, Y, names = d["X"], d["Y"], list(d["leds"])
    n = len(X)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = n // 5
    va, trn = perm[:n_val], perm[n_val:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lines = [f"{n} synthetic samples ({len(trn)} train / {n_val} val), {epochs} epochs, net width 8,12,16,24,32",
             "val MAE per label (0..1 scale); lower is better", "",
             f"{'variant':16s} {'all':>6s} " + " ".join(f"{l:>8s}" for l in LABELS)]
    for variant in VARIANTS:
        torch.manual_seed(0)
        Xi = build_inputs(X, names, variant, np.random.default_rng(1))
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
        line = f"{variant:16s} {best.mean():6.3f} " + " ".join(f"{m:8.3f}" for m in best)
        lines.append(line)
        print(line, flush=True)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "train"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    if args.cmd == "render":
        render(args.n)
    else:
        train_variants(args.epochs)


if __name__ == "__main__":
    main()
