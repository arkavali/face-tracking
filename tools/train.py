"""Train the small mouth CNN on prompted calibration recordings (out/calib/cal_*).

  python train.py [--epochs 60] [--val cal_...]     -> out/model/mouth.pt, mouth.onnx, report.txt

Input: whole grey frame resized to IN_W x IN_H (the camera is fixed in the mask, so no face
detection or crop). Output: the CHANNELS of calib_rec.py, sigmoid 0..1. The net is sized for the
ESP32(-S3) (~10 M MAC); int8 export for the chip comes later.
Labels are shifted by --lag seconds: the wearer follows the prompt with a delay.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calib_rec import CHANNELS

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "out" / "calib"
OUT = ROOT / "out" / "model"
IN_W, IN_H = 80, 60   # exact 4x box downscale of 320x240 (same on the chip)


class MouthNet(nn.Module):
    """5 conv blocks (conv3x3 + BN + ReLU, pool) -> global average -> linear -> sigmoid."""

    def __init__(self, n_out=len(CHANNELS), width=(16, 24, 32, 48, 64)):
        super().__init__()
        layers, c = [], 1
        for i, w in enumerate(width):
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            if i < len(width) - 1:
                layers.append(nn.MaxPool2d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_out)

    def forward(self, x):
        x = self.features(x)
        x = x.mean((2, 3))
        return torch.sigmoid(self.head(x))


def macs(model):
    total, h, w = 0, IN_H, IN_W
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            total += h * w * m.in_channels * m.out_channels * 9
        if isinstance(m, nn.MaxPool2d):
            h, w = h // 2, w // 2
    return total


def load_session(d: Path, lag: float, keep_rest: bool = False):
    raw = (d / "frames.mjpeg").read_bytes()
    frows = list(csv.DictReader(open(d / "frames.csv")))
    lrows = list(csv.DictReader(open(d / "labels.csv", encoding="utf-8")))
    imgs = np.empty((len(frows), IN_H, IN_W), np.uint8)
    for i, r in enumerate(frows):
        o, n = int(r["offset"]), int(r["len"])
        g = cv2.imdecode(np.frombuffer(raw[o:o + n], np.uint8), cv2.IMREAD_GRAYSCALE)
        imgs[i] = cv2.resize(g, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    y = np.array([[float(r[c]) for c in CHANNELS] for r in lrows], np.float32)
    if "valid" in lrows[0]:                        # rewound / redone prompts
        y[np.array([r["valid"] == "0" for r in lrows])] = np.nan
    if not keep_rest:   # the face is not always neutral between prompts; "neutral" covers zero
        y[np.array([r["phase"] == "rest" for r in lrows])] = np.nan
    t = np.array([float(r["pc_time"]) for r in frows])
    # label at time t applies to the frame at t + lag
    if lag > 0:
        src = np.searchsorted(t, t - lag)
        y = y[np.clip(src, 0, len(y) - 1)]
        y[t - lag < t[0]] = 0
        # frames right after a pause / rewind: their shifted label belongs to another take
        y[np.abs(t - lag - t[np.clip(src, 0, len(t) - 1)]) > 0.2] = np.nan
    prompts = [r["prompt"] for r in lrows]
    return imgs, y, prompts


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, W) in 0..1 on the GPU. Shift / scale / rotate, brightness, contrast, gamma,
    blur and noise: the mask will sit a bit differently and the IR light will differ."""
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) - 0.5) * np.radians(12)
    sc = 1 + (torch.rand(b, device=dev) - 0.5) * 0.2
    tx = (torch.rand(b, device=dev) - 0.5) * 0.16
    ty = (torch.rand(b, device=dev) - 0.5) * 0.16
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    gamma = torch.exp((torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.8)
    x = x.clamp(1e-4, 1) ** gamma
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.6
    bright = (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.3
    m = x.mean((2, 3), keepdim=True)
    x = (x - m) * contrast + m + bright
    if torch.rand(1).item() < 0.5:
        x = F.avg_pool2d(x, 3, 1, 1)
    x = x + torch.randn_like(x) * 0.02 * torch.rand(b, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lag", type=float, default=0.35)
    ap.add_argument("--val", help="session dir name held out for validation (default: the last one)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--width", default="16,24,32,48,64", help="conv channels per block")
    ap.add_argument("--all", action="store_true", help="train on all sessions, validate on held-out prompt blocks")
    ap.add_argument("--keep-rest", action="store_true", help="train on the rest phases as neutral too")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = sorted(p for p in CALIB.glob("cal_*") if (p / "labels.csv").exists()
                      and sum(1 for _ in open(p / "frames.csv")) > 50)
    if not sessions:
        raise SystemExit("no recordings in out/calib")
    val_name = None if args.all else args.val or (sessions[-1].name if len(sessions) > 1 else None)
    tr_x, tr_y, va_x, va_y, va_p = [], [], [], [], []
    for s in sessions:
        x, y, p = load_session(s, args.lag, args.keep_rest)
        ok = ~np.isnan(y).any(1)
        if s.name == val_name:
            va_x.append(x[ok]); va_y.append(y[ok]); va_p += [q for q, k in zip(p, ok) if k]
        elif args.all:      # every session: every 5th prompt block goes to validation
            pp = [q for q, k in zip(p, ok) if k]
            blocks = np.cumsum([0] + [a != b for a, b in zip(pp, pp[1:])])
            hold = blocks % 5 == 2
            tr_x.append(x[ok][~hold]); tr_y.append(y[ok][~hold])
            va_x.append(x[ok][hold]); va_y.append(y[ok][hold])
        else:
            tr_x.append(x[ok]); tr_y.append(y[ok])
        print(f"{s.name}: {len(x)} frames, {ok.sum()} labelled{' (validation)' if s.name == val_name else ''}")
    if not tr_x and va_p:              # a single session: hold out every 5th prompt block
        x, y = va_x[0], va_y[0]
        blocks = np.cumsum([0] + [a != b for a, b in zip(va_p, va_p[1:])])
        hold = blocks % 5 == 4
        tr_x, tr_y, va_x, va_y = [x[~hold]], [y[~hold]], [x[hold]], [y[hold]]
    Xtr = torch.tensor(np.concatenate(tr_x)[:, None], dtype=torch.float32, device=dev) / 255
    Ytr = torch.tensor(np.concatenate(tr_y), device=dev)
    Xva = torch.tensor(np.concatenate(va_x)[:, None], dtype=torch.float32, device=dev) / 255
    Yva = torch.tensor(np.concatenate(va_y), device=dev)

    width = tuple(int(v) for v in args.width.split(","))
    model = MouthNet(width=width).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())}, MAC {macs(model) / 1e6:.1f} M, "
          f"train {len(Xtr)}, val {len(Xva)}, device {dev}")
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    steps = args.epochs * max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    best, step = 1e9, 0
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tl = 0.0
        for a in range(0, len(Xtr) - args.batch + 1, args.batch):
            i = perm[a:a + args.batch]
            loss = F.mse_loss(model(augment(Xtr[i])), Ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            tl += loss.item()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xva[a:a + 512]) for a in range(0, len(Xva), 512)])
            vl = F.mse_loss(pv, Yva).item()
        if vl < best:
            best = vl
            torch.save(model.state_dict(), OUT / "mouth.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}  train {tl / max(1, len(Xtr) // args.batch):.4f}  val {vl:.4f}")

    model.load_state_dict(torch.load(OUT / "mouth.pt"))
    model.eval()
    with torch.no_grad():
        pv = torch.cat([model(Xva[a:a + 512]) for a in range(0, len(Xva), 512)]).cpu().numpy()
    yv = Yva.cpu().numpy()
    lines = [f"val MSE {best:.4f}, lag {args.lag} s, {args.epochs} epochs, MAC {macs(model) / 1e6:.1f} M",
             "channel        MAE    corr   (val frames where the channel is active / all)"]
    for k, c in enumerate(CHANNELS):
        act = yv[:, k] > 0.05
        corr = np.corrcoef(pv[:, k], yv[:, k])[0, 1] if yv[:, k].std() > 0 else float("nan")
        lines.append(f"{c:13s} {np.abs(pv[:, k] - yv[:, k]).mean():.3f}  {corr:6.2f}   {act.sum()}/{len(yv)}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    torch.onnx.export(model.cpu(), torch.zeros(1, 1, IN_H, IN_W), OUT / "mouth.onnx",
                      input_names=["img"], output_names=["out"], dynamic_axes={"img": {0: "b"}, "out": {0: "b"}})
    (OUT / "mouth.json").write_text(json.dumps({"channels": CHANNELS, "in_w": IN_W, "in_h": IN_H, "width": width}))
    print("saved", OUT / "mouth.onnx")


if __name__ == "__main__":
    main()
