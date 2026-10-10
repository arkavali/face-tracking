"""Render a sequence for each placement into out/seq/<name>.npz (frames + ground truth).

python render_seq.py [--names ...] [--seconds 60] [--screen 0] [--tag ""]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from eyesim import Look, Renderer
from placements import NAMES, make
from sequence import make_sequence

OUT = Path(__file__).resolve().parent.parent / "out" / "seq"


def _setup(name, screen, narrow):
    sc, cam, info = make(name, screen)
    if narrow:
        # closer to the user's phone clips: lower lid at the iris bottom, more upper-lid cover
        sc.lids.up_open, sc.lids.lo_open = 19.0, 10.0
    return sc, cam, info


def chunk(args):
    """Render frames [a, b) of one placement (one worker)."""
    name, seconds, screen, seed, narrow, a, b = args
    sc, cam, _ = _setup(name, screen, narrow)
    r = Renderer(sc, cam)
    r.noise_rng = np.random.default_rng(1000 + seed * 100000 + a)   # independent sensor noise per chunk
    frames = make_sequence(seconds, seed)[a:b]
    look = Look()
    imgs, gts = [], []
    for fr in frames:
        c = cam.moved(*fr.slip) if np.any(fr.slip[0]) or np.any(fr.slip[1]) else cam
        img, gt = r.render(fr.st, c, look)
        gt.update(phase=fr.phase, target=fr.target, settled=fr.settled, openness=fr.st.openness,
                  squint=fr.st.squint, slip=float(np.linalg.norm(fr.slip[0])))
        gt["glints"] = [list(map(float, g)) for g in gt["glints"]]
        imgs.append(img)
        gts.append(gt)
    return name, a, imgs, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=NAMES)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--screen", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--narrow", action="store_true", help="narrower lid opening")
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    n_frames = len(make_sequence(a.seconds, a.seed))
    step = 40
    jobs = [(n, a.seconds, a.screen, a.seed, a.narrow, k, min(k + step, n_frames))
            for n in a.names for k in range(0, n_frames, step)]
    parts = {n: {} for n in a.names}
    with Pool(a.workers) as p:
        for name, k, imgs, gts in p.imap_unordered(chunk, jobs):
            parts[name][k] = (imgs, gts)
            if len(parts[name]) * step >= n_frames:
                ks = sorted(parts[name])
                imgs = [im for k2 in ks for im in parts[name][k2][0]]
                gts = [g for k2 in ks for g in parts[name][k2][1]]
                _, cam, info = _setup(name, a.screen, a.narrow)
                OUT.mkdir(parents=True, exist_ok=True)
                out = OUT / f"{name}{a.tag}.npz"
                np.savez_compressed(out, imgs=np.array(imgs), gt=json.dumps(gts), cam_R=cam.R,
                                    cam_pos=cam.pos, f=cam.f, info=json.dumps(info))
                parts[name] = {}
                print(out, flush=True)


if __name__ == "__main__":
    main()
