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


def job(args):
    name, seconds, screen, tag, seed = args
    sc, cam, info = make(name, screen)
    r = Renderer(sc, cam)
    frames = make_sequence(seconds, seed)
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
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}{tag}.npz"
    np.savez_compressed(out, imgs=np.array(imgs), gt=json.dumps(gts), cam_R=cam.R, cam_pos=cam.pos,
                        f=cam.f, info=json.dumps(info))
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=NAMES)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--screen", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with Pool(len(a.names)) as p:
        for r in p.imap_unordered(job, [(n, a.seconds, a.screen, a.tag, a.seed) for n in a.names]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
