"""evaluate.py for a single non-RAVDESS frontal video (e.g. the user's own phone clip).

The first --calib frames are the neutral face (mask put on). Frames inside that range
are tagged emotion=1 ("neutral") so metrics.py can measure jitter at rest.
Teacher labels must exist: run teacher.py --videos <file> first.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from camsim import (Anchors, HeadTrack, Look, Rig, glasses_rim, head_track, ir_texture,
                    led_shading, render)
from common import BS_BROW, OUT
from evaluate import brow_roi, gt_strip_y, gt_visible, iter_frames, mp_height, pack, save_viz
from tracker import BrowTracker, TrackerConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", required=True)
    ap.add_argument("--calib", type=int, nargs=2, default=[0, 30], metavar=("START", "END"))
    ap.add_argument("--mode", default="side")
    ap.add_argument("--cam-y", type=float, default=57.0)
    ap.add_argument("--cam-z", type=float, default=39.0)
    ap.add_argument("--flange-top", type=float, default=29.0)
    ap.add_argument("--lights", default="cam")
    ap.add_argument("--ir", type=float, default=0.7)
    ap.add_argument("--method", default="ncc")
    ap.add_argument("--viz-every", type=int, default=0)
    args = ap.parse_args()

    tpath = args.video.parent.parent / "teacher" / f"{args.video.stem}.npz"
    d = np.load(tpath)
    tr = head_track(d["lm"], d["mat"], int(d["w"]), int(d["h"]))
    a, b = args.calib
    ntr = HeadTrack(tr.head[a:b], tr.src_px[a:b], tr.fwd[a:b], tr.valid[a:b], tr.mm_per_px)
    an = Anchors.from_track(ntr)
    rig = Rig(mode=args.mode, cam_y=args.cam_y, cam_z=args.cam_z, flange_top=args.flange_top,
              lights=tuple(args.lights.split(",")))
    look = Look(ir_contrast=args.ir)
    tcfg = TrackerConfig(method=args.method)
    cams = rig.cameras(an)
    brow_cam = {s: ("C" if "C" in cams else s) for s in "RL"}
    shades = {k: led_shading(an.ref_head, an.ref_fwd, c, look.ambient, rig.light_list(an, k, c))
              for k, c in cams.items()}
    quads = {k: rig.occluders(an, k) for k in cams}

    frames = list(iter_frames(args.video))
    rendered = []
    for fi, frame in enumerate(frames):
        if fi >= len(tr.valid) or not tr.valid[fi]:
            rendered.append(None)
            continue
        tex = ir_texture(frame, look, tr.mm_per_px)
        rendered.append({k: render(tex, tr.src_px[fi], tr.head[fi], tr.fwd[fi], c, look, quads[k], shades[k])
                         for k, c in cams.items()})
    trackers, rois, gt0 = {}, {}, {}
    for s in "RL":
        cam = cams[brow_cam[s]]
        rois[s] = brow_roi(an, cam, s)
        t = BrowTracker(rois[s], tcfg, inner_is_right=(s == "R"))
        calib = [rendered[i][brow_cam[s]]["img"] for i in range(a, b) if rendered[i] is not None]
        if not t.calibrate(calib):
            raise SystemExit(f"calibration failed for brow {s}")
        trackers[s] = t
        gt0[s] = gt_strip_y(an.ref_head, cam, s, t.strip_x)

    names = list(d["names"])
    bs_idx = [names.index(n) if n in names else -1 for n in BS_BROW]
    out_dir = OUT / "eval" / args.config
    rows = []
    for fi, rend in enumerate(rendered):
        if rend is None:
            continue
        row = {"emotion": 1 if a <= fi < b else 0, "intensity": 0, "clip": args.video.stem, "frame": fi,
               "bs": [float(d["bs"][fi, j]) if j >= 0 else np.nan for j in bs_idx]}
        for s in "RL":
            k = brow_cam[s]
            res = trackers[s].update(rend[k]["img"])
            gy = gt_strip_y(tr.head[fi], cams[k], s, trackers[s].strip_x)
            h, hi, ho = mp_height(tr.head[fi], an.brow_pts[s], s)
            row[s] = {"raise": res["raise"], "inner": res["inner"], "outer": res["outer"],
                      "valid": res["valid"], "off": res["off"].tolist(), "ok": res["ok"].tolist(),
                      "gt": (gt0[s] - gy).tolist(), "mp": h, "mp_inner": hi, "mp_outer": ho,
                      "vis": gt_visible(tr.head[fi], cams[k], s, rend[k])}
        rows.append(row)
        if args.viz_every and fi % args.viz_every == 0:
            save_viz(out_dir / "viz" / args.video.stem, args.video.stem, fi, rend, trackers, row, brow_cam,
                     tr.head[fi], cams)
    pack(out_dir / "actor_00.npz", rows, rois, trackers, cams, brow_cam, an)
    (out_dir / "config.json").write_text(json.dumps({"rig": asdict(rig), "look": {"ir_contrast": look.ir_contrast,
                                                     "noise_read": look.noise_read}, "tracker": asdict(tcfg),
                                                     "video": str(args.video), "calib": args.calib}, indent=1,
                                                    default=str))
    print(out_dir)


if __name__ == "__main__":
    main()
