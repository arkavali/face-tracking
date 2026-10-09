"""Run the brow tracker on simulated mask-camera footage and compare with MediaPipe.

Per actor:
  1. The first neutral clip calibrates everything, like putting the mask on:
     head-frame anchors, camera poses, LED lighting, per-brow ROI and the tracker's
     brow curve (from the first CALIB_FRAMES rendered neutral frames).
  2. Every selected clip is rendered frame by frame and tracked; the tracker state
     is reset to "neutral" at the start of each clip (RAVDESS clips start relaxed).
  3. Per frame we store the tracker outputs and three references:
       gt_px   MediaPipe's upper brow contour projected into the virtual camera,
               relative to its neutral position (px, positive = up), per strip
       mp_mm   MediaPipe brow height in the head frame relative to neutral (mm)
       bs      MediaPipe brow blendshapes
Results: out/eval/<config>/actor_XX.npz (+ optional overlay frames for the viewer).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from camsim import (Anchors, Look, Rig, brow_gt, glasses_rim, head_track, ir_texture,
                    led_shading, render)
from common import BROW, BS_BROW, OUT, actor_dir, parse_ravdess
from tracker import BrowTracker, TrackerConfig

CALIB_FRAMES = 30
ROI_UP_MM, ROI_DOWN_MM, ROI_SIDE_MM = 14.0, 8.0, 3.0


def load_track(actor: int, stem: str):
    d = np.load(actor_dir("teacher", actor) / f"{stem}.npz")
    return d, head_track(d["lm"], d["mat"], int(d["w"]), int(d["h"]))


def brow_roi(an: Anchors, cam, side: str) -> tuple:
    pts = an.brow_pts[side]
    ext = np.concatenate([pts, pts + [0, ROI_UP_MM, 0], pts - [0, ROI_DOWN_MM, 0],
                          pts + [ROI_SIDE_MM, 0, 0], pts - [ROI_SIDE_MM, 0, 0]])
    uv, _, ok = cam.project(ext)
    uv = uv[ok]
    x0, y0 = np.floor(uv.min(0)).astype(int)
    x1, y1 = np.ceil(uv.max(0)).astype(int)
    return (max(x0, 0), max(y0, 0), min(x1, cam.w), min(y1, cam.h))


def gt_strip_y(head_f, cam, side, strip_x) -> np.ndarray:
    up = brow_gt(head_f, cam, side)["upper"]
    o = np.argsort(up[:, 0])
    xc = np.array([(a + b) / 2 for a, b in strip_x])
    return np.interp(xc, up[o, 0], up[o, 1], left=np.nan, right=np.nan)


def gt_visible(head_f, cam, side, rend) -> float:
    """Share of the upper-contour points not hidden by occluders or the face itself."""
    from common import BROW as _B
    P = head_f[_B[side]["upper"]]
    uv, dist, ok = cam.project(P)
    vis = 0
    for (u, v), d, k in zip(uv, dist, ok):
        iu, iv = int(round(u)), int(round(v))
        if not k or not (0 <= iu < cam.w and 0 <= iv < cam.h):
            continue
        if rend["occ"][iv, iu]:
            continue
        if rend["depth"][iv, iu] < d - 3.0:
            continue
        vis += 1
    return vis / len(P)


def mp_height(head_f, neutral_pts, side) -> tuple:
    idx = BROW[side]["upper"] + BROW[side]["lower"]
    dy = head_f[idx, 1] - neutral_pts[:, 1]
    # contour order is outer -> inner for both lists (5 each)
    inner = dy[[3, 4, 8, 9]].mean()
    outer = dy[[0, 1, 5, 6]].mean()
    return float(dy.mean()), float(inner), float(outer)


def iter_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def run_actor(actor: int, rig: Rig, look: Look, tcfg: TrackerConfig, out_dir: Path,
              emotions=None, glasses=False, viz_every: int = 0, max_clips: int = 0) -> dict:
    neutral = f"02-01-01-01-01-01-{actor:02d}"
    _, ntr = load_track(actor, neutral)
    an = Anchors.from_track(ntr)
    cams = rig.cameras(an)
    brow_cam = {s: ("C" if "C" in cams else s) for s in "RL"}
    shades = {k: led_shading(an.ref_head, an.ref_fwd, c, look.ambient, rig.light_list(an, k, c))
              for k, c in cams.items()}
    quads = {k: rig.occluders(an, k) + (glasses_rim(an) if glasses else []) for k in cams}

    # Calibration on the first neutral frames.
    calib = {s: [] for s in "RL"}
    for fi, frame in enumerate(iter_frames(actor_dir("videos", actor) / f"{neutral}.mp4")):
        if fi >= CALIB_FRAMES:
            break
        if not ntr.valid[fi]:
            continue
        tex = ir_texture(frame, look, ntr.mm_per_px)
        imgs = {k: render(tex, ntr.src_px[fi], ntr.head[fi], ntr.fwd[fi], c, look, quads[k], shades[k])["img"]
                for k, c in cams.items()}
        for s in "RL":
            calib[s].append(imgs[brow_cam[s]])
    trackers, rois, gt0 = {}, {}, {}
    for s in "RL":
        cam = cams[brow_cam[s]]
        rois[s] = brow_roi(an, cam, s)
        tr = BrowTracker(rois[s], tcfg, inner_is_right=(s == "R"))
        if not tr.calibrate(calib[s]):
            raise RuntimeError(f"actor {actor} brow {s}: calibration failed")
        trackers[s] = tr
        gt0[s] = gt_strip_y(an.ref_head, cam, s, tr.strip_x)

    clips = sorted(actor_dir("videos", actor).glob("02-*.mp4"))
    if emotions:
        clips = [c for c in clips if parse_ravdess(c.stem)["emotion"] in emotions]
    if max_clips:
        clips = clips[:max_clips]
    rows = []
    viz_dir = out_dir / "viz" / f"actor_{actor:02d}"
    t0 = time.time()
    for clip in clips:
        meta = parse_ravdess(clip.stem)
        d, tr = load_track(actor, clip.stem)
        names = list(d["names"])
        bs_idx = [names.index(n) if n in names else -1 for n in BS_BROW]
        for s in "RL":
            trackers[s].prev[:] = 0.0
        for fi, frame in enumerate(iter_frames(clip)):
            if fi >= len(tr.valid) or not tr.valid[fi]:
                continue
            tex = ir_texture(frame, look, tr.mm_per_px)
            rend = {k: render(tex, tr.src_px[fi], tr.head[fi], tr.fwd[fi], c, look, quads[k], shades[k])
                    for k, c in cams.items()}
            row = {"emotion": meta["emotion"], "intensity": meta["intensity"], "clip": clip.stem, "frame": fi,
                   "bs": [float(d["bs"][fi, j]) if j >= 0 else np.nan for j in bs_idx]}
            for s in "RL":
                k = brow_cam[s]
                res = trackers[s].update(rend[k]["img"])
                gy = gt_strip_y(tr.head[fi], cams[k], s, trackers[s].strip_x)
                h, hi, ho = mp_height(tr.head[fi], an.brow_pts[s], s)
                row[s] = {
                    "raise": res["raise"], "inner": res["inner"], "outer": res["outer"],
                    "valid": res["valid"], "off": res["off"].tolist(), "ok": res["ok"].tolist(),
                    "gt": (gt0[s] - gy).tolist(),                     # px, positive = up
                    "mp": h, "mp_inner": hi, "mp_outer": ho,
                    "vis": gt_visible(tr.head[fi], cams[k], s, rend[k]),
                }
            rows.append(row)
            if viz_every and fi % viz_every == 0:
                save_viz(viz_dir, clip.stem, fi, rend, trackers, rows[-1], brow_cam, tr.head[fi], cams)
    dt = time.time() - t0
    pack(out_dir / f"actor_{actor:02d}.npz", rows, rois, trackers, cams, brow_cam, an)
    return {"actor": actor, "frames": len(rows), "seconds": round(dt, 1)}


def save_viz(viz_dir, stem, fi, rend, trackers, row, brow_cam, head_f, cams):
    viz_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k in rend:
        img = cv2.cvtColor(rend[k]["img"], cv2.COLOR_GRAY2BGR)
        for s in "RL":
            if brow_cam[s] != k:
                continue
            tr = trackers[s]
            cv2.rectangle(img, (tr.x0, tr.y0), (tr.x1 - 1, tr.y1 - 1), (90, 90, 90), 1)
            gt = brow_gt(head_f, cams[k], s)
            cv2.polylines(img, [np.round(gt["upper"]).astype(np.int32)], False, (0, 200, 0), 1, cv2.LINE_AA)
            off = np.array(row[s]["off"])
            for j, seg in enumerate(tr.curve_points(off)):
                col = (0, 200, 255) if row[s]["ok"][j] else (0, 0, 255)
                cv2.polylines(img, [np.round(seg).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        tiles.append(img)
    cv2.imwrite(str(viz_dir / f"{stem}_{fi:03d}.png"), np.hstack(tiles))


def pack(path: Path, rows, rois, trackers, cams, brow_cam, an):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "emotion": np.array([r["emotion"] for r in rows]),
        "intensity": np.array([r["intensity"] for r in rows]),
        "clip": np.array([r["clip"] for r in rows]),
        "frame": np.array([r["frame"] for r in rows]),
        "bs": np.array([r["bs"] for r in rows], np.float32),
    }
    for s in "RL":
        for key in ("raise", "inner", "outer", "valid", "mp", "mp_inner", "mp_outer", "vis"):
            out[f"{s}_{key}"] = np.array([r[s][key] for r in rows])
        for key in ("off", "ok", "gt"):
            out[f"{s}_{key}"] = np.array([r[s][key] for r in rows])
        cam = cams[brow_cam[s]]
        out[f"{s}_px_per_mm"] = cam.px_per_mm_at(an.brow_center[s])
        out[f"{s}_roi"] = np.array(rois[s])
    np.savez_compressed(path, **out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="name of the output folder under out/eval")
    ap.add_argument("--actors", type=int, nargs="+", default=[1])
    ap.add_argument("--emotions", type=int, nargs="*", default=None)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--mode", default="side")
    ap.add_argument("--up", type=float, default=28.0)
    ap.add_argument("--fwd", type=float, default=24.0)
    ap.add_argument("--flange-top", type=float, default=29.0)
    ap.add_argument("--res", default="320x240")
    ap.add_argument("--dfov", type=float, default=120.0)
    ap.add_argument("--lights", default="cam", help="comma list: cam,below")
    ap.add_argument("--ir", type=float, default=0.7)
    ap.add_argument("--noise", type=float, default=2.0)
    ap.add_argument("--glasses", action="store_true")
    ap.add_argument("--method", default="edge")
    ap.add_argument("--strips", type=int, default=5)
    ap.add_argument("--viz-every", type=int, default=0)
    args = ap.parse_args()
    w, h = (int(v) for v in args.res.split("x"))
    rig = Rig(mode=args.mode, cam_up=args.up, cam_fwd=args.fwd, flange_top=args.flange_top,
              width=w, height=h, dfov_deg=args.dfov, lights=tuple(args.lights.split(",")))
    look = Look(ir_contrast=args.ir, noise_read=args.noise)
    tcfg = TrackerConfig(method=args.method, strips=args.strips)
    out_dir = OUT / "eval" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "rig": asdict(rig), "look": {k: v for k, v in asdict(replace(look, rng=None)).items() if k != "rng"},
        "tracker": asdict(tcfg), "glasses": args.glasses, "emotions": args.emotions}, indent=1, default=str))
    for actor in args.actors:
        info = run_actor(actor, rig, look, tcfg, out_dir, args.emotions, args.glasses,
                         args.viz_every, args.max_clips)
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
