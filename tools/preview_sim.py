"""Render a contact sheet of simulated mask-camera frames for one clip (visual check)."""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from camsim import Anchors, Look, Rig, brow_gt, glasses_rim, head_track, ir_texture, led_shading, render
from common import OUT, actor_dir


def load(actor: int, stem: str):
    d = np.load(actor_dir("teacher", actor) / f"{stem}.npz")
    tr = head_track(d["lm"], d["mat"], int(d["w"]), int(d["h"]))
    return d, tr


def read_frames(actor: int, stem: str, idx):
    cap = cv2.VideoCapture(str(actor_dir("videos", actor) / f"{stem}.mp4"))
    frames, i, want = {}, 0, set(idx)
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = f
            want.discard(i)
        i += 1
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actor", type=int, default=1)
    ap.add_argument("--neutral", default="02-01-01-01-01-01-{a:02d}")
    ap.add_argument("--clip", default="02-01-08-02-01-01-{a:02d}")
    ap.add_argument("--mode", default="side")
    ap.add_argument("--cam-y", type=float, default=57, help="camera height above the eye line, mm")
    ap.add_argument("--cam-z", type=float, default=39, help="camera distance in front of the brow skin, mm")
    ap.add_argument("--no-holder", action="store_true")
    ap.add_argument("--flange-top", type=float, default=29.0, help="holder top above the eye line, mm")
    ap.add_argument("--ir", type=float, default=0.7, help="IR local-contrast factor (1 = visible light)")
    ap.add_argument("--glasses", action="store_true")
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 30, 60, 90])
    ap.add_argument("--open", action="store_true", help="open the contact sheet when done")
    args = ap.parse_args()
    neutral = args.neutral.format(a=args.actor)
    clip = args.clip.format(a=args.actor)

    _, ntr = load(args.actor, neutral)
    an = Anchors.from_track(ntr)
    d, tr = load(args.actor, clip)
    rig = Rig(mode=args.mode, cam_y=args.cam_y, cam_z=args.cam_z, flange_top=args.flange_top)
    cams = rig.cameras(an)
    look = Look(ir_contrast=args.ir)
    shades = {k: led_shading(an.ref_head, an.ref_fwd, c, look.ambient) for k, c in cams.items()}
    frames = read_frames(args.actor, clip, args.frames)
    rows = []
    for fi in args.frames:
        if fi not in frames:
            continue
        tex = ir_texture(frames[fi], look, tr.mm_per_px)
        tiles = []
        for key, cam in cams.items():
            quads = [] if args.no_holder else rig.occluders(an, key)
            if args.glasses:
                quads += glasses_rim(an)
            r = render(tex, tr.src_px[fi], tr.head[fi], tr.fwd[fi], cam, look, quads, shades[key])
            tile = cv2.cvtColor(r["img"], cv2.COLOR_GRAY2BGR)
            for s in ("RL" if key == "C" else key):
                gt = brow_gt(tr.head[fi], cam, s)
                for name, col in (("upper", (0, 200, 0)), ("lower", (0, 140, 255))):
                    pts = np.round(gt[name]).astype(np.int32)
                    cv2.polylines(tile, [pts], False, col, 1, cv2.LINE_AA)
            cv2.putText(tile, f"{clip} f{fi} cam {key}", (4, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
            tiles.append(tile)
        small = cv2.resize(frames[fi], (int(240 * frames[fi].shape[1] / frames[fi].shape[0]), 240))
        rows.append(np.hstack([small] + tiles))
    sheet = np.vstack(rows)
    OUT.mkdir(exist_ok=True)
    dst = OUT / f"preview_{clip}_{args.mode}_y{args.cam_y:.0f}_z{args.cam_z:.0f}_t{args.flange_top:.0f}{'_nh' if args.no_holder else ''}{'_g' if args.glasses else ''}.png"
    cv2.imwrite(str(dst), sheet)
    print(dst)
    if args.open:
        import os

        os.startfile(dst)
    for s in "RL":
        cam = cams.get(s) or cams["C"]
        print(s, "px/mm at brow:", round(cam.px_per_mm_at(an.brow_center[s]), 2),
              "brow centre (mm):", np.round(an.brow_center[s], 1), "eye centre:", np.round(an.eye_center[s], 1))


if __name__ == "__main__":
    main()
