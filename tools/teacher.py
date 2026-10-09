"""Run MediaPipe FaceLandmarker over RAVDESS clips and store per-frame teacher data.

For every clip ../_datasets/ravdess/videos/Actor_XX/<stem>.mp4 this writes
../_datasets/ravdess/teacher/Actor_XX/<stem>.npz with:
  lm   (N, 478, 3) float32  normalised landmarks (x, y in [0,1], z ~ x scale), NaN if no face
  bs   (N, 52)     float32  blendshape scores, NaN if no face
  mat  (N, 4, 4)   float32  facial transformation matrix, NaN if no face
  names (52,)               blendshape names
  fps, w, h                 video properties
The model is the same face_landmarker.task the Android/web prototype uses.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from common import MODEL, actor_dir, parse_ravdess


def make_landmarker() -> vision.FaceLandmarker:
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def label_clip(path, out_path) -> int:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lms, bss, mats, names = [], [], [], None
    # A fresh landmarker per clip: VIDEO mode needs monotonic timestamps and
    # must not carry tracking state over from the previous clip.
    with make_landmarker() as lmk:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                       int(i * 1000.0 / fps))
            if res.face_landmarks:
                lms.append([(p.x, p.y, p.z) for p in res.face_landmarks[0]])
                cats = res.face_blendshapes[0]
                if names is None:
                    names = [c.category_name for c in cats]
                bss.append([c.score for c in cats])
                mats.append(np.asarray(res.facial_transformation_matrixes[0], dtype=np.float32))
            else:
                lms.append(np.full((478, 3), np.nan))
                bss.append(np.full(52, np.nan))
                mats.append(np.full((4, 4), np.nan))
            i += 1
    cap.release()
    if names is None:
        names = [""] * 52
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, lm=np.asarray(lms, np.float32), bs=np.asarray(bss, np.float32),
                        mat=np.asarray(mats, np.float32), names=np.array(names),
                        fps=fps, w=w, h=h)
    return i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actors", type=int, nargs="*", default=[])
    ap.add_argument("--videos", type=Path, nargs="*", default=[],
                    help="arbitrary videos; labels go to <video dir>/../teacher/<stem>.npz")
    ap.add_argument("--emotions", type=int, nargs="*", default=None,
                    help="RAVDESS emotion codes to keep (default: all)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for v in args.videos:
        out = v.parent.parent / "teacher" / f"{v.stem}.npz"
        if out.exists() and not args.force:
            continue
        n = label_clip(v, out)
        d = np.load(out)
        found = int((~np.isnan(d["lm"][:, 0, 0])).sum())
        print(f"{v.name}: {n} frames, face found in {found}", flush=True)
    for actor in args.actors:
        clips = sorted(actor_dir("videos", actor).glob("02-*.mp4"))
        t0, frames = time.time(), 0
        for clip in clips:
            meta = parse_ravdess(clip.stem)
            if args.emotions and meta["emotion"] not in args.emotions:
                continue
            out = actor_dir("teacher", actor) / f"{clip.stem}.npz"
            if out.exists() and not args.force:
                continue
            frames += label_clip(clip, out)
        dt = time.time() - t0
        print(f"actor {actor:02d}: {len(clips)} clips, {frames} new frames in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
