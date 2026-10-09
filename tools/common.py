"""Shared paths, landmark index sets and dataset helpers for the brow tracker tools."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO.parent / "_datasets"
RAVDESS = DATASETS / "ravdess"
OUT = REPO / "out"
MODEL = REPO / "sim" / "vendor" / "mediapipe" / "face_landmarker.task"

# MediaPipe face mesh indices. "Right"/"left" are the subject's own sides, so the
# subject's right brow is on the image LEFT in a frontal (non-mirrored) video.
# Each list runs from the outer end of the brow to the inner end (towards the nose).
BROW = {
    "R": {"upper": [70, 63, 105, 66, 107], "lower": [46, 53, 52, 65, 55]},
    "L": {"upper": [300, 293, 334, 296, 336], "lower": [276, 283, 282, 295, 285]},
}
EYE_CORNERS = {"R": (33, 133), "L": (263, 362)}  # (outer, inner)
# Points that barely move with expressions; used to stabilise the head frame.
RIGID = [33, 133, 362, 263, 6, 197, 195, 5, 4, 1, 98, 327, 168]

# Blendshape names MediaPipe uses for the brows.
BS_BROW = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"]

RAVDESS_EMOTIONS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
_RAVDESS_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)$")


def parse_ravdess(stem: str) -> dict:
    """Decode a RAVDESS file stem like 02-01-08-02-01-01-01."""
    m = _RAVDESS_RE.match(stem)
    if not m:
        raise ValueError(f"not a RAVDESS stem: {stem}")
    mod, voc, emo, inten, stmt, rep, actor = (int(g) for g in m.groups())
    return {
        "stem": stem, "modality": mod, "vocal": voc, "emotion": emo,
        "emotion_name": RAVDESS_EMOTIONS[emo], "intensity": inten,
        "statement": stmt, "repetition": rep, "actor": actor,
    }


def actor_dir(kind: str, actor: int) -> Path:
    return RAVDESS / kind / f"Actor_{actor:02d}"


@lru_cache(maxsize=1)
def mesh_triangles() -> np.ndarray:
    """Triangles of the MediaPipe face mesh, rebuilt from its tessellation edges.

    Every triangle of the mesh has all three edges in the edge list, so the
    3-cliques of the edge graph are the triangles (separating 3-cycles do not
    occur in this mesh). Iris points (468+) are not part of the tessellation.
    """
    from mediapipe.tasks.python import vision

    edges = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    nbr: dict[int, set[int]] = {}
    for c in edges:
        nbr.setdefault(c.start, set()).add(c.end)
        nbr.setdefault(c.end, set()).add(c.start)
    tris = set()
    for a, na in nbr.items():
        for b in na:
            if b <= a:
                continue
            for c in na & nbr[b]:
                if c > b:
                    tris.add((a, b, c))
    return np.array(sorted(tris), dtype=np.int32)
