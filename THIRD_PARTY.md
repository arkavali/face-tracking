# Third-party code, models and data

## Bundled in this repo

| Path | What | Licence |
|---|---|---|
| `brows/sim/vendor/three/` | three.js build and example loaders | MIT (`LICENSE` in the folder) |
| `brows/sim/models/facecap.glb` | Face model from the three.js examples | see the three.js examples credits |

## Used but not bundled

| What | Used by | Licence | Where to get it |
|---|---|---|---|
| MediaPipe Face Landmarker (`face_landmarker.task`, pip package `mediapipe`) | brows (teacher labels) | Apache-2.0 | `python brows/tools/fetch_model.py` (official Google URL) |
| RAVDESS speech videos | brows (teacher labels, simulator) | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| Project Babble face model (`babble_face.onnx`) | mouth (teacher for training) | Babble Software Distribution License 1.0 (Apache-2.0 + non-commercial clause) | https://github.com/Project-Babble |
| JEOresearch 3DTracker | eyes (method reference, not copied) | see the project | https://github.com/JEOresearch |

Mouth `data/` (head and mask meshes from the suit CAD) is not published.
