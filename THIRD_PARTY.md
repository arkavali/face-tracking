# Third-party code, models and data

Nothing third-party is bundled in this repo: everything below is installed or downloaded
separately. No third-party source was copied into the trackers; methods taken from papers or
other projects are re-implemented and credited here.

## Models and data (downloaded separately)

| What | Used by | Licence | Where to get it |
|---|---|---|---|
| MediaPipe Face Landmarker (`face_landmarker.task`) | brows: teacher labels for the simulator studies | Apache-2.0 | `python brows/tools/fetch_model.py` (official Google URL) |
| RAVDESS speech videos | brows: simulator footage and teacher labels | CC BY-NC-SA 4.0 | https://zenodo.org/records/1188976 |
| Project Babble face model (Baballonia `faceModel.onnx`, saved as `mouth/data/babble_face.onnx`) | mouth: comparison bars in the live view (`mouth_live.bat --babble`) and `teacher.py`; **not used to train the mouth net** | Babble Software Distribution License 1.0 (Apache-2.0 + non-commercial copyleft) | https://github.com/Project-Babble/Baballonia |

RAVDESS: Livingstone SR, Russo FA (2018). The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391
Only numbers measured on it (tables in `brows/docs/`) are in this repo, no frames.

The Babble model is a personal-use reference only: do not run it as part of anything paid
(licence section 10).

## Methods (re-implemented, no code copied)

| Method | Used in | Source |
|---|---|---|
| Pupil seed, ellipse, eyeball centre from many pupil ellipses | eyes | Orlosky, JEOresearch EyeTracker `3DTracker` (MIT), https://github.com/JEOresearch/EyeTracker |
| Starburst pupil rays | eyes | Li, Winfield, Parkhurst (2005), "Starburst" |
| Direct least-squares ellipse fit | eyes | Fitzgibbon, Pilu, Fisher (1999) |
| Camera / LED recipe for the lower face, Unified Expressions channel names | mouth | Project Babble docs, VRCFaceTracking |

## Python packages (`requirements.txt`)

numpy, pyserial, PyTorch (BSD-3-Clause); matplotlib (PSF-based); OpenCV, MediaPipe (Apache-2.0);
onnx, onnxruntime, CuPy, ziglang (MIT / Apache-2.0).

## Firmware (fetched by PlatformIO)

pioarduino platform-espressif32 with arduino-esp32 (LGPL-2.1) and esp32-camera (Apache-2.0).
Only source is published here; anyone distributing built firmware binaries takes on the
LGPL obligations of arduino-esp32.

Mouth `data/` (head and mask meshes from the suit CAD) is not published.
