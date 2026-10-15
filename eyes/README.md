# eyes

Gaze, blink and squint tracking for the Arka'vali mask: one ESP32-S3 CAM (OV2640, 66 deg,
no IR filter) per eye, plain image maths on the chip, results sent to the hub. Sibling of
`../brows` (same layout, same venv).

Method base: JEOresearch 3DTracker (darkest-area seed, pupil ellipse, eyeball centre from
the minor axes of many pupil ellipses, gaze = centre -> pupil), reworked for the ESP:
starburst rays + Fitzgibbon ellipse fit with RANSAC instead of OpenCV contours, and the
eye-sphere model re-estimated continuously (slow forgetting) so it follows mask slip.

The boards are not here yet (ESP32-S3 CAM N16R8 + OV2640 66 deg no-IR, ~2026-10-20): the
tracker was developed on a synthetic eye, the firmware builds but has not run on hardware.

## Quick start (double-click in the repo folder, once the board is here)

| File | What it does |
|---|---|
| `flash_firmware.bat` | Build `firmware/` and flash the ESP32-S3 CAM over its "USB" port |
| `eye_live.bat COMx` | Live view with the tracker overlay (pupil ellipse, eyeball centre, glints); R records to `out/cam` |
| `serial_monitor.bat` | Plain text console of the board |

## Firmware (`firmware/`, ESP32-S3 CAM, OV2640)

`lib/eye_core` is the C++ port of `pupil.py` + `eyemodel.py` (detector, gate, glints,
eye-sphere model). `tools/parity_test.py` builds it for the PC with zig and compares it
with Python: double precision matches to ~1e-11, the float build (as on the ESP) to
~0.2 px with an occasional different RANSAC pick; 0.2 ms/frame on the PC. Commands and
the EYE output line: header of `firmware/src/main.cpp`. The camera pin map is the
Freenove / ESP32-S3-EYE layout and must be checked against the boards. Gaze calibration
(AR target points, glint mapping, blink ranges) is meant to run on the hub from the raw
values.

Measured on real boards (2026-10-10):
- ESP32-S3 N16R8 (hub test board, camera-less env `s3bench`, frames from the PC via
  `tools/hw_bench.py`): detect + gate + model + glints **18.5 ms per 320x240 frame**
  (seed 4.6, blur 3.1, starburst 3.0, RANSAC 5.6 ms); output identical to the Python
  reference. The first port took 163 ms (float full-frame blur in PSRAM, double-precision
  fit); now the seed runs on the raw frame, the blur is integer and only in a window
  around the seed (pixels outside it are computed on demand, same values), the fit is float.
- The brow tracker's AI-Thinker ESP32-CAM (env `esp32cam`, its real OV2640): camera at
  25 fps with tracking, 18 ms per frame when no pupil is in view (seed 9 ms: the classic
  ESP32 reads PSRAM slower). Back to the brow firmware:
  `../brows/flash_firmware.bat`.

The pupil seed box is a per-placement setting (`box` command): 16 px behind the AR
mirror (noisy frames), 8 px from below.

## Tools (`tools/`)

| Script | What it does |
|---|---|
| `eyesim.py` | Ray-traced near-IR eye: refracting cornea, iris/pupil, lids that follow the gaze, blink/squint, lashes, LED glints, AR screen reflection, semi-transparent combiner plate, glasses (lens power, surface reflections, rim), sensor noise |
| `sweep.py` | Camera placement sweep (distance x height x side): render, evaluate, heat maps (`out/sweep/`) |
| `led_glasses.py` | Where LED reflections in a spectacle lens land, per LED position |
| `placements.py` | Camera/LED placements for one eye (under the AR mirror at several distances, through the combiner) |
| `preview.py` | Contact sheet: placements x eye states (`out/preview.png`; examples in `docs/preview_glasses.png`, `docs/preview_narrow.png`) |
| `sequence.py` | 25 fps sequences: 9-point calibration, fixations/saccades/pursuit, blinks, squints, pupil size drift, mask slip |
| `render_seq.py` | Render a sequence per placement to `out/seq/*.npz` (parallel) |
| `pupil.py` | Pupil detector (seed, starburst, RANSAC ellipse) written for an ESP port |
| `eyemodel.py` | Incremental eye-sphere model, gaze vector, polynomial calibration |
| `evaluate.py` | Detector + model + calibrations (pupil, pupil-glint, pupil-centre, model) on rendered sequences vs ground truth (`out/eval/`) |
| `ir_meter.py` | Relative IR irradiance of an LED vs a reference light, using a camera with locked exposure |
| `own_video.py` | Detector on the user's phone clips (visible light, geometry check only) |
| `dbg_detect.py` | Debug sheet: edge points and ellipses on the preview states |
| `lids.py` | Patch-regression eyelid estimator (did not generalise on the simulated lids; kept for real footage) |
| `parity_test.py` | C++ core vs Python on rendered sequences or videos |
| `hw_bench.py`, `hw_profile.py` | Run the core on an S3 flashed with env `s3bench` (timing, parity, per-stage profile) |
| `eye_cam.py` | Serial helper: snap, live view, recording (`eye_live.bat`) |

Placement comparison: [docs/placement.md](docs/placement.md).

Setup: uses `../.venv` (Python 3.9, numpy, opencv-python, matplotlib). GPU,
ready but not used by the renderer yet: `cupy-cuda12x` + `nvidia-cuda-runtime-cu12`,
`nvidia-curand-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cublas-cu12` in the same venv;
`from gpu import cp` (tools/gpu.py) loads CuPy on the RTX 5070 (None without a GPU).

```
PY=../.venv/Scripts/python
cd tools
$PY preview.py
$PY render_seq.py            # ~10 min, all placements
$PY evaluate.py
```

## Simulator limits

- Geometry is textbook (single-surface cornea, sphere lids). Real lids are thicker and
  the upper lid fold hangs over the margin, so lid occlusion from below is probably
  underestimated.
- Reflectances at 850 nm are guesses; no cast shadows except a crude lid shadow, no tear
  film, no motion blur (saccades span 1-2 frames).
- Glasses are a thin lens; lashes are drawn without its refraction.
- The AR combiner pose is estimated from brow_tracker's holder numbers, not from the CAD.
- Noise is relative: the through-mirror camera gets 1/T less light at the same LED power.
