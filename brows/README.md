# brow_tracker

Feasibility study for tracking the wearer's eyebrows with a small camera inside the
Arka'vali mask, using plain image maths that can run on an ESP32 (no neural network,
no phone). Kept separate from `face_mesh_demo` (the working full-face MediaPipe
prototype) on purpose.

Why: inside the mask the AR periscope mirrors and lenses block most of the face, so
full-face trackers (MediaPipe) lose the face. MediaPipe also stops detecting a face
once the camera looks down at it steeper than ~45 deg, which is exactly the view a
camera mounted above the mirror holder gets. Brows are the main control input, so
they get their own small camera and tracker.

Results and recommendations: [docs/feasibility.md](docs/feasibility.md). Regenerate the
full HTML report with `tools/report.py` (-> `out/report.html`).

## Quick start (double-click in the repo folder)

| File | What it does |
|---|---|
| `brow_ui.bat` | Brow camera UI in the browser: live view, draw the brow ROIs with the mouse, calibrate, range up/down, live plots, snapshots and recordings (`out/cam`) |
| `flash_firmware.bat` | Build `firmware/` and flash the ESP32-CAM (lets the UI release the port while flashing) |
| `serial_monitor.bat` | Plain text console of the board (close the UI first) |
| `track_video.bat` | Drag a video, an image folder or a UI recording onto it: offline tracking with ROIs drawn in a window, results open when done (`out/video`) |
| `report.bat` | Rebuild and open `out/report.html` |
| `sim_preview.bat` | Simulated mask-camera view for one placement (arguments go to `tools/preview_sim.py`) |

Setup on the board, once per wearer / mounting: in the UI draw a rectangle around each
brow (from below the hairline to below the brow, room above it for a full raise), hold a
neutral face and press Calibrate, then hold the brows fully raised -> "Range up", fully
lowered -> "Range down". ROIs and calibration are kept in flash (NVS).

## Firmware (`firmware/`, AI-Thinker ESP32-CAM, OV2640)

Measured on the board (2026-10-09): one brow update 3-5 ms, both brows ~7.5 ms per frame;
camera 25 fps grey 320x240. esp32-camera runs the OV2640 at a quarter clock in non-JPEG
modes (12.5 fps); the firmware sets the sensor clock divider to 1 (`CLKRC` 0x111 = 1) -
divider 0 is too fast for the I2S DMA (every frame arrives short). The UI preview costs
~2-3 fps of tracking while it is on. Commands: see the header of `firmware/src/main.cpp`.

Hardware notes: the board is touch-sensitive (touching it can drop the camera); the
firmware restarts the camera after 2 s without frames (`camera restarts` in `status`).
The loop task needs a 16 KB stack (the tracker's re-acquire path overflowed 8 KB).

## How the tracker works (`tools/tracker.py`)

Seen from above, only the upper half of a brow faces the camera: the reliable feature
is the step from bright forehead to dark brow hair.

1. Calibration (neutral face, a few frames, one ROI per brow drawn once by the user):
   in every ROI column take the topmost strong bright->dark step that has a
   brow-thick dark band below it (glasses rims and eyelid creases are stronger but
   sit below the brow; thin forehead wrinkles fail the band test), then fit a
   quadratic curve through those points.
2. Tracking: the brow is cut into 5 strips along the curve. Each strip gives a 1-D
   profile sampled along the curve; its vertical shift against the neutral profile
   is found by normalised cross-correlation within +-2.5 mm of the previous frame.
   Strips that disagree with the others are pulled back to the median; after 15
   frames without a valid result the brow is searched again over the whole range.
3. Output per brow: raise/lower (mean of strips), inner and outer end.

Everything is 1-D sums and correlations over a few thousand pixels per brow per
frame - cheap enough for an ESP32 at camera frame rate.

The ROI must start below the hairline (hair is a thick dark band too).

## Tools

| Script | What it does |
|---|---|
| `teacher.py` | MediaPipe FaceLandmarker labels for RAVDESS clips or any video (`--videos`) |
| `camsim.py` | Mask-camera simulator: re-renders a frontal frame as seen by a fisheye camera fixed to the head above the AR mirror holder (mesh warp from MediaPipe landmarks, IR-like texture, LED lighting on the neutral face shape, holder and glasses occluders) |
| `evaluate.py` / `sweep.py` | Run the tracker on simulated footage of RAVDESS actors, compare with MediaPipe; `sweep.py` runs many configs in parallel (one thread per job) |
| `eval_own.py` | Same for one arbitrary frontal video |
| `stab_eval.py` | Tracker vs MediaPipe on REAL frontal footage aligned to a neutral frame (no 3D re-rendering) |
| `track_video.py` | Tracker on real top-down footage (e.g. phone above the brows) with hand-placed ROIs |
| `metrics.py`, `plots.py`, `report.py` | Summaries, time-series plots, `out/report.html` |
| `preview_sim.py` | Contact sheet of simulated frames for a camera placement |
| `ui/browui.py` (+ `index.html`, `app.js`, `style.css`) | Brow camera UI: web page + serial bridge (`brow_ui.bat`) |
| `brow_cam.py` | Minimal command-line serial helper (snap / setup / monitor / cmd) |
| `parity_test.py` | C++ core (`firmware/lib/brow_core`) vs Python reference on a video |
| `blender_inspect_mask.py`, `step_parts.py` | Mask model inspection (STL parts, STEP solids) |

Setup: Python 3.9 venv in `.venv` with `mediapipe opencv-python numpy matplotlib gmsh`.
Datasets live outside the repo in `../_datasets` (RAVDESS speech videos, CC BY-NC-SA 4.0,
https://zenodo.org/records/1188976; the user's own clips in `../_datasets/own`).

```
.venv/Scripts/python tools/teacher.py --actors 1 2 3
.venv/Scripts/python tools/sweep.py tools/jobs_v3.txt -j 28
.venv/Scripts/python tools/metrics.py h_red_v3
.venv/Scripts/python tools/report.py
```

## Simulator limits (read before trusting numbers)

- The texture is the frontal video re-projected onto MediaPipe's mesh: resolution is
  limited to the source (~2 px/mm for RAVDESS), surfaces the frontal camera does not
  see are missing, and the brow ridge depth comes from MediaPipe's z estimate.
- "IR look" is a guess (red channel, reduced local contrast); real 850/940 nm images
  of the user's brows have not been taken yet.
- No cast shadows, no specular highlights, no sweat, no mask slip beyond MediaPipe's
  own head-pose jitter.
- The reference (MediaPipe) is itself noisy, especially its absolute brow height.
