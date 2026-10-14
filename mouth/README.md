# mouth_tracker

Mouth / jaw tracking for the Arka'vali mask: a camera inside the beak, on an ESP32(-S3) CAM
node, raw values to the hub. Sibling of `../eye_tracker` and `../brow_tracker` (same venv).
Survey, camera choice and placement: [docs/survey.md](docs/survey.md).

## Quick start (double-click in the repo folder)

| File | What it does |
|---|---|
| `flash_firmware.bat` | Build `firmware/` (JPEG streamer) and flash the AI-Thinker ESP32-CAM |
| `calib_rec.bat [COMx] [--reps N]` | Prompted calibration recording (~3 min per pass, 30 expressions) to `out/calib/cal_*` with per-frame labels |
| `mouth_live.bat [--babble]` | Trained model on the PC: live picture + value bars (L LED) |
| `mouth_board.bat` | The model running on the ESP32-CAM itself (~10 fps, 93 ms per frame): bars from the board |
| `mouth_cam.bat [COMx]` | Live view; R records to `out/cam/rec_*` (frames.mjpeg + frames.csv), S snapshot, L flash LED, Q quit |

The brow firmware for the same board: `../brow_tracker/flash_firmware.bat`.

## Firmware (`firmware/`)

Two modes (command `mode stream|track`): hardware JPEG 320x240 over Serial at 921600 baud
(~15 fps, ~6 KB/frame, grey; 2 Mbaud lost data on the CH340), or on-chip tracking: raw grey
QVGA -> 4x box downscale to 80x60 -> `lib/mouth_net` (float CNN, weights from
`tools/export_c.py`, copied to internal RAM) -> `MOUTH` line with 22 values. Classic ESP32:
93 ms per frame for the 2.2 M MAC net (570 ms for the first 8.3 M MAC one, weights in flash),
~10 fps. Commands: header of `firmware/src/main.cpp`.

## Tools (`tools/`)

| Script | What it does |
|---|---|
| `scene.py` | Head + mask meshes (from the Fusion design) and GPU ray casting (CuPy, numpy fallback) |
| `sweep.py` | Camera placement sweep in the beak for 120/160/66 deg lenses (`out/sweep/`) |
| `preview.py` | IR-like render of the camera view from given placements (`out/preview.png`) |
| `fusion_mcp.py` | Minimal client for Fusion's local MCP server (read-only scripts, mesh export) |
| `mouth_cam.py` | Serial live view / recorder for the streamer |
| `calib_rec.py` | Prompted calibration recorder; own channel set (`CHANNELS`), labels = target x envelope |
| `teacher.py` | Babble face model (data/babble_face.onnx, non-commercial licence) on recordings -> babble.npz |
| `train.py` | Train the CNN on `out/calib` (`--all`, `--width`), export ONNX |
| `export_c.py` | Weights to `firmware/lib/mouth_net/mouth_weights.h` (BatchNorm folded) |
| `parity_test.py` | C++ net (zig host build) vs ONNX on recorded frames |
| `ledsim.py`, `ledsweep2.py`, `ledsim2ch.py` | LED layout sweeps (pixel metric), wall bounce, blinking check |
| `lipshapes.py`, `rt.py` | Mouth shape deformations of the head mesh; CUDA ray-triangle kernel |
| `synth.py` | Synthetic frames + trained-net comparison of lighting variants (`render` / `train`, `--set`) |
| `configs_catalog.py` | Writes `docs/lighting_configs.md/.json` (every tried LED config with results) |
| `gpu.py` | CuPy loader (copy of `../eye_tracker/tools/gpu.py`) |

`data/head.bin`, `data/mask.bin` (not in git): meshes dumped from the open Fusion document with a
read-only script through `fusion_mcp.py` (`ImageToStl.com_vhs+` mesh body, `forehead R` body).

Setup: `../brow_tracker/.venv` (Python 3.9, numpy, opencv, matplotlib, pyserial, cupy-cuda12x).
