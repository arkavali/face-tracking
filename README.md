# mouth_tracker

Mouth / jaw tracking for the Arka'vali mask: a camera inside the beak, on an ESP32(-S3) CAM
node, raw values to the hub. Sibling of `../eye_tracker` and `../brow_tracker` (same venv).
Survey, camera choice and placement: [docs/survey.md](docs/survey.md).

## Quick start (double-click in the repo folder)

| File | What it does |
|---|---|
| `flash_firmware.bat` | Build `firmware/` (JPEG streamer) and flash the AI-Thinker ESP32-CAM |
| `calib_rec.bat [COMx] [--reps N]` | Prompted calibration recording (~3 min per pass, 30 expressions) to `out/calib/cal_*` with per-frame labels |
| `mouth_cam.bat [COMx]` | Live view; R records to `out/cam/rec_*` (frames.mjpeg + frames.csv), S snapshot, L flash LED, Q quit |

The brow firmware for the same board: `../brow_tracker/flash_firmware.bat`.

## Firmware (`firmware/`)

Streamer only (no tracking yet): hardware JPEG 320x240 over Serial at 921600 baud,
~14 fps at ~6 KB/frame, no lost frames (2 Mbaud lost data on the CH340). Commands: header of
`firmware/src/main.cpp`.

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
| `gpu.py` | CuPy loader (copy of `../eye_tracker/tools/gpu.py`) |

`data/head.bin`, `data/mask.bin` (not in git): meshes dumped from the open Fusion document with a
read-only script through `fusion_mcp.py` (`ImageToStl.com_vhs+` mesh body, `forehead R` body).

Setup: `../brow_tracker/.venv` (Python 3.9, numpy, opencv, matplotlib, pyserial, cupy-cuda12x).
