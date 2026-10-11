# Mouth tracking for the Arka'vali mask: survey and plan

Goal: jaw / lip expression of the wearer from a small camera inside the muzzle, computed on an
ESP32-S3 CAM node (like `../eye_tracker` and `../brow_tracker`), raw values sent to the hub,
which drives the jaw motors and LED reactions.

## Existing open trackers (checked 2026-10-11)

| Project | What it is | Useful for us |
|---|---|---|
| **Project Babble** (`Project-Babble/ProjectBabble`, `Baballonia`, `BabbleTrainer`, `BabbleHardware`, `BabbleDocs`) | The only maintained open lower-face tracker for VR. Hardware: XIAO ESP32-S3 Sense or OpenIris ESP32 boards streaming an IR OV2640 to the PC. PC app runs `faceModel.onnx` | Camera/lens/LED recipe, output naming, a ready **teacher model** for exactly our view (close IR lower face) |
| VRCFaceTracking | OSC bridge + module host (Babble, Vive, Quest Pro...) | Only the Unified Expressions naming; no tracking itself |
| MediaPipe Face Landmarker + blendshapes | 478 landmarks + 52 ARKit blendshapes, needs a whole face | Teacher on frontal phone video (as in brow_tracker); useless on a close mouth crop |
| ESP-DL / esp-tflite-micro examples (face detect, MTCNN landmarks) | Whole-face detectors on the S3 | Runtime + int8 kernels only; no mouth/blendshape model exists for MCUs |
| Vive Facial Tracker, Quest Pro | Closed | Nothing |

### Babble details (from the cloned repos)

- Face model: EfficientNet-B0 (`EFFB0E11BS128V7.5`, 23.7 MB fp32, ~5.9 M params, ~0.4 GFLOP),
  224x224 grayscale in, Unified Expressions out (jawOpen, mouthSmile L/R, mouthFunnel,
  mouthPucker, mouthLeft/Right, mouthUpperUp, lowerDown, cheekPuff, tongueOut, ...).
  **Too heavy for the S3** (estimate: seconds per frame even int8), but fine on the PC.
- `BabbleTrainer` (newer, eye side): a tiny 6-conv CNN ("MicroChad", ~8 M MAC at 64x64)
  trained **per user** from a calibration routine on top of a baseline. That size runs in
  ~10-20 ms on the S3 with ESP-DL/ESP-NN int8 (estimate, to be measured).
- Camera recipe: OV2640, 120-160 deg lens, IR filter removed or "night" 850 nm module,
  2x 850 nm LEDs, about 10 cm from the lower face, centred, looking at the mouth.
- License: "Babble Software Distribution License 1.0" = Apache-2.0 text + **non-commercial
  copyleft** clause. OK for the personal suit; anything built from their code or model
  weights stays non-commercial and carries the license. Re-implementing the idea (small CNN,
  own training code, own data) is not affected.

## Camera

The ordered OV2640 66 deg modules do not fit the mouth: at the 4-6 cm that a muzzle allows,
66 deg covers 5-8 cm, i.e. the lips but not the corners in a wide smile or the chin/jaw.

| Distance lips-camera | Needed width ~9 cm (corners + chin margin) | Lens |
|---|---|---|
| 4 cm | 97 deg | 120 deg |
| 6 cm | 74 deg | 100-120 deg |
| 10 cm (Babble) | 48 deg | 66 deg would work |

Recommendation: **OV2640 120 deg, 850 nm "night" version (no IR-cut)**, short flex (the S3 CAM
boards take a 24-pin FPC; check flex length vs where the board sits), plus 2x 850 nm LEDs with
diffusion. 160 deg only if the camera must sit closer than ~4 cm (more distortion, fewer pixels
on the lips). The old 120 deg module from the first ESP32-CAM can be used for the first tests
after removing its IR filter, or with visible light.

Open questions for the mask CAD: free distance from lips to the muzzle wall, whether the camera
can be centred below the nose fan, where the jaw mechanism and the ESP32s block the view, and
whether the jaw of the mask moves in front of the camera.

## Plan (build on Babble, not from scratch)

1. **Bring-up as a Babble tracker.** ESP32-S3 CAM + 120 deg IR module running a plain JPEG
   streamer (OpenIris-compatible), Baballonia's face model on the PC. Gives a working mouth
   tracker on day one and verifies placement and lighting in the real mask.
2. **Teacher data.** Record the IR mouth stream together with Babble's outputs (and, without the
   mask, a frontal phone video for MediaPipe as a second teacher). A calibration routine with
   prompted expressions (neutral, jaw open, smile L/R, pucker, funnel, mouth L/R, tongue)
   covers the extremes.
3. **Student on the ESP.** A small own CNN (~64-96 px grayscale crop, 5-6 convs, a handful of
   outputs: jawOpen, smile L/R, pucker/funnel, mouthLeft/Right, mouthClose, tongueOut), distilled
   from the teacher, int8 via ESP-DL; per-user fine-tune from the calibration recordings.
4. **Classical fallback.** jawOpen from the dark mouth-opening area and lip-corner positions by
   NCC templates (brow_tracker style), working without any model, also a sanity check for the CNN.
5. Output line like EYE/BROW to the hub, Unified Expressions names so the values can also feed
   VRCFaceTracking if ever needed.

Sim: the brow_tracker head models plus a parametric jaw/lip rig can render the IR view from
candidate muzzle positions for a placement sweep (as in `../eye_tracker/docs/placement.md`)
once the muzzle geometry is known.
