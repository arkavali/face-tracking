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
once the muzzle geometry is known. The renderer is to be written for the GPU from the start
(CuPy via `../eye_tracker/tools/gpu.py`, RTX 5070), with a NumPy fallback.

## Placement sweep (2026-10-11, `tools/sweep.py`, `tools/preview.py`)

Head and mask meshes exported read-only from the Fusion design (`avali_head_V0.5 v63`: mask body
`forehead R`, head mesh `ImageToStl.com_vhs+`). Camera on the midline aimed at the mouth ROI (lips
+-34 mm with smile margin, lower lip and chin of a 22 deg open jaw), worst case over the camera
shifted +-10 mm in x/y/z, 320x240, fan zone (z > 340 mm, beak tip) kept free. Mouth slit at
y = 99, z = 246 mm; inner beak wall ~125 mm in front of it.

| Lens | Closest full-cover spot | Scale on lips | Full-cover spots outside the fan zone |
|---|---|---|---|
| 120 deg | 45 mm in front, 26 mm above the slit | 3.0 px/mm | 114 |
| 160 deg | 40 mm in front, 16 mm above | 2.5 px/mm | 136 |
| 66 deg | 75 mm in front, 46 mm above | 3.3 px/mm | 27 (narrow band) |

Recommended: **120 deg, ~55-60 mm in front of the mouth and ~30 mm above it** (just below the
nose tip, tilted down at the mouth): one step inside the full-cover zone, ~2.5 px/mm, mouth fills
most of the frame. 66 deg also works but only far forward and high, close to the fan zone, and a
small mask shift loses the chin. Maps: `placement_sweep.png`; views: `placement_preview.png`.

Caveats: generic head mesh, not the wearer's face; the mask's jaw/palate are not modelled yet;
lens FOV taken as diagonal (equidistant fisheye for 120/160 deg).

## IR LED layout (2026-10-11, `tools/ledsim.py`, `tools/lipshapes.py`, `tools/rt.py`)

Camera at the recommended spot (55 mm in front of the mouth slit, 30 mm above, 120 deg). Mouth
shapes are smooth deformations of the Fusion head mesh (jaw open 16 deg, smile, pucker, pucker up,
"o", cheek puff; rough, not anatomy). 137 candidate LED positions in the free space of the beak,
point LEDs with a cos^2 beam aimed at the mouth, shadows, wet-lip highlights, auto exposure.
Layout score = how far apart the shapes stay in the 80x60 net input relative to the spread from a
+-5 mm mask shift (mean log ratio over 8 shape pairs). Full table: `led_report.txt`, pictures:
`led_layouts.png` (rows: LED at the lens, then the best layouts).

- One LED at the lens (flat frontal light) is the worst: score -0.76, 58 % of the mouth region
  nearly black after exposure; relief (pucker, "o" rim) barely shows.
- **Best: a symmetric pair at the sides, x = +-45 mm, at mouth height (-10..+10 mm), 45-65 mm in
  front of the lips, aimed at the mouth**: score -0.43..-0.52, dark 30-37 %. Side light throws
  shading on the pushed-out lips, which is what separates pucker / "o" / open jaw.
- Two such pairs (one slightly above, one slightly below the mouth line) add a little (-0.29..-0.38).
- Hardest pairs in every layout: neutral vs smile and pucker vs pucker-up (small geometric change).
- Raw speed: the CUDA kernel casts 20k rays against 44k triangles in ~30 ms; the whole sweep
  (7 shapes x 7 shifts x 137 LEDs, all with shadow rays) takes ~100 s.
