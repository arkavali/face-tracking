# Mouth tracking for the Arka'vali mask: survey and plan

Goal: jaw / lip expression of the wearer from a small camera inside the muzzle, computed on an
ESP32-S3 CAM node (like `../eyes` and `../brows`), raw values sent to the hub,
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
candidate muzzle positions for a placement sweep (as in `../eyes/docs/placement.md`)
once the muzzle geometry is known. The renderer is to be written for the GPU from the start
(CuPy via `../eyes/tools/gpu.py`, RTX 5070), with a NumPy fallback.

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

## Open ideas

- **Time-multiplexed IR LEDs (photometric stereo).** 2-3 IR LEDs from different sides (e.g. the
  +-45 mm pair) switched alternately in step with camera frames (S3 GPIO on the VSYNC interrupt).
  Brightness ratios between frames give the surface normal, i.e. lip / cheek shape, not only
  texture. OV2640 is rolling-shutter: switch in vertical blanking with exposure shorter than the
  blanking, or use an A,A,B,B pattern and drop transition frames, or keep one LED steady and blink
  the other. Costs half the frame rate per source. Two-wavelength separation does not work (OV2640
  Bayer filters are nearly equal in NIR). Mouth LEDs may be brighter than eye LEDs, aimed away from
  the eyes and glasses.
  Decided form (user, 2026-10-11): two full frames (L lit, R lit) as two input channels, sliding
  pairs so the output keeps the camera rate; row-striped frames rejected (much harder, no gain).
  The net only grows in its first layer (+16 % MAC). Open points: frame/LED tagging, switching in
  vertical blanking, locked exposure, ambient (dark) frame, recorder for pairs, LED drivers, a
  fallback when one LED fails.
  **Simulation check (`tools/ledsim2ch.py`, `led_report_2ch.txt`, `led_2ch.png`): no gain.** With 15 random
  +-5 mm mask shifts, L/R as two channels (common or separate exposure) or a (L-R)/(L+R) ratio
  channel separate the shapes no better than both side LEDs on at once (mean log separation
  -0.65..-1.09 vs -0.52..-0.63; nearest-centroid accuracy 0.46-0.58 vs 0.59-0.65), also with a
  random albedo pattern and +-15 % LED gain noise, which is where photometric stereo should help.
  The mask-shift spread dominates and the ratio image is very sensitive to it. Caveats: rough
  synthetic shapes, uniform skin, linear / nearest-centroid metrics instead of a trained net.
  Conclusion: side pair lit together by default; alternation only as a cheap hardware experiment
  later (e.g. one LED steady, one blinking) once the mask exists.

## LED layout, round 2 and trained-net study (2026-10-11/12)

**Round 2 sweep** (`tools/ledsweep2.py`, `led2_report.txt`, `led2_layouts.png`, `led2_map.png`):
one diffuse bounce off the inner beak wall (wall sampled into patches, albedo 0 / 0.5 / 0.8),
769 LED spots incl. near the camera, vertical pairs, triples, crosses. The bounce changes the
scores by <= 0.05; side pair +-45 mm at mouth height still best by the pixel metric, a cross
(side pair + vertical pair) slightly better, LEDs closer to the camera worse. Wider pairs
(+-55-65 mm, `led2_wide_report.txt`) leave less of the mouth dark but separate the shapes worse.

**Trained-net study** (`tools/synth.py`, reports `synth_report_*.txt`, picture `synth_samples.png`):
6000 synthetic frames per LED set = random mixes of 8 mouth deformations (jaw, smile L/R, pucker,
"o" opening, lips up, cheek puff L/R), random mask shift +-5 mm and aim change, smooth albedo
pattern, +-15 % LED gain, wall bounce, sensor noise, 8-bit; every LED rendered alone, lighting
variants built afterwards; the mouth CNN (width 8,12,16,24,32) trained per variant, 3-5 seeds,
val MAE on 1200 held-out frames (seed spread +-0.001).

| Lighting | val MAE | pucker |
|---|---|---|
| any LEDs lit together (lens, side pairs, vertical pair, cross, triple) | 0.031-0.033 | 0.026-0.033 |
| blinking side pair (L frame + R frame as 2 channels) | 0.030 | 0.027 |
| **blinking vertical pair** (LED above/behind the camera + LED below the lips) | **0.025-0.026** | **0.019-0.021** |
| control: vertical pair lit together, fed twice (2 identical channels) | 0.033 | 0.031 |
| blinking vertical pair + sides as 3rd/4th channel | 0.026-0.027 | 0.020 |

- The pixel-distance metric of round 1/2 could not see this: the gain is non-linear (light from
  below vs from above makes pushed-out lips cast different shadows), the trained net uses it.
- The control rules out noise averaging: two channels only help when they carry different light.
- Position of the vertical pair is not critical: all 16 tested up x down combinations give
  0.025-0.029; best: up (0, +55, +60) = above the camera, down (0, -35, +65) (x, mm above the
  slit, mm in front). Worse: up LED in front of the camera, down LED close to the face.
- All tested LED spots are outside the camera view (checked by projection); the earlier
  sweeps did not filter for this, but their winners are outside the view too.
- The beak has an opening at the bottom front (about 20-75 mm below the mouth line in the
  model): outside light can come in there (matters for blinking; a dark frame may be needed).
- Caveats: synthetic, rough deformations on a generic head, uniform NIR skin with blotches,
  no teeth / tongue; confirm on hardware once the mask is built.
- **Triangle (user's sketch, `synth_report_triangle.txt`, `synth_triangle.png`)**: top LED
  between camera and nose (0, +38, +40) or under the lens (0, +22, +55) + two low side LEDs
  (+-35..55, -12..-35, +55..65), 3-channel blinking: 0.027-0.029; top vs both low together
  (2 channels): 0.027; all three lit together: 0.034; reference vertical pair blink 0.026.
  Splitting the two low LEDs adds nothing.
- **Conclusion:** what matters is light from above and light from below in separate frames. Two
  LEDs (above the camera, below the lips) with 2-phase blinking are enough; the triangle is an
  equivalent fallback if a low centre mount is awkward (both low LEDs fire together). Balance
  the LED currents so both frames have similar brightness (the study used one locked exposure).

**All tried configurations** (LED spots, variants, results, re-run commands): `lighting_configs.md`
(+ `lighting_configs.json`), generated by `tools/configs_catalog.py`. The rendered synthetic sets stay
on disk in `out/synth/*.npz` (~1.5 GB, not in git): a new variant made of already rendered LEDs only
needs training (~16 s per run).
