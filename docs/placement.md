# Eye camera placement: under the AR mirror vs through the combiner (simulation, 2026-10-10)

Setup: `tools/eyesim.py` ray-traced right eye, OV2640 66 deg at 320x240 grey, 25 fps,
70 s per run (9-point calibration, then fixations / saccades / pursuit within
+-30 deg yaw, +-20 deg pitch, blinks every 2.5-5.5 s, two 2 s squints, pupil 1.5-3.5 mm,
mask slip of 1.8 mm + 2 deg in the second half). Numbers: `out/summary.txt`
(`evaluate.py`). "narrow" = lid opening narrowed to match the user's phone clips
(`_datasets/own/eye`), which show a narrower opening than the textbook default.

| placement | pupil found, open eye | while squinting | gaze err, 9-pt calib (before / after slip) | pupil+glint calib (before / after slip) |
|---|---|---|---|---|
| under (cam 39 mm, 43 deg below), LEDs below | 0.92 | 0.32 | 2.4 / 13.4 deg | no glints visible |
| under, LEDs at the top corners of the frame | 0.89 | 0.30 | 2.8 / 12.9 | 6.7 / 7.1 |
| same, narrow lids | **0.58** | **0.00** | 3.9 / 15.7 | 12.5 / 18.6 |
| through the combiner (49 mm, 9 deg below), LEDs on the frame sides | 0.95 | 0.96 | 1.2 / 5.5 | **2.6 / 2.7** |
| same, narrow lids | 0.94 | 0.98 | 2.5 / 7.1 | 11.7 / 11.6 |
| through, LEDs behind the combiner | 0.97 | 0.89 | 1.3 / 5.0 | 2.5 / 2.3 |

Findings

1. **Through the combiner is the better view.** The pupil stays visible when the lids
   narrow (squint 0.96-0.98 vs 0-0.3 from below), vertical gaze is not compressed
   (0.95 vs 0.74 px/deg) and the calibrated error is about half. From below, the lower
   lid and lashes cover the pupil as soon as the opening narrows.
2. **Its cost is light and stray reflections.** Each pass through the two-way mirror
   keeps ~T of the light (0.35 assumed), so the camera needs ~3x the LED power or
   exposure; LEDs behind the mirror get T^2 and put a ghost of the LED in the image.
   The camera has no IR filter, so the AR screen, room light and their reflections in
   the mirror reach it: a visible-blocking IR-pass filter on the lens is needed, LEDs go
   on the eye side of the mirror, and the camera side should be dark.
3. **Mask slip dominates the error.** A 1.8 mm / 2 deg shift turns a 1-3 deg
   calibration into 5-16 deg. Corneal glints fix it while two glints are reliably found
   (2.3-2.7 deg after slip), and the eye-sphere model halves the slip error but is noisy
   (5-9 deg). From below, LEDs under the mirror give no usable glints at all (the
   reflection lands behind the lower lid); LEDs at the top corners of the frame do.
4. **Blink / squint (update, same day).** v1: no accepted pupil = closed; the pupil's
   visible share under the upper lid = partial opening. A gate (ring contrast >= 0.45 x
   the running median, >= 55% of rays on the ellipse) removed the lash-clump "pupils" on
   closed eyes in the through view (0.67 -> 0.10 of closed frames). Blinks are caught in
   80-100% of closed frames everywhere, with 3-9% false blinks on open eyes (21-41% for
   the steep / narrow views from below). Squint is only visible this way from below
   (the lid covers the pupil); from the front the pupil stays clear, so the through view
   needs a lid-edge measure. A ridge regressor on a raw eye patch (`lids.py`) did not
   generalise beyond the three calibration prompts on the simulated lids; lid features
   will be designed on real IR footage. The eye-sphere model works poorly in the
   frontal through view (pupil ellipses are nearly round, minor axes ill-defined), so
   there glints are the slip fix, not the sphere.
5. **Glints, made robust (update).** Glint candidates must be isolated peaks (frame
   around the blob >= 50 grey levels darker), and the LED pair is picked by its learned
   left->right spacing (`GlintPairer`). Through the mirror with LEDs on the frame sides
   the pupil-glint calibration now gives **1.6 deg before / 1.4 deg after the slip**
   (p90 3.7); with narrow lids 5.8 / 5.1 (was 11.7). Full table: `summary_sim.txt`.
6. **Glasses (the user will wear them; -2 D uncoated lens 12 mm from the cornea, frame
   rim, both lens surfaces mirror the LEDs).** Each LED shows up twice more as a big
   saturated disc. Where it lands depends only on where the LED sits
   (`led_glasses.txt`): LEDs next to the camera behind the mirror put the reflection
   right on the pupil (5-14 px away) and detection collapses to 62%; LEDs on the mirror
   frame's sides or top corners keep it 65-120 px away. Through the mirror with side
   LEDs, tracking with glasses holds: pupil found 95%, blinks 97% caught, pupil-glint
   gaze 5.9 / 5.2 deg with narrow lids (same as without glasses); squint detection drops
   to 64% (`summary_glasses.txt`). From below, the frame's bottom rim crosses the view
   (as in the user's clip) and detection falls to 64%. An anti-reflection coating only
   dims the reflections (still saturated in the sim at 0.8% per surface). Best LED spots
   with glasses: top corners of the mirror frame, or the sides set wide (+-32 mm).
7. **Open problems in the tracker**: the detector "finds" a pupil in the lash line of a
   closed eye (fixed by the gate, see 4); squint from the front (needs real IR footage);
   glint pairs with narrow lids still sometimes wrong.

The user's phone clips are visible light: pupil and iris are equally dark there, so they
only check geometry (lids, view angle), not the detector (23% detections on the clip
from below). Real tests need the OV2640 with 850 nm LEDs.
