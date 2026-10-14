# Brow tracking inside the mask - feasibility (2026-10-09)

Question: can a small camera inside the Arka'vali mask track the wearer's eyebrows
(raise/lower per brow first, frown second, inner/outer ends third) with plain image
maths cheap enough for an ESP32, before rebuilding the mask around a camera mount?

## Short answer

Yes, with conditions that are mostly about where the camera and the LEDs go:

1. Seen from above the mirror holder, the reliable feature is the **upper edge of the
   brow** (bright forehead -> dark hair). The lower half of the brow turns away under the
   brow ridge. The tracker follows that edge in 5 strips along a calibrated curve.
2. On the user's **real footage** (phone held above the brows, with glasses), the tracker
   follows raises and frowns on both brows; MediaPipe finds no face at all in those clips.
   On aligned frontal clips it matches MediaPipe with r 0.73-0.90 on 4 of 6 brows
   (failures: a lamp highlight on the forehead, a hand-held camera offset).
3. In the **simulator** (8 actors, mask-camera view re-rendered from frontal video), the CAD
   "red dot" placement tells a deliberate raise / lowering from rest in ~90% of single
   frames (AUC 0.85 / 0.97). An extra LED below the mirror lighting the eye socket (the brow
   becomes a dark band between two lit areas) is the best variant (raise AUC 0.99). Camera
   positions too close / too steep are clearly worse; the other placements tried are within
   the spread between people.
4. The tracker is ported to C++ (`firmware/lib/brow_core`), matches the Python reference
   (double: 1e-10 px, float: <0.001 px on most brows) and needs ~0.1 ms per brow per
   frame on a PC. Measured on the plain ESP32-CAM: 3-5 ms per brow, ~7.5 ms for both, at
   25 fps camera rate (after raising the OV2640 clock, see README) - the S3 is not needed.

What is NOT proven yet: the real near-IR appearance of the user's brows under 850/940 nm
LEDs, sweat/specular highlights, and mask slip. These need the real camera (next step).

## Evidence

### Real footage of the user (no simulation)

Aligned frontal clips (similarity transform on rigid landmarks = "camera fixed to the
head"), tracker vs MediaPipe brow landmarks on the same frames (`tools/stab_eval.py`).
Slope = tracker mm per MediaPipe mm; RMSE after a linear fit (what the per-user range
calibration does).

| clip | brow | r raise | r inner | r outer | slope | RMSE, mm | MediaPipe span, mm | notes |
|---|---|---|---|---|---|---|---|---|
| own_89 (close, dim, side light) | R | 0.89 | 0.77 | 0.76 | 1.22 | 0.89 | 6.2 | |
| own_89 | L | 0.54 | 0.64 | 0.43 | 0.80 | 2.36 | 8.8 | lamp highlight on this side of the forehead |
| own_511 (far, head pitch 17-39 deg) | R | 0.73 | 0.77 | 0.40 | 1.06 | 1.86 | 8.3 | 2-D alignment cannot undo pitch changes |
| own_511 | L | 0.57 | 0.68 | 0.55 | 0.79 | 2.32 | 9.0 | |
| own_530 (far, even light) | R | 0.83 | 0.87 | 0.56 | 0.87 | 1.65 | 10.4 | |
| own_530 | L | 0.90 | 0.91 | 0.69 | 1.11 | 1.19 | 8.8 | |

Top-down clips (phone above the brows, `tools/track_video.py`, ROIs placed by hand):
own_90 and own_96 show clean step-like raise/lower/frown signals per brow, including one-
brow movements; own_534 (two brows) too. The glasses rim is a stronger edge than the brow
and sits below it - the calibration therefore takes the topmost brow-like edge, not the
strongest one. Hand-held camera motion shows up as brow motion in these clips (the mask
camera will be fixed).

MediaPipe on the same top-down clips: 0-1 faces found in 2000-3000 frames each; on the
mixed clips it loses the face once the camera is more than ~45 deg above the face.

### Simulator

8 RAVDESS actors x 60 clips (all emotions), 16 brows per config, OV2640-like 120 deg
fisheye, 320x240 grey, IR-like texture, LED next to the lens unless stated. Holder =
mirror frame 31.7 mm in front of the brow + diagonal arm to 14 mm (CAD sections).

| config | camera (above eye line / in front of brow skin) | r vs MP, camera view | r vs MP, head frame | raise vs rest: AUC / acc | lower vs rest: AUC / acc |
|---|---|---|---|---|---|
| **l_below**: + LED under the mirror | 57 / 39 mm | **0.86** | **0.77** | **0.99 / 97%** | 0.93 / 88% |
| r_160: 160x120 frames | 57 / 39 | 0.79 | 0.76 | 0.84 / 90% | 0.97 / 91% |
| l_noise6: 3x sensor noise | 57 / 39 | 0.78 | 0.73 | 0.90 / 94% | 0.95 / 87% |
| **h_red**: the CAD "red dot" | 57 / 39 | 0.77 | 0.74 | 0.85 / 91% | 0.97 / 90% |
| l_glasses: + glasses rim | 57 / 39 | 0.77 | 0.73 | 0.94 / 93% | 0.96 / 88% |
| h_center_w: one camera, 150 deg | 57 / 39, centre | 0.76 | 0.74 | 0.96 / 93% | 0.88 / 76% |
| l_ir05: weaker brow contrast in IR | 57 / 39 | 0.72 | 0.69 | 0.84 / 88% | 0.94 / 86% |
| h_red_low10: holder 10 mm lower | 57 / 39 | 0.70 | 0.67 | 0.80 / 88% | 0.93 / 87% |
| h_lateral: camera 10 mm towards the temple | 57 / 39 | 0.69 | 0.68 | 0.89 / 95% | 0.97 / 90% |
| h_center: one camera, 120 deg | 57 / 39, centre | 0.69 | 0.66 | 0.92 / 91% | 0.89 / 80% |
| h_red_edge: plain edge method | 57 / 39 | 0.67 | 0.69 | 0.87 / 89% | 0.95 / 89% |
| h_close | 50 / 25 | 0.67 | 0.59 | 0.86 / 88% | 0.72 / 69% |
| h_low10_close | 50 / 25, holder -10 | 0.65 | 0.60 | 0.89 / 88% | 0.75 / 69% |
| h_steep | 70 / 30 | 0.57 | 0.45 | 0.80 / 81% | 0.72 / 65% |
| h_vclose | 45 / 15 | 0.44 | 0.45 | 0.81 / 81% | 0.71 / 68% |

Metrics: r vs MediaPipe's brow contour projected into the simulated camera (same viewpoint);
r vs MediaPipe brow height in the head frame; median over 16 brows. Control columns: telling
a raise (> +3 mm) or a lowering (< -2 mm) from rest (|h| < 1 mm) per frame without any
smoothing - AUC and accuracy of a threshold halfway between the group medians
(`tools/control_check.py`). MediaPipe itself is a noisy reference, so these are lower bounds.

Readings:
- The differences between the red dot, a central camera and a temple-side camera are within
  the spread between people (2 actors looked very different from 8 - small samples mislead).
- Robust effects: the extra LED below the mirror helps everything (raise detection 0.99 AUC);
  too close / too steep placements lose the lowering direction (AUC ~0.72) because the brow
  is seen edge-on and slides under the brow ridge.
- 160x120 frames are as good as 320x240 (the brow is a large, soft structure).
- Glasses (thin rim below the brow) and 3x noise do not hurt; weaker IR contrast costs a little.
- The holder's diagonal arm under the outer brow end is the main source of outer-end errors
  (the tracker strips can stick to its sharp edge); keep it out of the brow ROI if possible.
- Frown proxy (inner end lowering vs MediaPipe browDown): r ~0.4 for most placements,
  0.58 with the LED below; a dedicated frown channel is still to do.

## Recommendations for the mask

- **LEDs**: add (or share with a future pupil camera) a diffused IR LED below the mirror
  that lights the eye socket and the lower brow, plus one near the brow camera. Keep LEDs
  out of direct view of the eye (diffuser / baffle), low current.
- **Camera**: the planned spot above the mirror holder (~45 deg above the brow, 5-6 cm away)
  is fine; one central wide camera for both brows works about as well. Do not move it closer
  or steeper (lens next to the holder tip): the brow is seen edge-on and lowering is lost.
  If possible keep the holder's arm out of the image area right under the brow.
- **Lens**: the 120 deg OV2640 is enough; 320x240 or even 160x120 grey frames suffice.
- **ROI**: drawn once per brow, from below the hairline to below the brow, with ~12 mm of
  room above the brow for a full raise.
- **Calibration each time the mask is put on**: neutral face (~15 frames), then "range up"
  and "range down" to scale the output to -1..+1 for control.

## Next steps

1. Mount the ESP32-CAM (IR-cut filter removed) on a temporary bracket in the planned spot,
   flash `firmware/`, run `tools/brow_cam.py COMx setup`, then `monitor`.
2. Record raw grey frames with IR LEDs (`snap`, or a short recording mode) to replace the
   simulator's IR guess with real data; rerun `track_video.py` on them.
3. Send the values to `main_hub` over UDP (same text format as the face-tracking link).
4. Frown as its own channel (inner ends lowering + moving together) on top of raise/lower.
