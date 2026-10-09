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
3. In the **simulator** (8 actors, mask-camera view re-rendered from frontal video), the
   result depends strongly on placement and lighting - see the table below. Best: an LED
   below the mirror lighting the eye socket (the brow becomes a dark band between two lit
   areas), a camera that does not see the holder's arm right under the brow, about 45 deg
   above the brow at 5-6 cm. Worst: camera too close / too steep.
4. The tracker is ported to C++ (`firmware/lib/brow_core`), matches the Python reference
   (double: 1e-10 px, float: <0.001 px on most brows) and needs ~0.1 ms per brow per
   frame on a PC; an ESP32 estimate is a few ms (measure with the `bench` command).

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

SIM_TABLE_PLACEHOLDER

Metrics: r vs MediaPipe's brow contour projected into the simulated camera (same viewpoint),
r vs MediaPipe brow height in the head frame, median over actors and both brows.

## Recommendations for the mask

- **LEDs**: add (or share with a future pupil camera) a diffused IR LED below the mirror
  that lights the eye socket and the lower brow, plus one near the brow camera. Keep LEDs
  out of direct view of the eye (diffuser / baffle), low current.
- **Camera**: above the mirror holder, ~45 deg above the brow, 5-6 cm away; avoid placing
  it so the holder's arm or frame edge appears right under the brow in the image (shift it
  towards the temple, or use one central camera above the nose bridge for both brows).
  Too close / too steep (lens next to the holder tip) does not work: the brow is seen edge-on.
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
