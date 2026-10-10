# eye_tracker

Gaze, blink and squint tracking for the Arka'vali mask: one ESP32-S3 CAM (OV2640, 66 deg,
no IR filter) per eye, plain image maths on the chip, results sent to the hub. Sibling of
`../brow_tracker` (same layout, same venv).

Method base: JEOresearch 3DTracker (darkest-area seed, pupil ellipse, eyeball centre from
the minor axes of many pupil ellipses, gaze = centre -> pupil), reworked for the ESP:
starburst rays + Fitzgibbon ellipse fit with RANSAC instead of OpenCV contours, and the
eye-sphere model re-estimated continuously (slow forgetting) so it follows mask slip.

The boards are not here yet, so everything so far runs on a synthetic eye.

## Tools (`tools/`)

| Script | What it does |
|---|---|
| `eyesim.py` | Ray-traced near-IR eye: refracting cornea, iris/pupil, lids that follow the gaze, blink/squint, lashes, LED glints, AR screen reflection, semi-transparent combiner plate, sensor noise |
| `placements.py` | Camera/LED placements for one eye (under the AR mirror at several distances, through the combiner) |
| `preview.py` | Contact sheet: placements x eye states (`out/preview.png`) |
| `sequence.py` | 25 fps sequences: 9-point calibration, fixations/saccades/pursuit, blinks, squints, pupil size drift, mask slip |
| `render_seq.py` | Render a sequence per placement to `out/seq/*.npz` (parallel) |
| `pupil.py` | Pupil detector (seed, starburst, RANSAC ellipse) written for an ESP port |
| `eyemodel.py` | Incremental eye-sphere model, gaze vector, polynomial calibration |
| `evaluate.py` | Detector + model on rendered sequences vs ground truth (`out/eval/`) |

Setup: uses `../brow_tracker/.venv` (Python 3.9, numpy, opencv-python, matplotlib).

```
PY=../brow_tracker/.venv/Scripts/python
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
  film, no motion blur (saccades span 1-2 frames), no glasses yet.
- The AR combiner pose is estimated from brow_tracker's holder numbers, not from the CAD.
- Noise is relative: the through-mirror camera gets 1/T less light at the same LED power.
