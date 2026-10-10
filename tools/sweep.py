"""Camera placement sweep: how freely can the eye camera sit in the helmet?

Renders (render_seq.py, with glasses and narrow lids) and evaluates (evaluate.py) a grid of
camera positions around the eye, then summarises them in out/sweep/summary.json + a
heat-map sheet. Positions are named sweep_d<dist>_e<elev>_a<azim> (placements.py).
python sweep.py render|evaluate|report
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
PY = sys.executable
DISTS = [30, 45, 65]
ELEVS = [-50, -35, -20, -5, 10, 25]
AZIMS = [0, 25, -25]
TAG = "_ng"   # narrow lids + glasses


def nm(d, e, a) -> str:
    f = lambda v: f"{v}".replace("-", "m")
    return f"sweep_d{d}_e{f(e)}_a{f(a)}"


NAMES = [nm(d, e, a) for a in AZIMS for e in ELEVS for d in DISTS]


def render():
    todo = [n for n in NAMES if not (OUT / "seq" / f"{n}{TAG}.npz").exists()]
    print(len(todo), "to render")
    # ss=1 and no lid prompts: ~5x cheaper than the main runs (only compared within the sweep)
    subprocess.run([PY, str(HERE / "render_seq.py"), "--names", *todo, "--seconds", "30", "--narrow",
                    "--glasses", "0.04", "--tag", TAG, "--workers", "30", "--ss", "1", "--no-lidcal"], check=True)


def evaluate():
    jobs = [(n + TAG, box) for n in NAMES for box in (8, 16)]

    def one(job):
        name, box = job
        env = dict(os.environ, EYE_BOX=str(box), EYE_EVAL_DIR=f"sweep/box{box}")
        r = subprocess.run([PY, str(HERE / "evaluate.py"), name], env=env, capture_output=True, text=True)
        return name, box, r.returncode, r.stderr[-300:]

    with ThreadPoolExecutor(30) as ex:
        for name, box, rc, err in ex.map(one, jobs):
            if rc:
                print("FAILED", name, box, err)
    print("done")


KEYS = ["detect_open", "detect_squint", "blink_caught", "open_false_blink", "before_slip/pupil_cal/med",
        "before_slip/glint_cal/med", "after_slip/glint_cal/med", "after_slip/glint_cal/p90", "after_slip/glint2",
        "px_per_deg_yaw", "px_per_deg_pitch"]


def report():
    rows = []
    for a in AZIMS:
        for e in ELEVS:
            for d in DISTS:
                best = None
                for box in (8, 16):
                    f = OUT / "sweep" / f"box{box}" / f"{nm(d, e, a)}{TAG}.json"
                    if not f.exists():
                        continue
                    r = json.loads(f.read_text())
                    score = (r["detect_open"] + 0.5 * r.get("detect_squint", 0) + 0.5 * r.get("blink_caught", 0)
                             - r.get("open_false_blink", 1))
                    if best is None or score > best[0]:
                        best = (score, box, r)
                if best is None:
                    continue
                _, box, r = best
                rows.append(dict(d=d, e=e, a=a, box=box, **{k: r.get(k) for k in KEYS}))
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)
    (OUT / "sweep" / "summary.json").write_text(json.dumps(rows, indent=1))
    heat(rows)
    for r in rows:
        print(f"d{r['d']:3d} e{r['e']:4d} a{r['a']:4d} box{r['box']:3d}  det {r['detect_open']:.2f} sq {r['detect_squint']:.2f} "
              f"blink {r['blink_caught']:.2f}/{r['open_false_blink']:.2f}  pupil {r['before_slip/pupil_cal/med']:.1f}  "
              f"glint {fmt(r['before_slip/glint_cal/med'])}/{fmt(r['after_slip/glint_cal/med'])}")


def fmt(v):
    return "  - " if v is None else f"{v:4.1f}"


def heat(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("detect_open", "pupil found, open eye", 0, 1, "RdYlGn"),
              ("detect_squint", "pupil found while squinting", 0, 1, "RdYlGn"),
              ("blink_caught", "blink frames caught", 0, 1, "RdYlGn"),
              ("before_slip/pupil_cal/med", "gaze error, 9-pt calibration, deg", 0, 8, "RdYlGn_r"),
              ("after_slip/glint_cal/med", "gaze error after mask slip, deg (glints)", 0, 15, "RdYlGn_r")]
    fig, ax = plt.subplots(len(panels), len(AZIMS), figsize=(4.2 * len(AZIMS), 3.3 * len(panels)))
    for j, a in enumerate(AZIMS):
        for i, (k, title, lo, hi, cmap) in enumerate(panels):
            M = np.full((len(ELEVS), len(DISTS)), np.nan)
            for r in rows:
                if r["a"] == a and r[k] is not None:
                    M[ELEVS.index(r["e"]), DISTS.index(r["d"])] = r[k]
            axx = ax[i, j]
            im = axx.imshow(M, cmap=cmap, vmin=lo, vmax=hi, origin="lower", aspect="auto")
            for yi in range(len(ELEVS)):
                for xi in range(len(DISTS)):
                    if np.isfinite(M[yi, xi]):
                        axx.text(xi, yi, f"{M[yi, xi]:.2f}" if hi <= 1 else f"{M[yi, xi]:.0f}", ha="center",
                                 va="center", fontsize=8)
            axx.set_xticks(range(len(DISTS)))
            axx.set_xticklabels([f"{d} mm" for d in DISTS])
            axx.set_yticks(range(len(ELEVS)))
            axx.set_yticklabels([f"{e:+d} deg" for e in ELEVS])
            side = "straight" if a == 0 else ("nasal" if a > 0 else "temporal") + f" {abs(a)} deg"
            axx.set_title(f"{title}\n{side}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "sweep" / "sweep_heat.png", dpi=80)


if __name__ == "__main__":
    {"render": render, "evaluate": evaluate, "report": report}[sys.argv[1]]()
