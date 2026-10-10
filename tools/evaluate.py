"""Run the pupil detector + eye model on rendered sequences and compare with ground truth.

python evaluate.py [names ...]   (reads out/seq/<name>.npz, writes out/eval/<name>.json + plot)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from eyemodel import Calib, EyeModel, yaw_pitch
from pupil import PupilDetector, PupilGate, find_glints

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "out" / "seq"
EVAL = ROOT / "out" / "eval"


def ang_err(y1, p1, y2, p2):
    a = np.stack([np.sin(np.radians(y1)) * np.cos(np.radians(p1)), np.sin(np.radians(p1)),
                  np.cos(np.radians(y1)) * np.cos(np.radians(p1))], -1)
    b = np.stack([np.sin(np.radians(y2)) * np.cos(np.radians(p2)), np.sin(np.radians(p2)),
                  np.cos(np.radians(y2)) * np.cos(np.radians(p2))], -1)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def run(name: str, warmup_s: float = 15.0) -> dict:
    d = np.load(SEQ / f"{name}.npz")
    imgs, gts = d["imgs"], json.loads(str(d["gt"]))
    camR, f = d["cam_R"], float(d["f"])
    h, w = imgs.shape[1:]
    det = PupilDetector()
    gate = PupilGate()
    model = EyeModel(f, w, h)
    free = [i for i, g in enumerate(gts) if g["phase"] == "free"]
    t0 = time.perf_counter()
    for i in free[: int(warmup_s * 25)]:       # look around before calibrating
        e = det.detect(imgs[i])
        if gate(e):
            model.update(e)
    rows = []
    for i, g in enumerate(gts):
        e = det.detect(imgs[i])
        raw = e is not None
        if not gate(e):
            e = None
        r = dict(i=i, yaw=g["yaw"], pitch=g["pitch"], open=g["openness"], squint=g["squint"],
                 phase=g["phase"], target=g["target"], settled=g["settled"], slip=g["slip"],
                 vis=g["vis_px"] / max(g["full_px"], 1e-9) if g["full_px"] > 0 else 0.0, det=e is not None,
                 det_raw=raw)
        if e is not None:
            model.update(e)
            r.update(cx=e.cx, cy=e.cy, a=e.a, b=e.b, support=e.support, contrast=e.contrast, top_vis=e.top_vis)
            gl = find_glints(imgs[i], e)
            r["n_glints"] = len(gl)
            if len(gl) >= 2:     # pupil minus the mean of the two nearest glints
                r["gx"] = e.cx - (gl[0][0] + gl[1][0]) / 2
                r["gy"] = e.cy - (gl[0][1] + gl[1][1]) / 2
            if "pc_vis" in g:
                r["pc_err"] = float(np.hypot(e.cx - g["pc_vis"][0], e.cy - g["pc_vis"][1]))
            gc = model.gaze_cam(e)
            if gc is not None:
                r["m_yaw"], r["m_pitch"] = yaw_pitch(camR.T @ gc)
        if model.ready:
            r["c"] = model.c.tolist()
            r["R"] = model.R
        r["eye_c"] = g["eye_c"]
        rows.append(r)
    # eyelid opening v1: no pupil -> closed; pupil partly under the upper lid -> its visible
    # share of the upward extent. The patch regressor (lids.py) did not generalise past the
    # prompts on the simulated lids; revisit on real IR footage.
    for r in rows:
        r["open_est"] = r.get("top_vis", 1.0) if r["det"] else 0.0
    ms = (time.perf_counter() - t0) / (len(gts) + len(free[: int(warmup_s * 25)])) * 1000

    # calibration fits (settled calibration frames with a detection)
    cal = [r for r in rows if r["phase"] == "calib" and r["settled"] and r["det"] and "m_yaw" in r]
    Y = np.array([[r["yaw"], r["pitch"]] for r in cal])
    cal_pupil = Calib().fit([[r["cx"], r["cy"]] for r in cal], Y)
    cal_model = Calib(order=1).fit([[r["m_yaw"], r["m_pitch"]] for r in cal], Y)
    # pupil relative to the model's eyeball centre: the centre follows the camera when the mask slips
    cal_rel = Calib().fit([[r["cx"] - r["c"][0], r["cy"] - r["c"][1]] for r in cal], Y)
    calg = [r for r in cal if "gx" in r]
    cal_glint = Calib().fit([[r["gx"], r["gy"]] for r in calg], [[r["yaw"], r["pitch"]] for r in calg])         if len(calg) >= 30 else None
    for r in rows:
        if cal_glint is not None and "gx" in r and r["det"]:
            r["g_yaw"], r["g_pitch"] = cal_glint([[r["gx"], r["gy"]]])[0]
        if r["det"]:
            r["p_yaw"], r["p_pitch"] = cal_pupil([[r["cx"], r["cy"]]])[0]
            if "m_yaw" in r:
                r["mc_yaw"], r["mc_pitch"] = cal_model([[r["m_yaw"], r["m_pitch"]]])[0]
            if "c" in r:
                r["r_yaw"], r["r_pitch"] = cal_rel([[r["cx"] - r["c"][0], r["cy"] - r["c"][1]]])[0]

    free_rows = [r for r in rows if r["phase"] == "free"]
    slip_max = max(r["slip"] for r in free_rows)
    is_open = lambda r: r["open"] >= 0.8 and r["squint"] < 0.05
    sets = {
        "before_slip": [r for r in free_rows if is_open(r) and r["slip"] == 0],
        "after_slip": [r for r in free_rows if is_open(r) and r["slip"] >= 0.99 * slip_max],
    }
    res = dict(name=name, ms_per_frame_py=ms, n=len(rows))
    open_rows = [r for r in free_rows if is_open(r)]
    res["detect_open"] = float(np.mean([r["det"] for r in open_rows]))
    res["detect_squint"] = float(np.mean([r["det"] for r in free_rows if r["squint"] > 0.5]))
    res["false_in_blink"] = float(np.mean([r["det"] for r in free_rows if r["open"] <= 0.1]))
    oe = [(r["open"], r["open_est"]) for r in free_rows if "open_est" in r]
    if oe:
        o = np.array(oe)
        res["open_corr"] = float(np.corrcoef(o[:, 0], o[:, 1])[0, 1])
        res["open_mae"] = float(np.mean(np.abs(o[:, 0] - o[:, 1])))
        res["blink_caught"] = float(np.mean(o[o[:, 0] <= 0.1, 1] < 0.3))
        res["open_false_blink"] = float(np.mean(o[o[:, 0] >= 0.9, 1] < 0.3))
        sq = [r["open_est"] for r in free_rows if r["squint"] > 0.5 and "open_est" in r]
        res["squint_est_mean"] = float(np.mean(sq)) if sq else None
    res["false_in_blink_raw"] = float(np.mean([r["det_raw"] for r in free_rows if r["open"] <= 0.1]))
    pe = [r["pc_err"] for r in open_rows if "pc_err" in r and r["vis"] > 0.95]
    res["pupil_px_err_med"] = float(np.median(pe)) if pe else None
    for key, rs in sets.items():
        for meth, (ky, kp) in {"model_raw": ("m_yaw", "m_pitch"), "model_cal": ("mc_yaw", "mc_pitch"),
                               "pupil_cal": ("p_yaw", "p_pitch"), "rel_cal": ("r_yaw", "r_pitch"),
                               "glint_cal": ("g_yaw", "g_pitch")}.items():
            rr = [r for r in rs if ky in r]
            if not rr:
                continue
            e = ang_err(np.array([r[ky] for r in rr]), np.array([r[kp] for r in rr]),
                        np.array([r["yaw"] for r in rr]), np.array([r["pitch"] for r in rr]))
            res[f"{key}/{meth}/med"] = float(np.median(e))
            res[f"{key}/{meth}/p90"] = float(np.percentile(e, 90))
        res[f"{key}/detect"] = float(np.mean([r["det"] for r in rs]))
        res[f"{key}/glint2"] = float(np.mean([r.get("n_glints", 0) >= 2 for r in rs]))
    # model centre vs the true projection of the eyeball centre
    last = [r for r in rows if "c" in r]
    if last:
        res["centre_err_px_end"] = float(np.hypot(last[-1]["c"][0] - last[-1]["eye_c"][0],
                                                   last[-1]["c"][1] - last[-1]["eye_c"][1]))
        res["R_px_end"] = last[-1]["R"]
    # sensitivity: pupil px per degree of gaze around the centre (from the calibration fit)
    A = np.c_[Y, np.ones(len(Y))]
    J = np.linalg.lstsq(A, np.array([[r["cx"], r["cy"]] for r in cal]), rcond=None)[0][:2]
    res["px_per_deg_yaw"] = float(np.linalg.norm(J[0]))
    res["px_per_deg_pitch"] = float(np.linalg.norm(J[1]))
    res["info"] = json.loads(str(d["info"]))
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"{name}.json").write_text(json.dumps(res, indent=1))
    np.save(EVAL / f"{name}_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
    return res


def main():
    names = sys.argv[1:] or sorted(p.stem for p in SEQ.glob("*.npz"))
    for n in names:
        res = run(n)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
