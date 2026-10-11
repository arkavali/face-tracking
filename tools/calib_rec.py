"""Prompted calibration recording: the window shows an expression, the wearer copies it, every
frame gets a label vector (own channel set, see CHANNELS).

  python calib_rec.py [COMx] [--reps 1] [--seed N]
Keys: Space pause/resume, N skip the current prompt, Q/Esc stop (what is recorded is kept).
Mistake: Space (pause), Left/Right to pick the prompt to go back to, Space again: that prompt is
redone from its rest phase and the frames recorded for it and after it get valid=0.

Per prompt: rest (neutral, shows what comes next) -> ramp in -> hold -> ramp out. The label is the
prompt's target vector times an envelope 0..1 (0 at rest, 1 while holding), so the model also
sees intermediate intensities. The "free" prompt (talking / anything) is stored with NaN labels.
Output out/calib/cal_YYYYmmdd_HHMMSS/: frames.mjpeg, frames.csv (as mouth_cam.py) and
labels.csv (seq, prompt, phase, envelope, one column per channel).
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mouth_cam import Stream, find_port

OUT = Path(__file__).resolve().parent.parent / "out" / "calib"
CHANNELS = [
    "jawOpen", "smileL", "smileR", "frownL", "frownR", "funnel", "pucker", "mouthLeft", "mouthRight",
    "lipsUp", "lipsDown", "snarl", "press", "cheekPuffL", "cheekPuffR", "jawLeft", "jawRight",
    "tongueOut", "tongueLeft", "tongueRight", "tongueUp", "tongueDown",
]
# (id, text shown to the wearer, {channel: target})
PROMPTS = [
    ("neutral", "Нейтральное лицо, рот закрыт", {}),
    ("jaw_full", "Рот широко открыт", {"jawOpen": 1}),
    ("jaw_half", "Рот приоткрыт наполовину", {"jawOpen": 0.5}),
    ("smile", "Улыбка, рот закрыт", {"smileL": 1, "smileR": 1}),
    ("smile_l", "Улыбка только ЛЕВОЙ стороной", {"smileL": 1}),
    ("smile_r", "Улыбка только ПРАВОЙ стороной", {"smileR": 1}),
    ("smile_open", "Улыбка с открытым ртом", {"smileL": 1, "smileR": 1, "jawOpen": 0.6}),
    ("smile_l_open", "Левая улыбка + рот открыт", {"smileL": 1, "jawOpen": 0.6}),
    ("smile_r_open", "Правая улыбка + рот открыт", {"smileR": 1, "jawOpen": 0.6}),
    ("frown", "Уголки рта вниз (грусть)", {"frownL": 1, "frownR": 1}),
    ("funnel", "Губы буквой «О»", {"funnel": 1, "jawOpen": 0.3}),
    ("pucker", "Трубочка (поцелуй)", {"pucker": 1}),
    ("pucker_l", "Трубочка влево", {"pucker": 1, "mouthLeft": 1}),
    ("pucker_r", "Трубочка вправо", {"pucker": 1, "mouthRight": 1}),
    ("pucker_up", "Трубочка вверх, к носу", {"pucker": 1, "lipsUp": 1}),
    ("pucker_down", "Трубочка вниз", {"pucker": 1, "lipsDown": 1}),
    ("snarl", "Оскал: губы трубочкой, но зубы видны", {"snarl": 1, "funnel": 0.5}),
    ("mouth_l", "Рот (закрытый) сдвинуть влево", {"mouthLeft": 1}),
    ("mouth_r", "Рот (закрытый) сдвинуть вправо", {"mouthRight": 1}),
    ("press", "Сжать губы", {"press": 1}),
    ("puff", "Надуть обе щёки", {"cheekPuffL": 1, "cheekPuffR": 1}),
    ("puff_l", "Надуть ЛЕВУЮ щёку", {"cheekPuffL": 1}),
    ("puff_r", "Надуть ПРАВУЮ щёку", {"cheekPuffR": 1}),
    ("jaw_l", "Челюсть влево", {"jawLeft": 1}),
    ("jaw_r", "Челюсть вправо", {"jawRight": 1}),
    ("tongue_out", "Язык наружу, прямо", {"tongueOut": 1, "jawOpen": 0.4}),
    ("tongue_l", "Рот открыт, язык в ЛЕВЫЙ уголок", {"tongueOut": 0.5, "tongueLeft": 1, "jawOpen": 0.5}),
    ("tongue_r", "Рот открыт, язык в ПРАВЫЙ уголок", {"tongueOut": 0.5, "tongueRight": 1, "jawOpen": 0.5}),
    ("tongue_up", "Язык наружу и на верхнюю губу", {"tongueOut": 1, "tongueUp": 1, "jawOpen": 0.4}),
    ("tongue_down", "Язык наружу и на нижнюю губу", {"tongueOut": 1, "tongueDown": 1, "jawOpen": 0.4}),
    ("free", "Свободно: говори, гримасничай", None),
]
# seconds: rest, ramp in, hold, ramp out
PHASES = [("rest", 1.5), ("in", 1.0), ("hold", 2.0), ("out", 1.0)]
FREE_SECONDS = 8.0
KEY_LEFT, KEY_RIGHT = 2424832, 2555904   # cv2.waitKeyEx codes on Windows

FONT = None
for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
    try:
        FONT = ImageFont.truetype(f, 26), ImageFont.truetype(f, 18)
        break
    except OSError:
        pass


def panel(width, text, sub, progress, colour):
    img = Image.new("RGB", (width, 90), (25, 25, 25))
    d = ImageDraw.Draw(img)
    big, small = FONT or (ImageFont.load_default(), ImageFont.load_default())
    d.text((12, 8), text, font=big, fill=colour)
    d.text((12, 50), sub, font=small, fill=(170, 170, 170))
    d.rectangle([0, 84, int(width * progress), 90], fill=colour)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def schedule(reps, seed):
    rng = random.Random(seed)
    items = []
    for _ in range(reps):
        body = PROMPTS[1:-1][:]
        rng.shuffle(body)
        items += [PROMPTS[0]] + body + [PROMPTS[-1]]
    return items


def envelope(phase, f):
    return {"rest": 0.0, "in": f, "hold": 1.0, "out": 1.0 - f}[phase]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    items = schedule(args.reps, args.seed if args.seed is not None else int(time.time()))
    total = sum(FREE_SECONDS if p[2] is None else sum(s for _, s in PHASES) for p in items)
    print(f"{len(items)} prompts, ~{total / 60:.1f} min. Space pause, N skip, Q stop.", flush=True)

    st = Stream(args.port or find_port())
    time.sleep(0.3)
    st.send("stream on")
    d = OUT / time.strftime("cal_%Y%m%d_%H%M%S")
    d.mkdir(parents=True)
    mj = open(d / "frames.mjpeg", "wb")
    fcsv = open(d / "frames.csv", "w")
    fcsv.write("seq,dev_ms,pc_time,offset,len\n")
    rows = []   # labels, written at the end: rewinding marks earlier rows invalid

    k, t0, paused, pause_t = 0, time.time(), True, time.time()   # start paused: Space to begin
    n = 0
    try:
        for head, data in st.frames():
            if head is None:
                continue
            seq, w, h, ln, ms = head
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            now = time.time()
            if paused:
                t0 += now - pause_t
                pause_t = now
            pid, text, target = items[k]
            el = now - t0
            if target is None:
                phase, f, dur = "free", el / FREE_SECONDS, FREE_SECONDS
            else:
                acc, phase, f = 0.0, "out", 1.0
                for name, s in PHASES:
                    if el < acc + s:
                        phase, f = name, (el - acc) / s
                        break
                    acc += s
                dur = sum(s for _, s in PHASES)
            if el >= dur:
                k += 1
                t0 = now
                if k >= len(items):
                    break
                continue
            if not paused:
                off = mj.tell()
                mj.write(data)
                fcsv.write(f"{seq},{ms},{now:.4f},{off},{ln}\n")
                if target is None:
                    lab = ["nan"] * len(CHANNELS)
                    env = "nan"
                else:
                    e = envelope(phase, f)
                    lab = [f"{target.get(c, 0) * e:.3f}" for c in CHANNELS]
                    env = f"{e:.3f}"
                rows.append([seq, pid, phase, env, lab, k, True])
                n += 1
            view = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2), cv2.COLOR_GRAY2BGR)
            if paused:
                p = panel(view.shape[1], "ПАУЗА: пробел - продолжить, стрелки - выбрать эмоцию",
                          f"с: {text}   ({k + 1}/{len(items)})   (она и всё после неё перепишутся)",
                          0, (80, 200, 255))
            elif phase == "rest":
                p = panel(view.shape[1], "Нейтрально, отдохни", f"дальше: {text}   ({k + 1}/{len(items)})",
                          f, (150, 150, 150))
            else:
                col = {"in": (0, 200, 255), "hold": (0, 230, 0), "out": (0, 200, 255), "free": (255, 160, 60)}[phase]
                sub = {"in": "плавно входи", "hold": "держи", "out": "плавно отпускай", "free": "свободно"}[phase]
                p = panel(view.shape[1], text, f"{sub}   ({k + 1}/{len(items)})", f, col)
            cv2.imshow("mouth calibration", np.vstack([view, p]))
            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF if key_ex >= 0 else -1
            if key in (ord("q"), 27):
                break
            if paused and key_ex in (KEY_LEFT, KEY_RIGHT):
                k = max(0, k - 1) if key_ex == KEY_LEFT else min(len(items) - 1, k + 1)
            if key == ord(" "):
                if paused:
                    # resume = redo the selected prompt from its rest phase; frames recorded for it
                    # and everything after it (a mistake, or a rewind) no longer count
                    bad = 0
                    for r in rows:
                        if r[6] and r[5] >= k:
                            r[6] = False
                            bad += 1
                    if bad:
                        print(f"marked {bad} frames invalid, redo from prompt {k + 1}", flush=True)
                    t0 = now
                paused = not paused
                pause_t = now
            if key == ord("n"):
                k += 1
                t0 = now
                if k >= len(items):
                    break
    finally:
        mj.close(); fcsv.close()
        with open(d / "labels.csv", "w", encoding="utf-8") as lcsv:
            lcsv.write("seq,prompt,phase,envelope,valid," + ",".join(CHANNELS) + "\n")
            for seq, pid, phase, env, lab, _, ok in rows:
                lcsv.write(f"{seq},{pid},{phase},{env},{int(ok)}," + ",".join(lab) + "\n")
        cv2.destroyAllWindows()
        good = sum(r[6] for r in rows)
        print(f"saved {n} frames ({good} valid labels) to {d}", flush=True)


if __name__ == "__main__":
    main()
