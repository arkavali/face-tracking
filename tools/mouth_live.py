"""Live demo: camera picture + the trained model's values as bars (out/model/mouth.onnx).

  python mouth_live.py [COMx] [--smooth 0.5] [--babble] [--board]
--babble adds the Babble face model's matching channels as thin bars for comparison.
--board runs the net ON THE ESP32 (firmware "mode track"): bars show the board's values, the
picture is the 80x60 net input the board sends back.
Keys: L cycle the flash LED (starts at --led, default dim), Q/Esc quit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from mouth_cam import Stream, find_port

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "out" / "model"
# own channel -> Babble channel(s) averaged, for the comparison bars
BABBLE_MAP = {
    "jawOpen": ["jawOpen"], "smileL": ["mouthSmileLeft"], "smileR": ["mouthSmileRight"],
    "frownL": ["mouthFrownLeft"], "frownR": ["mouthFrownRight"], "funnel": ["mouthFunnel"],
    "pucker": ["mouthPucker"], "mouthLeft": ["mouthLeft"], "mouthRight": ["mouthRight"],
    "press": ["mouthPressLeft", "mouthPressRight"], "cheekPuffL": ["cheekPuffLeft"],
    "cheekPuffR": ["cheekPuffRight"], "jawLeft": ["jawLeft"], "jawRight": ["jawRight"],
    "tongueOut": ["tongueOut"], "tongueLeft": ["tongueLeft"], "tongueRight": ["tongueRight"],
    "tongueUp": ["tongueUp"], "tongueDown": ["tongueDown"],
}


def draw_bars(names, val, height, by=None, bidx=None):
    bars = np.full((height, 330, 3), 25, np.uint8)
    rh = height // len(names)
    for k, nm in enumerate(names):
        yy = k * rh
        v = float(val[k])
        col = (0, 220, 0) if v > 0.5 else (0, 170, 230) if v > 0.2 else (120, 120, 120)
        cv2.rectangle(bars, (110, yy + 3), (110 + int(200 * v), yy + rh - 4), col, -1)
        if by is not None and nm in bidx:
            bv = float(np.mean(by[bidx[nm]]))
            cv2.rectangle(bars, (110, yy + rh - 6), (110 + int(200 * bv), yy + rh - 3), (255, 120, 60), -1)
        cv2.putText(bars, nm, (4, yy + rh - 6), 0, 0.42, (220, 220, 220), 1)
    return bars


def run_board(st, names, args):
    """Values from the board's MOUTH lines, picture from its PRV previews."""
    st.send("mode track")
    led = sorted({0, 20, 80, args.led})
    led_i = led.index(args.led)
    st.send(f"led {args.led}")
    val = np.zeros(len(names), np.float32)
    view = np.zeros((480, 640, 3), np.uint8)
    t_last, n_fps, fps, net_ms = time.time(), 0, 0.0, 0.0
    try:
        for head, data in st.frames():
            if head is not None:
                continue            # a JPEG still in flight from stream mode
            if isinstance(data, tuple):
                _, (seq, w, h, n), raw = data
                img = np.frombuffer(raw, np.uint8).reshape(h, w)
                view = cv2.cvtColor(cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            elif data.startswith("MOUTH"):
                parts = data.split()
                if len(parts) != 4 + len(names):
                    continue
                net_ms = float(parts[3])
                y = np.array([int(v) for v in parts[4:]], np.float32) / 1000
                val = args.smooth * val + (1 - args.smooth) * y
                n_fps += 1
                if time.time() - t_last >= 1:
                    fps, n_fps, t_last = n_fps / (time.time() - t_last), 0, time.time()
            else:
                print(data)
                continue
            frame = view.copy()
            cv2.putText(frame, f"ON BOARD: {fps:.1f} fps, net {net_ms:.0f} ms, LED {led[led_i]}", (8, 22), 0, 0.6,
                        (0, 255, 0), 2)
            cv2.imshow("mouth live", np.hstack([frame, draw_bars(names, val, frame.shape[0])]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                led_i = (led_i + 1) % len(led)
                st.send(f"led {led[led_i]}")
    finally:
        st.send("mode stream")   # leave the board as the other tools expect it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA factor of the previous value (0 = raw)")
    ap.add_argument("--babble", action="store_true")
    ap.add_argument("--board", action="store_true", help="values computed on the ESP32")
    ap.add_argument("--led", type=int, default=20, help="flash LED level at start (0..255, L cycles)")
    args = ap.parse_args()
    meta = json.loads((MODEL / "mouth.json").read_text())
    names, iw, ih = meta["channels"], meta["in_w"], meta["in_h"]
    prov = ["CPUExecutionProvider"]   # ~1 ms on the CPU; CUDA EP would need cuDNN in PATH
    sess = ort.InferenceSession(str(MODEL / "mouth.onnx"), providers=prov)
    bab = None
    if args.babble:
        import teacher
        bab = ort.InferenceSession(str(teacher.MODEL), providers=prov)
        bidx = {k: [teacher.NAMES.index(n) for n in v] for k, v in BABBLE_MAP.items()}
    st = Stream(args.port or find_port())
    time.sleep(0.3)
    st.send("stream on")
    if args.board:
        run_board(st, names, args)
        return
    led = sorted({0, 20, 80, args.led})
    led_i = led.index(args.led)
    st.send(f"led {args.led}")
    val = np.zeros(len(names), np.float32)
    t_last, n_fps, fps, ms = time.time(), 0, 0.0, 0.0
    for head, data in st.frames():
        if head is None:
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        t0 = time.perf_counter()
        x = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32)[None, None] / 255
        y = sess.run(None, {"img": x})[0][0]
        ms = (time.perf_counter() - t0) * 1000
        val = args.smooth * val + (1 - args.smooth) * y
        by = None
        if bab is not None:
            b = bab.run(None, {bab.get_inputs()[0].name: (cv2.resize(img, (224, 224)).astype(np.float32) / 255)[None, None]})[0][0]
            by = np.clip(b, 0, 1)
        n_fps += 1
        if time.time() - t_last >= 1:
            fps, n_fps, t_last = n_fps / (time.time() - t_last), 0, time.time()
        view = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2), cv2.COLOR_GRAY2BGR)
        bars = np.full((view.shape[0], 330, 3), 25, np.uint8)
        rh = view.shape[0] // len(names)
        for k, nm in enumerate(names):
            yy = k * rh
            v = float(val[k])
            col = (0, 220, 0) if v > 0.5 else (0, 170, 230) if v > 0.2 else (120, 120, 120)
            cv2.rectangle(bars, (110, yy + 3), (110 + int(200 * v), yy + rh - 4), col, -1)
            if by is not None and nm in bidx:
                bv = float(np.mean(by[bidx[nm]]))
                cv2.rectangle(bars, (110, yy + rh - 6), (110 + int(200 * bv), yy + rh - 3), (255, 120, 60), -1)
            cv2.putText(bars, nm, (4, yy + rh - 6), 0, 0.42, (220, 220, 220), 1)
        cv2.putText(view, f"{fps:.1f} fps, model {ms:.1f} ms, LED {led[led_i]}", (8, 22), 0, 0.6, (0, 255, 0), 2)
        cv2.imshow("mouth live", np.hstack([view, bars]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("l"):
            led_i = (led_i + 1) % len(led)
            st.send(f"led {led[led_i]}")


if __name__ == "__main__":
    main()
