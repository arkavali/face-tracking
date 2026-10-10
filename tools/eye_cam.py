"""PC helper for the eye camera firmware (firmware/src/main.cpp) over the USB serial port.

  python eye_cam.py COM9 snap [out.png]    save the current camera frame
  python eye_cam.py COM9 live [--scale 1]  live view with the tracker overlay;
                                           R = start/stop recording to out/cam/, Q/Esc = quit
  python eye_cam.py COM9 cmd "status"      send one raw command and print the reply

Recordings (out/cam/eye_<time>.npz) hold the preview frames, the EYE lines and the time
of each, so tools/evaluate-style scripts and parity_test.py can replay them.
Opening the port with DTR/RTS low avoids resetting the board on connect.
Not run against hardware yet (boards arrive ~2026-10-20).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import serial

OUT = Path(__file__).resolve().parent.parent / "out" / "cam"


def open_port(port: str) -> serial.Serial:
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, 921600, 0.5
    s.dtr = False
    s.rts = False
    s.open()
    time.sleep(0.2)
    s.reset_input_buffer()
    return s


def command(s: serial.Serial, cmd: str, wait: float = 0.5) -> list:
    s.write((cmd + "\n").encode())
    t0, out = time.time(), []
    while time.time() - t0 < wait:
        ln = s.readline().decode(errors="replace").strip()
        if ln and not ln.startswith(("EYE", "IMG")):
            out.append(ln)
    return out


def snap(s: serial.Serial) -> np.ndarray:
    s.write(b"preview off\nstream off\n")
    time.sleep(0.3)
    s.reset_input_buffer()
    s.write(b"snap\n")
    t0 = time.time()
    while time.time() - t0 < 5:
        ln = s.readline().decode(errors="replace").strip()
        if ln.startswith("SNAP"):
            _, w, h = ln.split()
            w, h = int(w), int(h)
            data = s.read(w * h)
            if len(data) == w * h:
                return np.frombuffer(data, np.uint8).reshape(h, w)
    raise RuntimeError("no SNAP reply")


def parse_eye(ln: str) -> dict:
    return {k: float(v) for k, v in (p.split("=", 1) for p in ln.split()[1:] if "=" in p)}


class Stream:
    """Splits the serial byte stream into EYE lines and complete preview frames."""

    def __init__(self, s: serial.Serial):
        self.s = s
        self.frames = {}          # seq -> (w, h, bytearray, received bytes)

    def poll(self):
        """Yields ("eye", dict) and ("img", ndarray) events."""
        ln = self.s.readline()
        if not ln:
            return
        txt = ln.decode(errors="replace").strip()
        if txt.startswith("EYE"):
            yield "eye", parse_eye(txt)
        elif txt.startswith("IMG"):
            try:
                _, seq, w, h, off, n = txt.split()
                seq, w, h, off, n = map(int, (seq, w, h, off, n))
            except ValueError:
                return
            data = self.s.read(n)
            if seq not in self.frames:
                self.frames = {seq: (w, h, bytearray(w * h), 0)}
            fw, fh, buf, got = self.frames[seq]
            buf[off: off + len(data)] = data
            got += len(data)
            self.frames[seq] = (fw, fh, buf, got)
            if got >= fw * fh:
                del self.frames[seq]
                yield "img", np.frombuffer(bytes(buf), np.uint8).reshape(fh, fw)
        elif txt:
            print(txt)


def live(s: serial.Serial, scale: int) -> None:
    command(s, f"preview {scale}")
    command(s, "stream on")
    st = Stream(s)
    last_eye, rec = {}, None
    frame = None
    try:
        while True:
            for kind, val in st.poll():
                if kind == "eye":
                    last_eye = val
                    if rec is not None:
                        rec["eye"].append(val)
                        rec["t_eye"].append(time.time())
                else:
                    frame = val
                    if rec is not None:
                        rec["img"].append(val)
                        rec["t_img"].append(time.time())
            if frame is None:
                continue
            view = cv2.cvtColor(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            z = 640 / 320
            e = last_eye
            if e.get("p") == 1:
                cv2.ellipse(view, ((e["cx"] * z, e["cy"] * z), (4 * e["a"], 4 * e["b"]), np.degrees(e["ang"])),
                            (0, 0, 255), 1)
            if "ex" in e:
                cv2.drawMarker(view, (int(e["ex"] * z), int(e["ey"] * z)), (0, 255, 0), cv2.MARKER_CROSS, 12)
            for g in ("g1", "g2"):
                if g + "x" in e:
                    cv2.circle(view, (int(e[g + "x"] * z), int(e[g + "y"] * z)), 4, (255, 255, 0), 1)
            info = f"{'REC ' if rec is not None else ''}ms={e.get('ms', 0):.1f} p={int(e.get('p', 0))}"
            cv2.putText(view, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("eye_cam (R record, Q quit)", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if rec is None:
                    rec = {"img": [], "t_img": [], "eye": [], "t_eye": []}
                    print("recording...")
                else:
                    save(rec)
                    rec = None
    finally:
        if rec is not None:
            save(rec)
        command(s, "preview off")
        cv2.destroyAllWindows()


def save(rec: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / time.strftime("eye_%Y%m%d_%H%M%S.npz")
    np.savez_compressed(out, imgs=np.array(rec["img"]), t_img=np.array(rec["t_img"]),
                        eye=json.dumps(rec["eye"]), t_eye=np.array(rec["t_eye"]))
    print(f"saved {out} ({len(rec['img'])} frames, {len(rec['eye'])} EYE lines)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("action", choices=["snap", "live", "cmd"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--scale", type=int, default=1, choices=[1, 2, 4], help="preview downscale")
    args = ap.parse_args()
    s = open_port(args.port)
    if args.action == "snap":
        img = snap(s)
        out = args.arg or "eye_snap.png"
        cv2.imwrite(out, img)
        print(out, img.shape)
    elif args.action == "live":
        live(s, args.scale)
    else:
        for ln in command(s, args.arg or "status", wait=1.0):
            print(ln)


if __name__ == "__main__":
    main()
