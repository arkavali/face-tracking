"""Brow camera UI: serves the web page and bridges it to the ESP32-CAM's serial console
(firmware/src/main.cpp).

  python tools/ui/browui.py [PORT] [--http 8766] [--no-browser]

PORT is detected automatically (CH340 / CP210x / FTDI / ESP32 USB) when omitted. The
bridge keeps the port open and reconnects when the board is unplugged or reset. Opening
the port does not reset the board.

HTTP API (used by the page, handy for scripts too):
  GET  /events   server-sent events (JSON): {"t":"line"}, {"t":"brow"}, {"t":"img"},
                 {"t":"status"}
  POST /cmd      body: command lines, sent to the board as they are
  POST /port     body: "close" releases the port (to flash firmware), "open" takes it again
  POST /record   body: "start" | "stop"; recordings go to out/cam/rec_<time>/
                 (brow.csv with every BROW line, frames/NNNNN.png with every preview frame)
  POST /snap     save the next full-resolution frame to out/cam/snap_<time>.png
  GET  /status   {"port": ..., "open": ..., "recording": ...}
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import serial
import serial.tools.list_ports

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "out" / "cam"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
KNOWN_VIDS = {0x1A86, 0x10C4, 0x0403, 0x303A}  # CH340, CP210x, FTDI, Espressif
BAUD = 921600
BROW_KEYS = ["rL", "iL", "oL", "vL", "nL", "rR", "iR", "oR", "vR", "nR"]


def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid in KNOWN_VIDS:
            return p.device
    return ports[0].device if len(ports) == 1 else None


def parse_brow(line: str) -> dict:
    kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
    out = {}
    for k, v in kv.items():
        if k in ("sL", "sR"):
            out[k] = [float(x) for x in v.split(",") if x]
        elif k in ("seq", "vL", "vR"):
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


class Hub:
    """Fan-out of events to the open pages (one queue per page)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clients: list[queue.Queue] = []

    def add(self) -> queue.Queue:
        q = queue.Queue(maxsize=400)
        with self.lock:
            self.clients.append(q)
        return q

    def remove(self, q) -> int:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)
            return len(self.clients)

    def publish(self, ev: dict, droppable: bool = False) -> None:
        data = json.dumps(ev, separators=(",", ":"))
        with self.lock:
            for q in self.clients:
                if droppable and q.qsize() > 50:
                    continue  # a slow page skips images / values rather than lagging behind
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.dir: Path | None = None
        self.csv = None
        self.writer = None
        self.frames = 0
        self.t0 = 0.0

    def start(self) -> Path:
        with self.lock:
            self._close()
            self.dir = OUT / time.strftime("rec_%Y%m%d_%H%M%S")
            (self.dir / "frames").mkdir(parents=True, exist_ok=True)
            self.csv = open(self.dir / "brow.csv", "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.csv)
            self.writer.writerow(["t", "seq", "ms"] + BROW_KEYS + ["sL", "sR"])
            self.frames = 0
            self.t0 = time.time()
            (self.dir / "frames.csv").write_text("frame,t,seq\n", encoding="utf-8")
            return self.dir

    def _close(self):
        if self.csv:
            self.csv.close()
        self.csv = self.writer = None

    def stop(self) -> Path | None:
        with self.lock:
            self._close()
            d, self.dir = self.dir, None
            return d

    def brow(self, v: dict) -> None:
        with self.lock:
            if not self.writer:
                return
            row = [f"{time.time() - self.t0:.3f}", v.get("seq", ""), v.get("ms", "")]
            row += [v.get(k, "") for k in BROW_KEYS]
            row += [" ".join(f"{x:.1f}" for x in v.get(k, [])) for k in ("sL", "sR")]
            self.writer.writerow(row)

    def frame(self, img: np.ndarray, seq: int) -> None:
        with self.lock:
            if not self.dir:
                return
            cv2.imwrite(str(self.dir / "frames" / f"{self.frames:05d}.png"), img)
            with open(self.dir / "frames.csv", "a", encoding="utf-8") as f:
                f.write(f"{self.frames},{time.time() - self.t0:.3f},{seq}\n")
            self.frames += 1

    @property
    def active(self) -> bool:
        return self.dir is not None


class Bridge:
    """Owns the serial port: reads lines and binary frames, writes commands."""

    def __init__(self, port: str | None, hub: Hub, rec: Recorder):
        self.want_port = port
        self.port = None
        self.ser: serial.Serial | None = None
        self.hub, self.rec = hub, rec
        self.enabled = True
        self.wlock = threading.Lock()
        self.save_snap = False
        self.img = None  # (seq, w, h, bytearray, received)
        threading.Thread(target=self._run, daemon=True).start()

    # ---- port handling
    def status(self) -> dict:
        return {"t": "status", "port": self.port, "open": self.ser is not None, "recording": self.rec.active,
                "rec_dir": str(self.rec.dir) if self.rec.dir else None}

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self._drop()

    def _drop(self, only=None):
        """Close the port (only if it is still `only`, when given: a reader that failed on an
        old, already replaced port must not close the new one)."""
        if only is not None and self.ser is not only:
            return
        s, self.ser = self.ser, None
        if s:
            try:
                s.close()
            except Exception:
                pass
            self.hub.publish(self.status())

    def _open(self):
        port = self.want_port or find_port()
        if not port:
            return
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, BAUD, 0.05
        s.dtr = False  # DTR/RTS low: opening the port must not reset or bootstrap the ESP32
        s.rts = False
        try:
            s.open()
        except serial.SerialException:
            return
        self.port, self.ser = port, s
        self.hub.publish(self.status())
        self.hub.publish({"t": "line", "s": f"# connected to {port}"})

    def send(self, text: str) -> bool:
        s = self.ser
        if not s:
            return False
        data = "".join(ln.strip() + "\n" for ln in text.splitlines() if ln.strip())
        if not data:
            return True
        # leading newline: ends whatever partial line the board may hold (noise around a
        # reset), which would otherwise be glued to the first command and reject it
        data = "\n" + data
        try:
            with self.wlock:
                s.write(data.encode())
        except (serial.SerialException, OSError, TypeError, AttributeError):
            self._drop(only=s)
            return False
        for ln in data.splitlines():
            if ln:
                self.hub.publish({"t": "line", "s": "> " + ln})
        return True

    # ---- reader
    def _read_exact(self, s, n: int) -> bytes | None:
        buf = bytearray()
        t0 = time.time()
        while len(buf) < n:
            chunk = s.read(n - len(buf))
            if chunk:
                buf += chunk
                t0 = time.time()
            elif time.time() - t0 > 2.0:
                return None
        return bytes(buf)

    def _run(self):
        while True:
            if not self.enabled:
                time.sleep(0.2)
                continue
            if not self.ser:
                self._open()
                if not self.ser:
                    time.sleep(1.0)
                    continue
            s = self.ser
            try:
                raw = s.readline()
                if raw:
                    self._handle(s, raw)
            except (serial.SerialException, OSError, TypeError, AttributeError):
                # port unplugged, or closed under us by /port close (pyserial then fails
                # with TypeError / AttributeError on Windows)
                self._drop(only=s)
                time.sleep(0.5)
            except Exception as e:  # a garbled frame header etc. must not kill the reader
                self.hub.publish({"t": "line", "s": f"# bridge: {type(e).__name__}: {e}"})

    def _handle(self, s, raw: bytes):
        line = raw.decode("ascii", errors="replace").strip()
        if any(c < " " or c > "~" for c in line):
            return  # binary: boot ROM noise at 115200 baud, or joined in the middle of a frame
        if line.startswith("IMG "):
            parts = line.split()
            if len(parts) != 6:
                return
            seq, w, h, off, n = (int(x) for x in parts[1:])
            data = self._read_exact(s, n)
            if data is None:
                return
            if not self.img or self.img[0] != seq or self.img[1] != w or self.img[2] != h:
                self.img = [seq, w, h, bytearray(w * h), 0]
            self.img[3][off:off + n] = data
            self.img[4] += n
            if self.img[4] >= w * h:
                frame = bytes(self.img[3])
                self.img = None
                self.hub.publish({"t": "img", "seq": seq, "w": w, "h": h,
                                  "data": base64.b64encode(frame).decode()}, droppable=True)
                if self.rec.active:
                    self.rec.frame(np.frombuffer(frame, np.uint8).reshape(h, w), seq)
            return
        if line.startswith("SNAP "):
            _, w, h = line.split()
            w, h = int(w), int(h)
            data = self._read_exact(s, w * h)
            if data is None:
                return
            img = np.frombuffer(data, np.uint8).reshape(h, w)
            msg = f"# snap {w}x{h}"
            if self.save_snap:
                self.save_snap = False
                OUT.mkdir(parents=True, exist_ok=True)
                path = OUT / time.strftime("snap_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(path), img)
                msg += f" saved to {path}"
            self.hub.publish({"t": "img", "seq": -1, "w": w, "h": h, "data": base64.b64encode(data).decode()})
            self.hub.publish({"t": "line", "s": msg})
            return
        if line.startswith("BROW "):
            v = parse_brow(line)
            self.rec.brow(v)
            v["t"] = "brow"
            self.hub.publish(v, droppable=True)
            return
        if line:
            self.hub.publish({"t": "line", "s": line})


def make_handler(bridge: Bridge, hub: Hub, rec: Recorder):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in STATIC:
                name, ctype = STATIC[path]
                self._reply(200, (HERE / name).read_bytes(), ctype)
            elif path == "/status":
                self._reply(200, json.dumps(bridge.status()).encode(), "application/json")
            elif path == "/events":
                self._events()
            else:
                self._reply(404, b"not found")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8", errors="replace") if n else ""
            path = self.path.split("?")[0]
            if path == "/cmd":
                ok = bridge.send(body)
                self._reply(200 if ok else 503, b"ok" if ok else b"port not open")
            elif path == "/port":
                bridge.set_enabled(body.strip() != "close")
                hub.publish(bridge.status())
                self._reply(200, b"ok")
            elif path == "/record":
                if body.strip() == "start":
                    d = rec.start()
                    hub.publish({"t": "line", "s": f"# recording to {d}"})
                else:
                    d = rec.stop()
                    if d:
                        hub.publish({"t": "line", "s": f"# recording saved: {d}"})
                hub.publish(bridge.status())
                self._reply(200, b"ok")
            elif path == "/snap":
                bridge.save_snap = True
                ok = bridge.send("snap")
                self._reply(200 if ok else 503, b"ok" if ok else b"port not open")
            else:
                self._reply(404, b"not found")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            q = hub.add()
            try:
                self.wfile.write(f"data: {json.dumps(bridge.status())}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        data = q.get(timeout=10)
                    except queue.Empty:
                        data = None
                    self.wfile.write((f"data: {data}\n\n" if data else ": ping\n\n").encode())
                    self.wfile.flush()
            except (ConnectionError, OSError):
                pass
            finally:
                if hub.remove(q) == 0:
                    # last page closed: stop spending the serial link on images
                    bridge.send("preview off")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?")
    ap.add_argument("--http", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    hub, rec = Hub(), Recorder()
    bridge = Bridge(args.port, hub, rec)
    srv = ThreadingHTTPServer(("127.0.0.1", args.http), make_handler(bridge, hub, rec))
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{args.http}/"
    print(f"Brow UI at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rec.stop()
        bridge.send("preview off")
        sys.exit(0)


if __name__ == "__main__":
    main()
