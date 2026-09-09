"""
yolo_vision_server.py — WebSocket semantic-inference server for the dashboard.
==============================================================================
The browser dashboard sends raw webcam pixels (RGBA) over a WebSocket; this
server runs the YOLO semantic shape detector on them and replies with every
detected object (person / car / truck / bus / motorcycle / bicycle), its stable
tracking ID, confidence, wheel/headlight part markers and range band.

Because detection is per-frame (single-image), a STILL photo or video held in
front of the camera IS detected — the temporal-difference is zero, but the
shape is recognised from the pixels of one frame. This is the fix for the
"display image is static" problem.

Protocol
--------
browser -> server (binary):  <u32 BE width><u32 BE height><w*h*4 RGBA bytes>
browser -> server (text):    {"cmd": "reset" | "ping"}
server  -> browser (text):   {"type":"detections","backend":...,"fps":...,
                               "infer_ms":...,"objects":[{id,label,conf,
                               box,parts,range_band}, ...]}

Usage:
    python yolo_vision_server.py --port 8090 --backend cv-cascade
    python yolo_vision_server.py --selftest          # offline sanity check
"""

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from semantic_detector import (SemanticDetector, VEHICLE_LABELS)  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

STATUS_PATH = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "web", "ui", "yolo_status.json"))


def _write_status(payload):
    """Write the /yolo-status.json file served by the Node bridge."""
    try:
        os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:  # pragma: no cover
        print(f"[yolo][warn] status write failed: {exc}", file=sys.stderr)


class VisionServer:
    """WebSocket endpoint wrapping SemanticDetector + multi-object tracking."""

    def __init__(self, backend="auto", model_path=None, port=8090,
                 conf_threshold=0.30, max_clients=4):
        self.port = port
        self.conf_threshold = conf_threshold
        self.detector = SemanticDetector(backend=backend, model_path=model_path,
                                         conf_threshold=conf_threshold)
        self._last_frame_at = time.time()
        self._fps_ema = 0.0
        self._frame_no = 0
        self.max_clients = max_clients

    # -- frame decode --------------------------------------------------------
    @staticmethod
    def decode_frame(payload):
        if len(payload) < 8:
            return None
        w = int.from_bytes(payload[0:4], "big")
        h = int.from_bytes(payload[4:8], "big")
        need = w * h * 4
        if w < 16 or h < 16 or len(payload) < 8 + need:
            return None
        rgba = np.frombuffer(payload[8:8 + need], dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    # -- message handling ----------------------------------------------------
    async def handle(self, ws):
        while True:
            try:
                msg = await ws.recv()
            except Exception:
                return
            if isinstance(msg, bytes):
                frame = self.decode_frame(msg)
                if frame is None:
                    reply = {"type": "error", "reason": "bad-frame"}
                else:
                    reply = self.process_frame(frame)
                try:
                    await ws.send(json.dumps(reply))
                except Exception:
                    return
            else:
                try:
                    cmd = json.loads(msg)
                except Exception:
                    cmd = {}
                if cmd.get("cmd") == "reset":
                    self.detector.tracker.reset()
                    await ws.send(json.dumps({"type": "ok", "cmd": "reset"}))
                elif cmd.get("cmd") == "ping":
                    await ws.send(json.dumps({
                        "type": "pong", "backend": self.detector.backend_name}))
                elif cmd.get("cmd") == "set_conf":
                    self.conf_threshold = float(cmd.get("value", 0.30))

    def process_frame(self, frame_bgr):
        t0 = time.time()
        objs = self.detector.detect_and_track(frame_bgr)
        infer_ms = (time.time() - t0) * 1e3
        self._frame_no += 1
        now = time.time()
        inst = 1.0 / max(1e-3, now - self._last_frame_at)
        self._fps_ema = self._fps_ema * 0.9 + inst * 0.1 if self._fps_ema else inst
        self._last_frame_at = now
        return {
            "type": "detections",
            "backend": self.detector.backend_name,
            "frame": self._frame_no,
            "fps": round(self._fps_ema, 1),
            "infer_ms": round(infer_ms, 1),
            "objects": objs,
        }

    async def run(self):
        if websockets is None:
            print("[ERROR] the 'websockets' package is required. "
                  "pip install websockets", file=sys.stderr)
            sys.exit(1)

        async def handler(ws):
            try:
                await self.handle(ws)
            except Exception:
                pass

        async with websockets.serve(handler, "127.0.0.1", self.port,
                                    max_queue=16, ping_interval=20,
                                    ping_timeout=20, close_timeout=10):
            print("=" * 60)
            print(f"  YOLO Semantic Vision Server")
            print(f"  ws://127.0.0.1:{self.port}   backend={self.detector.backend_name}")
            print("  Waiting for webcam frames from the dashboard...")
            print("=" * 60)
            _write_status({
                "ok": True, "port": self.port,
                "backend": self.detector.backend_name,
                "endpoint": f"ws://127.0.0.1:{self.port}",
            })
            await asyncio.Future()  # run forever


def selftest(backend="cv-cascade"):
    """Offline sanity check: synthetic persons + car with a wheel/headlight."""
    if cv2 is None:
        print("[ERROR] opencv-python required for selftest", file=sys.stderr)
        return 1
    w, h = 640, 360
    frame = np.full((h, w, 3), 128, dtype=np.uint8)  # neutral road-ish gray
    # "Car": luma<100 body + two DARK circular wheels + two BRIGHT headlights,
    # so the pixel-shape analysis has cleanly separated part shapes.
    body = frame[170:270, 220:460]
    body[:] = (78, 78, 88)
    cv2.circle(frame, (250, 265), 22, (25, 25, 25), -1)   # wheel 1
    cv2.circle(frame, (410, 265), 22, (25, 25, 25), -1)   # wheel 2
    cv2.circle(frame, (310, 212), 10, (245, 245, 245), -1)  # headlight 1
    cv2.circle(frame, (345, 212), 10, (245, 245, 245), -1)  # headlight 2
    # Two "persons": skin-tone oval (face) above a dark torso.
    for cx in (140, 520):
        cv2.ellipse(frame, (cx, 232), (30, 56), 0, 0, 360,
                    (60, 60, 70), -1)                   # torso/clothing
        cv2.ellipse(frame, (cx, 196), (14, 16), 0, 0, 360,
                    (90, 150, 220), -1)                 # skin tones (BGR)
    det = SemanticDetector(backend=backend)
    objs = det.detect_and_track(frame)
    print(json.dumps({"backend": det.backend_name, "objects": objs}, indent=2))
    if not objs:
        print("[selftest] no objects found on synthetic frame (unexpected)")
        return 2
    persons = [o for o in objs if o["label"] == "person"]
    vehicles = [o for o in objs if o["label"] in VEHICLE_LABELS]
    parts = [p for o in vehicles for p in o.get("parts", [])]
    print(f"[selftest] persons={len(persons)} vehicles={len(vehicles)} "
          f"parts(found)={len(parts)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="YOLO semantic vision server")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "yolov8", "yolov5", "cv-cascade"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(backend=args.backend))
    server = VisionServer(backend=args.backend, model_path=args.weights,
                          port=args.port, conf_threshold=args.conf)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[yolo] server stopped.")


if __name__ == "__main__":
    main()