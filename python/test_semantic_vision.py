"""
test_semantic_vision.py — Offline tests for the semantic vision stack.
========================================================================
Validates (no network, no ultralytics):
  1. detect_vehicle_parts(): pixel-shape analysis finds wheels + headlights.
  2. MultiObjectTracker(): stable per-object IDs across frames.
  3. End-to-end WebSocket: a binary RGBA frame sent by a "browser" client is
     decoded, detected (cv-cascade backend) and answered with JSON objects.

Run:  python test_semantic_vision.py
"""

import asyncio
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from semantic_detector import (  # noqa: E402
    MultiObjectTracker, SemanticDetector, detect_vehicle_parts,
)
from yolo_vision_server import VisionServer, selftest  # noqa: E402


def _synthetic_car():
    """640x360 frame: 1 car (wheels + headlights) + 2 skin-tone persons."""
    w, h = 640, 360
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    body = frame[170:270, 220:460]
    body[:] = (78, 78, 88)
    cv2.circle(frame, (250, 265), 22, (25, 25, 25), -1)
    cv2.circle(frame, (410, 265), 22, (25, 25, 25), -1)
    cv2.circle(frame, (310, 212), 10, (245, 245, 245), -1)
    cv2.circle(frame, (345, 212), 10, (245, 245, 245), -1)
    for cx in (140, 520):
        cv2.ellipse(frame, (cx, 232), (30, 56), 0, 0, 360, (60, 60, 70), -1)
        cv2.ellipse(frame, (cx, 196), (14, 16), 0, 0, 360, (90, 150, 220), -1)
    return frame


def test_parts():
    car = _synthetic_car()[170:288, 220:460]
    parts = detect_vehicle_parts(car)
    kinds = [p["kind"] for p in parts]
    assert "wheel" in kinds, f"expected wheel, got {kinds}"
    assert "headlight" in kinds, f"expected headlight, got {kinds}"
    print(f"[test] parts ok: {kinds}")


def test_tracker_ids():
    trk = MultiObjectTracker()
    f1 = _synthetic_car()
    dets = SemanticDetector(backend="cv-cascade").detect(f1)
    r1 = trk.update(dets)
    # Same frame -> same stable ids.
    r2 = trk.update(dets)
    ids1 = sorted(o["id"] for o in r1)
    ids2 = sorted(o["id"] for o in r2)
    assert ids1 == ids2, f"ids changed between identical frames: {ids1} vs {ids2}"
    persons = [o for o in r2 if o["label"] == "person"]
    assert len(persons) >= 2, f"expected >=2 persons, got {len(persons)}"
    print(f"[test] tracker ids stable: {ids1}; persons tracked: {len(persons)}")


async def test_websocket():
    import websockets
    server = VisionServer(backend="cv-cascade", port=0, conf_threshold=0.30)
    seen = {}

    async def handler(ws):
        try:
            await server.handle(ws)
        except Exception:
            pass

    async with websockets.serve(handler, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            frame = _synthetic_car()
            h, w = frame.shape[:2]
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            payload = (w.to_bytes(4, "big") + h.to_bytes(4, "big")
                       + rgba.tobytes())
            await ws.send(payload)
            reply = json.loads(await ws.recv())
            assert reply["type"] == "detections", reply
            assert reply["objects"], "expected at least one object"
            labels = {o["label"] for o in reply["objects"]}
            assert "person" in labels, f"expected person, got {labels}"
            seen["ok"] = True
            print(f"[test] websocket round-trip ok: {len(reply['objects'])} "
                  f"object(s), backend={reply['backend']}")
    assert seen.get("ok")


def main():
    rc = selftest(backend="cv-cascade")
    assert rc == 0, f"selftest failed rc={rc}"
    test_parts()
    test_tracker_ids()
    asyncio.run(test_websocket())
    print("\nALL SEMANTIC VISION TESTS PASSED")


if __name__ == "__main__":
    main()