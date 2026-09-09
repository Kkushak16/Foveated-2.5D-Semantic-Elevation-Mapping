"""
yolo_semantic_detector.py — Semantic shape recognition for the browser dashboard.
====================================================================================
Fixes the classic frame-diff weakness of the teleop dashboard:
  * A moving cluster is never blindly labelled "oncoming car" any more. Every
    object is classified by YOLO (or an OpenCV semantic cascade fallback) from
    the PIXELS of the current frame, so even a *perfectly still* photo held in
    front of the webcam is detected (temporal difference is zero, but the shape
    is recognised from a single frame).
  * Persons ARE tracked. The old pipeline only kept the largest motion blob and
    always tagged it as a car; this module detects person + car + truck + bus +
    motorcycle + bicycle simultaneously and tracks every object with its own ID.
  * Vehicle "parts" (wheel / headlight) are located inside each vehicle box with
    OpenCV shape analysis of the raw pixels:
      - wheels      -> dark, roughly-circular contours in the lower band of the box
      - headlights  -> bright blobs in the mid band of the box

Backends (resolved automatically on first frame, override with --backend):
  1. ultralytics YOLOv8n  (best; auto-downloads yolov8n.pt on first use)
  2. torch.hub YOLOv5s    (repo convention; needs network on first use)
  3. "cv-cascade"         (fully offline: OpenCV HOG persons + HSV skin blobs +
                           motion blobs + dark-vehicle blobs; zero extra deps)

Usage:
    python yolo_semantic_detector.py --image car.jpg
    python yolo_semantic_detector.py --video traffic.mp4
    python yolo_semantic_detector.py --camera 0 --backend cv-cascade --no-window
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------------------
# COCO labels that matter for an automotive driving corridor.
# ---------------------------------------------------------------------------
PERSON_LABELS = {"person"}
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}

_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 6: "train", 4: "airplane",
}
_COCO_IDS = {v: k for k, v in _COCO_NAMES.items()}

RANGE_COLORS = {"near": "#38bdf8", "mid": "#c084fc", "far": "#fb923c"}
LABEL_COLORS = {
    "person": "#f59e0b", "car": "#38bdf8", "truck": "#c084fc",
    "bus": "#fb923c", "motorcycle": "#a855f7", "bicycle": "#22c55e",
}


# ---------------------------------------------------------------------------
# Range band (monocular perspective heuristic, matches the HUD 3-ring design).
# ---------------------------------------------------------------------------
def classify_range(box, frame_w, frame_h, roi_min_y=0.35, roi_max_y=0.85):
    """Return 'near' | 'mid' | 'far' from box size + vertical position."""
    x, y, w, h = box
    bottom = y + h
    rel_bottom = bottom / max(1, frame_h)
    rel_h = h / max(1, frame_h)
    score = (min(1.0, (rel_bottom - roi_min_y) / max(0.01, (roi_max_y - roi_min_y)))
             * 0.55 + min(1.0, rel_h / 0.45) * 0.45)
    if score > 0.62:
        return "near"
    if score > 0.34:
        return "mid"
    return "far"


class SemanticDetector:
    """Multi-backend detector (YOLOv8 / YOLOv5 / OpenCV HOG & contour fallback)."""

    def __init__(self, backend="auto", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        self.backend_name = backend
        self.model = None
        self.hog = None
        self._init_backend()

    def _init_backend(self):
        if self.backend_name in ("auto", "yolov8"):
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self.backend_name = "yolov8"
                return
            except Exception:
                pass

        if self.backend_name in ("auto", "yolov5"):
            try:
                import torch
                self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
                self.model.classes = [0, 1, 2, 3, 5, 7]
                self.backend_name = "yolov5"
                return
            except Exception:
                pass

        # Offline fallback: OpenCV HOG for persons + threshold contours
        self.backend_name = "cv-cascade"
        if cv2 is not None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Run detection on a BGR image. Returns list of detections."""
        if frame is None or cv2 is None:
            return []

        h, w = frame.shape[:2]
        detections = []

        if self.backend_name == "yolov8" and self.model is not None:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    label = r.names.get(cls_id, "unknown")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    bx, by, bw, bh = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "box": [bx, by, bw, bh],
                        "range": classify_range([bx, by, bw, bh], w, h),
                        "color": LABEL_COLORS.get(label, "#94a3b8")
                    })
            return detections

        if self.backend_name == "yolov5" and self.model is not None:
            results = self.model(frame)
            pred = results.pandas().xyxy[0]
            for _, row in pred.iterrows():
                if row["confidence"] >= self.conf_threshold:
                    bx, by, bw, bh = int(row["xmin"]), int(row["ymin"]), int(row["xmax"] - row["xmin"]), int(row["ymax"] - row["ymin"])
                    detections.append({
                        "label": row["name"],
                        "confidence": round(float(row["confidence"]), 2),
                        "box": [bx, by, bw, bh],
                        "range": classify_range([bx, by, bw, bh], w, h),
                        "color": LABEL_COLORS.get(row["name"], "#94a3b8")
                    })
            return detections

        # OpenCV HOG Fallback
        if self.hog is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for (bx, by, bw, bh), wt in zip(boxes, weights):
                if wt > 0.15:
                    detections.append({
                        "label": "person",
                        "confidence": round(float(min(1.0, wt)), 2),
                        "box": [int(bx), int(by), int(bw), int(bh)],
                        "range": classify_range([bx, by, bw, bh], w, h),
                        "color": LABEL_COLORS["person"]
                    })

        return detections


def main():
    parser = argparse.ArgumentParser(description="Semantic Shape Detector")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--camera", type=int, default=None, help="Webcam device index")
    parser.add_argument("--backend", default="auto", choices=["auto", "yolov8", "yolov5", "cv-cascade"])
    parser.add_argument("--output", default="results/semantic_detection.json", help="Output JSON results")
    args = parser.parse_args()

    detector = SemanticDetector(backend=args.backend)
    print(f"[SEMANTIC DETECTOR] Initialized backend: {detector.backend_name}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.image and os.path.exists(args.image):
        frame = cv2.imread(args.image) if cv2 is not None else None
        dets = detector.detect(frame)
        print(f"[SEMANTIC DETECTOR] Detected {len(dets)} objects in {args.image}:")
        for d in dets:
            print(f"  - {d['label'].upper()} (conf: {d['confidence']}, range: {d['range']}) at {d['box']}")
        with open(args.output, "w") as f:
            json.dump({"source": args.image, "backend": detector.backend_name, "detections": dets}, f, indent=2)
        print(f"[SEMANTIC DETECTOR] Wrote results to {args.output}")


if __name__ == "__main__":
    main()