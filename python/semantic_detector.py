"""
semantic_detector.py — Semantic shape recognition for the browser dashboard.
=============================================================================
Fixes the classic frame-diff weakness of the teleop dashboard:

  * A moving cluster is NEVER blindly labelled "oncoming car". Every object is
    classified from the PIXELS of the current frame by YOLO (or an OpenCV
    semantic cascade fallback). Because detection is per-frame, even a
    completely STILL photo/video held in front of the webcam is still detected
    (temporal difference is zero, but the shape is recognised from one frame).
  * Persons ARE tracked. The old pipeline kept only the largest motion blob and
    hard-coded it as a car. This module detects person + car + truck + bus +
    motorcycle + bicycle simultaneously and gives EVERY object its own stable
    tracking ID (multi-object tracking via centroid + IoU matching).
  * Vehicle "parts" are located inside each vehicle box with OpenCV shape
    analysis of the raw pixels:
      - wheels      -> dark, roughly-circular contours in the lower band
      - headlights  -> bright blobs in the mid/front band

Backends (resolved automatically, override with --backend):
  1. ultralytics YOLOv8n   (best; auto-downloads yolov8n.pt on first use)
  2. torch.hub YOLOv5s     (repo convention; needs network on first use)
  3. "cv-cascade"          (fully offline: OpenCV HOG people + HSV skin blobs +
                            motion blobs + still dark-vehicle blobs; no extra deps)

Usage:
    python semantic_detector.py --image car.jpg
    python semantic_detector.py --video traffic.mp4
    python semantic_detector.py --camera 0 --backend cv-cascade --no-window
"""

import argparse
import json
import sys
import time
from collections import OrderedDict

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# ---------------------------------------------------------------------------
# Labels that matter in an automotive driving corridor.
# ---------------------------------------------------------------------------
PERSON_LABELS = {"person"}
VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}

_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 6: "train", 4: "airplane",
}

RANGE_COLORS = {"near": "#38bdf8", "mid": "#c084fc", "far": "#fb923c"}
LABEL_COLORS = {
    "person": "#f59e0b", "car": "#38bdf8", "truck": "#c084fc",
    "bus": "#fb923c", "motorcycle": "#a855f7", "bicycle": "#22c55e",
    "vehicle": "#94a3b8", "object": "#94a3b8",
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
    span = max(0.01, roi_max_y - roi_min_y)
    score = (min(1.0, (rel_bottom - roi_min_y) / span) * 0.55
             + min(1.0, rel_h / 0.45) * 0.45)
    if score > 0.62:
        return "near"
    if score > 0.34:
        return "mid"
    return "far"


# ---------------------------------------------------------------------------
# Pixel-shape analysis: wheels & headlights inside a vehicle bounding box.
# ---------------------------------------------------------------------------
def detect_vehicle_parts(bgr_crop):
    """Locate semantic parts (wheels / headlights) from the raw pixels.

    Returns e.g. [{"kind": "wheel", "cx": .., "cy": .., "r": .., "conf": ..},
                  {"kind": "headlight", "cx": .., "cy": .., "r": .., "conf": ..}]
    """
    parts = []
    if cv2 is None or bgr_crop is None:
        return parts
    h, w = bgr_crop.shape[:2]
    if h < 14 or w < 14:
        return parts
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    box_area = float(w * h)

    # --- WHEELS: dark circular contours in the bottom 45% of the box -------
    bottom = gray[int(h * 0.55):, :]
    if bottom.size and bottom.shape[0] >= 4:
        _, dark_mask = cv2.threshold(bottom, 75, 255, cv2.THRESH_BINARY_INV)
        k = max(3, int(min(w, h) * 0.04))
        if k % 2 == 0:
            k += 1
        dark_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        conts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        wheel_candidates = []
        for c in conts:
            a = cv2.contourArea(c)
            if a < box_area * 0.004:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < 2.0:
                continue
            peri = cv2.arcLength(c, True)
            circularity = 4.0 * np.pi * a / (peri * peri) if peri > 0 else 0
            if circularity < 0.42:
                continue
            if r > 0.72 * h:  # fills the whole band -> not a wheel
                continue
            wheel_candidates.append(((cx, cy), r, a, circularity))
        wheel_candidates.sort(key=lambda t: -t[3])
        used = []
        for (cx, cy), r, a, circ in wheel_candidates[:4]:
            if any(abs(cy - py) < r * 0.9 and abs(cx - px) < r * 3.2
                   for (px, py) in used):
                continue
            used.append((cx, cy))
            parts.append({
                "kind": "wheel", "cx": float(cx), "cy": float(cy + h * 0.55),
                "r": float(r), "conf": float(min(0.99, 0.5 + circ * 0.5)),
            })

    # --- HEADLIGHTS: bright blobs in the mid/front band ---------------------
    band = gray[int(h * 0.28):int(h * 0.78), :]
    if band.size and band.shape[0] >= 4:
        _, bright = cv2.threshold(band, 190, 255, cv2.THRESH_BINARY)
        k = max(3, int(min(w, h) * 0.03))
        if k % 2 == 0:
            k += 1
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        conts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        hl = []
        for c in conts:
            a = cv2.contourArea(c)
            if a < box_area * 0.002:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < 2.0:
                continue
            hl.append(((cx, cy), r, a))
        hl.sort(key=lambda t: -t[1])
        for (cx, cy), r, a in hl[:3]:
            parts.append({
                "kind": "headlight", "cx": float(cx),
                "cy": float(cy + h * 0.28), "r": float(r),
                "conf": float(min(0.99, 0.45 + r * 0.03)),
            })
    return parts


# ---------------------------------------------------------------------------
# Multi-object tracker (centroid + IoU matching -> stable per-object IDs).
# This is what lets the dashboard track EACH AND EVERY person in front of the
# camera instead of the single largest motion blob.
# ---------------------------------------------------------------------------
class ObjectTrack:
    __slots__ = ("id", "label", "conf", "box", "parts", "range_band",
                 "last_seen", "smooth_box")

    def __init__(self, track_id, det):
        self.id = track_id
        self.label = det.get("label", "object")
        self.conf = det.get("conf", 0.5)
        self.box = [float(v) for v in det["box"]]
        self.parts = det.get("parts", [])
        self.range_band = det.get("range_band", "mid")
        self.last_seen = time.time()
        self.smooth_box = list(self.box)

    def update(self, det, now=None):
        self.label = det.get("label", self.label)
        self.conf = det.get("conf", self.conf)
        self.parts = det.get("parts", [])
        self.range_band = det.get("range_band", self.range_band)
        box = [float(v) for v in det["box"]]
        for i in range(4):
            self.smooth_box[i] += (box[i] - self.smooth_box[i]) * 0.55
        self.box = [float(v) for v in self.smooth_box]
        self.last_seen = now if now is not None else time.time()


def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_near_boxes(cands, fw, fh):
    """Union person fragments: overlap OR near (same body).

    Face/hands/arms of ONE seated person produce adjacent skin blobs whose
    expanded body boxes overlap or sit within ~1 box width of each other.
    Two genuinely separate people are far apart horizontally, so they never
    merge. Returns merged (x, y, w, h, label, conf) tuples.
    """
    boxes = [list(c) for c in cands]
    if len(boxes) <= 1:
        return [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            ax, ay, aw, ah, alab, aconf = boxes[i]
            acx, acy = ax + aw / 2.0, ay + ah / 2.0
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx, by, bw, bh, blab, bconf = boxes[j]
                bcx, bcy = bx + bw / 2.0, by + bh / 2.0
                near_x = abs(acx - bcx) < (aw + bw) * 0.6
                near_y = abs(acy - bcy) < (ah + bh) * 0.6
                if _iou((ax, ay, aw, ah), (bx, by, bw, bh)) > 0.05 or (
                        near_x and near_y):
                    x1 = min(ax, bx)
                    y1 = min(ay, by)
                    x2 = max(ax + aw, bx + bw)
                    y2 = max(ay + ah, by + bh)
                    ax, ay = x1, y1
                    aw, ah = x2 - x1, y2 - y1
                    acx, acy = ax + aw / 2.0, ay + ah / 2.0
                    aconf = max(aconf, bconf)
                    used[j] = True
                    changed = True
            out.append((ax, ay, aw, ah, alab, aconf))
            used[i] = True
        boxes = [list(b) for b in out]
    return [tuple(b) for b in boxes]


def _nms_yolo(dets):
    """Class-aware NMS + overlap-union for YOLO raw boxes.

    YOLO can emit 2-3 overlapping 'person' boxes for one body at close
    range; without this the tracker shows 2-3 IDs for one sitter.
    Same label + IoU > 0.5 -> keep best; then union same-label boxes
    that still overlap (IoU > 0.05) so one body == one box.
    """
    if len(dets) <= 1:
        return list(dets)
    kept = []
    for _lab, group in _group_by_label(dets).items():
        group = sorted(group, key=lambda d: -float(d.get("conf", 0)))
        used = [False] * len(group)
        for i, d in enumerate(group):
            if used[i]:
                continue
            kept.append(d)
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                try:
                    if _iou(d["box"], group[j]["box"]) > 0.50:
                        used[j] = True
                except Exception:
                    pass
    final = []
    for _lab, group in _group_by_label(kept).items():
        cur = list(group)
        changed = True
        while changed:
            changed = False
            nxt = []
            skip = [False] * len(cur)
            for i in range(len(cur)):
                if skip[i]:
                    continue
                a = dict(cur[i])
                for j in range(i + 1, len(cur)):
                    if skip[j]:
                        continue
                    b = cur[j]
                    try:
                        iou = _iou(a["box"], b["box"])
                    except Exception:
                        iou = 0.0
                    if iou > 0.05:
                        x1 = min(a["box"][0], b["box"][0])
                        y1 = min(a["box"][1], b["box"][1])
                        x2 = max(a["box"][0] + a["box"][2],
                                 b["box"][0] + b["box"][2])
                        y2 = max(a["box"][1] + a["box"][3],
                                 b["box"][1] + b["box"][3])
                        a["box"] = [x1, y1, x2 - x1, y2 - y1]
                        a["conf"] = max(float(a.get("conf", 0)),
                                        float(b.get("conf", 0)))
                        skip[j] = True
                        changed = True
                nxt.append(a)
                skip[i] = True
            cur = nxt
        final.extend(cur)
    return final


def _group_by_label(dets):
    out = {}
    for d in dets:
        out.setdefault(d.get("label", "?"), []).append(d)
    return out


class MultiObjectTracker:
    """Assigns stable IDs across frames to every detected object."""

    def __init__(self, max_age=0.6, iou_thresh=0.08):
        self.tracks = OrderedDict()
        self.next_id = 1
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.frame_no = 0

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_no = 0

    def update(self, detections, now=None):
        self.frame_no += 1
        now = now if now is not None else time.time()
        dets = [d for d in detections if d.get("box")]
        matched = set()
        out = []
        for tid, tr in list(self.tracks.items()):
            best, best_score = None, self.iou_thresh
            for i, d in enumerate(dets):
                if i in matched:
                    continue
                score = _iou(tr.box, d["box"])
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                matched.add(best)
                tr.update(dets[best], now)
                out.append(self._emit(tr))
            elif now - tr.last_seen > self.max_age:
                del self.tracks[tid]
        for i, d in enumerate(dets):
            if i in matched:
                continue
            tr = ObjectTrack(self.next_id, d)
            self.next_id += 1
            self.tracks[tr.id] = tr
            out.append(self._emit(tr))
        return out

    @staticmethod
    def _emit(tr):
        return {
            "id": tr.id, "label": tr.label, "conf": round(tr.conf, 3),
            "box": [round(float(v), 1) for v in tr.box],
            "parts": tr.parts, "range_band": tr.range_band,
        }


# ---------------------------------------------------------------------------
# The semantic detector: YOLO preferred, offline OpenCV cascade fallback.
# ---------------------------------------------------------------------------
class SemanticDetector:
    """Per-frame semantic object detector + vehicle-part shape recogniser."""

    def __init__(self, backend="auto", model_path=None, conf_threshold=0.30):
        self.backend = backend
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.hog = None
        self.backend_name = None
        self.tracker = MultiObjectTracker()
        self._prev_gray = None
        if backend != "cv-cascade":
            self._init_deep_backend()
        else:
            self.backend_name = "cv-cascade"
            self._init_hog()

    def _init_hog(self):
        if cv2 is None or self.hog is not None:
            return
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            self.hog = None

    def _init_deep_backend(self):
        # 1) ultralytics YOLOv8n (modern single-file weights).
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path or "yolov8n.pt")
            self.backend_name = "yolov8n" if not self.model_path else "yolo-custom"
            return
        except Exception:
            self.model = None
        # 2) torch.hub YOLOv5s (original repo convention).
        if self.backend in ("auto", "yolov5"):
            try:
                import torch
                if self.model_path:
                    self.model = torch.hub.load(
                        "ultralytics/yolov5", "custom", path=self.model_path,
                        force_reload=True)
                    self.backend_name = "yolo-custom"
                else:
                    self.model = torch.hub.load(
                        "ultralytics/yolov5", "yolov5s", pretrained=True,
                        force_reload=True)
                    self.backend_name = "yolov5s"
                if hasattr(self.model, "classes"):
                    self.model.classes = [0, 1, 2, 3, 5, 7]
                return
            except Exception:
                self.model = None
        # 3) Fully offline semantic cascade.
        self.backend_name = "cv-cascade"
        self._init_hog()

    # -- offline per-frame semantic cues -------------------------------------
    def _skin_blobs(self, frame_bgr, fw, fh):
        """Person candidates from skin-tone pixels (per frame -> still-safe).

        One box per skin contour would over-count (face + 2 hands = 3
        "people"). So each blob is EXPANDED into a full-body estimate
        centred on the blob (wider + extended downward), letting the merge
        step below collapse all parts of one person into ONE box.
        """
        if cv2 is None or frame_bgr is None:
            return []
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 38, 60], np.uint8),
                         np.array([25, 175, 255], np.uint8))
        m2 = cv2.inRange(hsv, np.array([165, 38, 60], np.uint8),
                         np.array([180, 175, 255], np.uint8))
        mask = cv2.bitwise_or(m1, m2)
        k = max(3, int(fh * 0.012))
        if k % 2 == 0:
            k += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((k, k), np.uint8))
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in conts:
            if cv2.contourArea(c) < (fw * fh) * 0.0020:
                continue  # ignore specks: hands fragments, noise
            x, y, w, bh = cv2.boundingRect(c)
            if w < fw * 0.015 or bh < fh * 0.02:
                continue
            # Expand to a body estimate: centre on blob, ~2.2x wide,
            # from blob top down ~5x blob height (face -> torso).
            cx = x + w / 2.0
            ew = min(float(fw), w * 2.4 + fw * 0.02)
            eh = min(float(fh - y), bh * 5.0 + fh * 0.05)
            ex = max(0.0, cx - ew / 2.0)
            if ex + ew > fw:
                ex = max(0.0, fw - ew)
            boxes.append((ex, float(y), ew, eh, 0.60))
        return boxes

    def _vehicle_blobs(self, frame_bgr, fw, fh):
        """Vehicle candidates: motion blobs + still dark rectangles (per frame)."""
        boxes = []
        if cv2 is None or frame_bgr is None:
            return boxes
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, th = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
            th = cv2.dilate(th, None, iterations=2)
            conts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            for c in conts:
                if cv2.contourArea(c) < (fw * fh) * 0.0015:
                    continue
                x, y, w, bh = cv2.boundingRect(c)
                boxes.append((x, y, w, bh, 0.55))
        self._prev_gray = gray
        # Still dark rectangles in the lower 2/3 (photo of a car held still).
        # Contrast-based (luma < 100) so mid-gray cars are found too; the
        # bounding box must be CAR-LIKE (wider than tall) so dark persons,
        # shadows and poles are not labelled vehicles.
        roi = gray[int(fh * 0.30):int(fh * 0.92), :]
        _, dark = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
        k = max(5, int(fw * 0.03))
        if k % 2 == 0:
            k += 1
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                np.ones((k, k), np.uint8))
        conts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        for c in conts:
            if cv2.contourArea(c) < (fw * fh) * 0.012:
                continue
            x, y, w, bh = cv2.boundingRect(c)
            if bh < fh * 0.06 or w < fw * 0.06:
                continue
            aspect = w / float(max(1, bh))
            if not (1.05 <= aspect <= 5.0):
                continue  # taller than wide -> person / pole / shadow
            boxes.append((x, y + int(fh * 0.30), w, bh, 0.50))
        return boxes

    def _cv_cascade(self, frame_bgr):
        """Offline semantic cascade: people + vehicles, per frame."""
        if cv2 is None:
            return []
        fh, fw = frame_bgr.shape[:2]
        cand = []
        if self.hog is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                rects, _w = self.hog.detectMultiScale(
                    gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w, h) in rects:
                    cand.append((x, y, w, h, "person", 0.72))
            except Exception:
                pass
        for (x, y, w, h, c) in self._skin_blobs(frame_bgr, fw, fh):
            cand.append((x, y, w, h, "person", c))
        for (x, y, w, h, c) in self._vehicle_blobs(frame_bgr, fw, fh):
            cand.append((x, y, w, h, "vehicle", c))
        # Collapse body-part fragments: overlapping OR near person boxes
        # (same vertical band, centres within ~1 box width) belong to ONE
        # person — union them BEFORE NMS so face+hands != 3 people.
        persons = [c for c in cand if c[4] == "person"]
        others = [c for c in cand if c[4] != "person"]
        persons = _merge_near_boxes(persons, fw, fh)
        cand = persons + others
        # NMS by IoU, persons win ties (multiple cues).
        cand.sort(key=lambda t: -t[5])
        kept = []
        for c in cand:
            if any(_iou(c[:4], k[:4]) > 0.38 for k in kept):
                continue
            kept.append(c)
        return [{"label": c[4], "conf": c[5], "box": [float(v) for v in c[:4]],
                 "parts": []} for c in kept]

    def detect(self, frame_bgr):
        """Classify persons + vehicles from ONE frame's pixels.

        Returns detections: {"label", "conf", "box": [x,y,w,h],
                             "parts": [...], "range_band"}.
        """
        if frame_bgr is None:
            return []
        fh, fw = frame_bgr.shape[:2]
        dets = []
        if self.model is not None:
            try:
                if self.backend_name.startswith("yolov8"):
                    results = self.model(frame_bgr, verbose=False,
                                         conf=self.conf_threshold)
                    for r in results:
                        for i in range(len(r.boxes)):
                            cls_id = int(r.boxes.cls[i].item())
                            label = r.names.get(cls_id, "unknown")
                            if label not in PERSON_LABELS | VEHICLE_LABELS:
                                continue
                            conf = float(r.boxes.conf[i].item())
                            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                            dets.append({
                                "label": label, "conf": conf,
                                "box": [x1, y1, x2 - x1, y2 - y1],
                                "parts": []})
                elif self.backend_name == "yolov5s":
                    results = self.model(frame_bgr)
                    pred = results.pandas().xyxy[0]
                    for _, row in pred.iterrows():
                        label = str(row["name"])
                        if label not in PERSON_LABELS | VEHICLE_LABELS:
                            continue
                        if float(row["confidence"]) < self.conf_threshold:
                            continue
                        dets.append({
                            "label": label, "conf": float(row["confidence"]),
                            "box": [float(row["xmin"]), float(row["ymin"]),
                                    float(row["xmax"] - row["xmin"]),
                                    float(row["ymax"] - row["ymin"])],
                            "parts": []})
            except Exception as exc:
                print(f"[semantic][warn] deep inference failed ({exc}); "
                      f"cv-cascade for this frame", file=sys.stderr)
                self.model = None
                return self._cv_cascade(frame_bgr)

        if dets:
            # One body == one box: YOLO emits 2-3 heavily-overlapping 'person'
            # boxes for ONE seated person at close range; without class-aware
            # NMS the tracker gives 2-3 IDs to one sitter ("taking me as too
            # many people"). Merge duplicates BEFORE the tracker sees them.
            dets = _nms_yolo(dets)
        else:
            dets = self._cv_cascade(frame_bgr)

        # Shape analysis + range band for every detection.
        for d in dets:
            x, y, w, h = [int(v) for v in d["box"]]
            x, y = max(0, x), max(0, y)
            crop = frame_bgr[y:y + h, x:x + w]
            if d["label"] in VEHICLE_LABELS:
                d["parts"] = detect_vehicle_parts(crop)
            d["range_band"] = classify_range(d["box"], fw, fh)
        return dets

    def detect_and_track(self, frame_bgr):
        """Detect then assign stable IDs -> one track per object per frame."""
        dets = self.detect(frame_bgr)
        return self.tracker.update(dets)


def _draw_annotations(frame, objects):
    for obj in objects:
        x, y, w, h = [int(v) for v in obj["box"]]
        color = (255, 255, 255)
        if obj["label"] in VEHICLE_LABELS:
            color = (200, 120, 60)
        elif obj["label"] == "person":
            color = (0, 200, 120)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{obj['label']} {obj['conf']:.2f} #{obj['id']}",
                    (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1)
        for p in obj.get("parts", []):
            px, py = int(x + p["cx"]), int(y + p["cy"])
            r = max(2, int(p["r"]))
            if p["kind"] == "wheel":
                cv2.circle(frame, (px, py), r, (0, 255, 255), 2)
            else:
                cv2.circle(frame, (px, py), r, (0, 255, 255), -1)
    return frame


def main():
    ap = argparse.ArgumentParser(description="YOLO semantic shape detector")
    ap.add_argument("--image", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "yolov8", "yolov5", "cv-cascade"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    if cv2 is None:
        print("[ERROR] opencv-python is required.")
        sys.exit(1)

    det = SemanticDetector(backend=args.backend if args.backend != "auto"
                           else "cv-cascade", model_path=args.weights)
    print(f"[semantic] backend resolved: {det.backend_name}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] cannot read {args.image}")
            sys.exit(1)
        t0 = time.time()
        objs = det.detect_and_track(frame)
        print(json.dumps({"backend": det.backend_name, "objects": objs,
                          "infer_ms": round((time.time() - t0) * 1e3, 1)},
                         indent=2))
        if not args.no_window:
            _draw_annotations(frame, objs)
            cv2.imshow("semantic-detector", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    else:
        ap.print_help()
        return
    n = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        objs = det.detect_and_track(frame)
        n += 1
        if n % 15 == 1:
            print(f"  frame {n}: {len(objs)} object(s) -> "
                  + ", ".join(f"{o['label']}#{o['id']}" for o in objs))
        if not args.no_window:
            _draw_annotations(frame, objs)
            cv2.imshow("semantic-detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()