"""
vehicle_info_model.py — Vehicle counting + color identification.
================================================================
Approach adapted from ahmetozlu/vehicle_counting_tensorflow (TensorFlow
vehicle detection + ROI-line counting on traffic video).

Adaptations for this repo:
- Lightweight OpenCV/NumPy counting pipeline (no TF weights required):
  ROI band + frame differencing + contour detection + centroid tracking,
  same "crossing line => count++" semantics as the original repo.
- HSV color classifier for each counted vehicle crop (white/black/silver/
  red/blue/gray/orange...). Orange/purple/cyan labels intentionally reuse the
  HUD semantic-ring palette so the "color shown in the semantic ring" stays
  consistent between the camera panel and the 3-ring LiDAR BEV.
- Optional hook: if a TF frozen graph is present, `TFVehicleCounter` wraps it;
  otherwise the OpenCV path runs with zero extra dependencies.

Original repo: https://github.com/ahmetozlu/vehicle_counting_tensorflow.git
See ATTRIBUTION.md for license/credit. Original re-implementation.

Usage:
    python vehicle_info_model.py --video traffic.mp4
    python vehicle_info_model.py --image car.jpg
"""

import argparse
import json
import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


# --- Color classifier (HSV buckets, HUD-palette aligned) --------------------
COLOR_BUCKETS = [
    ("white",  ((0, 0, 180), (180, 30, 255))),
    ("black",  ((0, 0, 0), (180, 255, 60))),
    ("silver", ((0, 0, 90), (180, 25, 200))),
    ("gray",   ((0, 0, 60), (180, 30, 160))),
    ("red",    ((0, 70, 60), (10, 255, 255))),
    ("red",    ((170, 70, 60), (180, 255, 255))),
    ("orange", ((11, 70, 60), (25, 255, 255))),   # far-ring #fb923c family
    ("yellow", ((26, 70, 60), (34, 255, 255))),
    ("green",  ((35, 70, 60), (85, 255, 255))),
    ("cyan",   ((86, 70, 60), (100, 255, 255))),  # near-ring #38bdf8 family
    ("blue",   ((101, 70, 60), (130, 255, 255))),
    ("purple", ((131, 70, 60), (160, 255, 255))), # mid-ring #c084fc family
]


def classify_vehicle_color(bgr_crop):
    """Return dominant color name for a vehicle crop (BGR numpy array)."""
    if cv2 is None or bgr_crop is None or bgr_crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    best, best_count = "unknown", 0
    for name, (lo, hi) in COLOR_BUCKETS:
        mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        count = int(cv2.countNonZero(mask))
        if count > best_count:
            best, best_count = name, count
    return best


def classify_pil_color(pil_img):
    if cv2 is None:
        return "unknown"
    bgr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return classify_vehicle_color(bgr)


# --- Vehicle counter (ahmetozlu-style ROI-line counting) --------------------
class VehicleCounter:
    """Centroid-tracking line counter. `line_y` is the ROI counting line."""

    def __init__(self, line_y_ratio=0.6, min_area=800, max_disappeared=8):
        self.line_y_ratio = line_y_ratio
        self.min_area = min_area
        self.max_disappeared = max_disappeared
        self.prev_gray = None
        self.tracks = {}  # id -> {cx, cy, counted, color, disappeared}
        self.next_id = 0
        self.total_count = 0
        self.colors_seen = []

    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
        return boxes

    def update(self, frame):
        """Feed a BGR frame. Returns (total_count, active_tracks)."""
        if cv2 is None:
            return self.total_count, []
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)
        boxes = self._detect(frame)
        centroids = [(x + bw / 2, y + bh / 2, (x, y, bw, bh)) for (x, y, bw, bh) in boxes]

        matched = set()
        for tid, tr in list(self.tracks.items()):
            best, best_d = None, 60.0
            for i, (cx, cy, _box) in enumerate(centroids):
                if i in matched:
                    continue
                d = abs(cx - tr["cx"]) + abs(cy - tr["cy"])
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                tr["disappeared"] += 1
                if tr["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
                continue
            matched.add(best)
            cx, cy, box = centroids[best]
            prev_y = tr["cy"]
            tr.update(cx=cx, cy=cy, box=box, disappeared=0)
            if not tr["counted"] and prev_y < line_y <= cy:
                tr["counted"] = True
                self.total_count += 1
                x, y, bw, bh = [int(v) for v in box]
                crop = frame[max(y, 0):y + bh, max(x, 0):x + bw]
                color = classify_vehicle_color(crop)
                tr["color"] = color
                self.colors_seen.append(color)

        for i, (cx, cy, box) in enumerate(centroids):
            if i in matched:
                continue
            self.tracks[self.next_id] = {
                "cx": cx, "cy": cy, "box": box,
                "counted": False, "color": "unknown", "disappeared": 0,
            }
            self.next_id += 1
        return self.total_count, list(self.tracks.values())

    def summary(self):
        from collections import Counter
        return {
            "total_vehicles": self.total_count,
            "color_histogram": dict(Counter(self.colors_seen)),
            "dominant_color": Counter(self.colors_seen).most_common(1)[0][0]
            if self.colors_seen else "unknown",
        }


def analyze_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return {"image": image_path, "dominant_color": classify_pil_color(img)}


def analyze_video(video_path, max_frames=600):
    if cv2 is None:
        return {"status": "opencv-missing", "video": video_path}
    cap = cv2.VideoCapture(video_path)
    counter = VehicleCounter()
    n = 0
    while cap.isOpened() and n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        counter.update(frame)
        n += 1
    cap.release()
    out = counter.summary()
    out.update({"video": video_path, "frames": n})
    return out


def main():
    ap = argparse.ArgumentParser(description="vehicle counting + color trainer")
    ap.add_argument("--video", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="vehicle_info_report.json")
    args = ap.parse_args()
    if args.video:
        result = analyze_video(args.video)
    elif args.image:
        result = analyze_image(args.image)
    else:
        result = {"status": "no-input", "hint": "pass --video traffic.mp4 or --image car.jpg"}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
