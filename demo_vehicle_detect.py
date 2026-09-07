"""
demo_vehicle_detect.py — Quick demo: detect vehicles in a single image or video.
=================================================================================
Uses the VehicleCounter from python/vehicle_info_model.py. Works in three
modes depending on what is installed:

  1. torch + yolov5 available  → YOLOv5 deep detection (best accuracy)
  2. torch + custom weights    → Custom-trained YOLOv5 model (--weights flag)
  3. No torch, only OpenCV     → Frame-diff + red-mask fallback (lightweight)

Usage:
    python demo_vehicle_detect.py --image car.jpg
    python demo_vehicle_detect.py --image car.jpg --weights my_yolo_models/car_detector/weights/best.pt
    python demo_vehicle_detect.py --video traffic.mp4
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(SCRIPT_DIR, "python")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Ensure python/ is importable
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy is required. Install it: pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None
    print("[WARN] OpenCV not installed — visual output will be limited.")

try:
    from PIL import Image
except ImportError:
    Image = None

from vehicle_info_model import VehicleCounter, classify_vehicle_color


def draw_boxes(frame, tracks, line_y):
    """Draw bounding boxes, centroids, and the ROI counting line."""
    annotated = frame.copy()
    # ROI counting line
    h, w = annotated.shape[:2]
    cv2.line(annotated, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(annotated, "ROI Line", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for tr in tracks:
        box = tr.get("box")
        if box is None:
            continue
        x, y, bw, bh = [int(v) for v in box]
        color_name = tr.get("color", "unknown")
        counted = tr.get("counted", False)

        # Box color: green if counted, blue otherwise
        box_color = (0, 200, 0) if counted else (255, 180, 0)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)

        # Label
        label = color_name
        cv2.putText(annotated, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Centroid dot
        cx, cy = int(tr["cx"]), int(tr["cy"])
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return annotated


def demo_image(image_path, counter):
    """Run detection on a single image and save annotated result."""
    if cv2 is None:
        print("[ERROR] OpenCV is required for image demo. pip install opencv-python")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Image loaded: {w}x{h} — {image_path}")

    # Run detection (for a single image, we call update twice:
    # once to prime the frame-diff baseline, once to detect)
    counter.update(frame)
    total, tracks = counter.update(frame)

    # If the deep detector is active, tracks come from the first call too
    if not tracks:
        # Re-run _detect directly for single-image mode
        boxes = counter._detect(frame)
        tracks = []
        for (x, y, bw, bh) in boxes:
            crop = frame[max(y, 0):y + bh, max(x, 0):x + bw]
            color = classify_vehicle_color(crop)
            tracks.append({
                "cx": x + bw / 2, "cy": y + bh / 2,
                "box": (x, y, bw, bh), "color": color, "counted": False,
            })

    line_y = int(h * counter.line_y_ratio)
    annotated = draw_boxes(frame, tracks, line_y)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "demo_output.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"[SUCCESS] Annotated image saved -> {out_path}")

    # Also update web/ui/vehicle_info.json for the dashboard HUD
    info = {
        "vehicles_detected": len(tracks),
        "colors": [t.get("color", "unknown") for t in tracks],
        "source": os.path.basename(image_path),
    }
    hud_path = os.path.join(SCRIPT_DIR, "web", "ui", "vehicle_info.json")
    try:
        with open(hud_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[INFO] Dashboard HUD updated -> {hud_path}")
    except Exception as e:
        print(f"[WARN] Could not write HUD JSON: {e}")

    print(json.dumps(info, indent=2))
    return info


def demo_video(video_path, counter, max_frames=600):
    """Process a video frame-by-frame and save the last annotated frame."""
    if cv2 is None:
        print("[ERROR] OpenCV is required for video demo. pip install opencv-python")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    print(f"[INFO] Processing video: {video_path} (max {max_frames} frames)")
    n = 0
    last_frame = None
    last_tracks = []
    while cap.isOpened() and n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        total, tracks = counter.update(frame)
        last_frame = frame
        last_tracks = tracks
        n += 1
        if n % 50 == 0:
            print(f"  Frame {n}: count={total}, active_tracks={len(tracks)}")
    cap.release()

    summary = counter.summary()
    summary["video"] = video_path
    summary["frames_processed"] = n

    # Save last annotated frame
    if last_frame is not None:
        h = last_frame.shape[0]
        line_y = int(h * counter.line_y_ratio)
        annotated = draw_boxes(last_frame, last_tracks, line_y)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "demo_video_last_frame.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[SUCCESS] Last frame saved -> {out_path}")

    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Demo: YOLOv5 vehicle detection & counting")
    ap.add_argument("--image", default=None,
                    help="Path to a single image")
    ap.add_argument("--video", default=None,
                    help="Path to a video file")
    ap.add_argument("--weights", default=None,
                    help="Path to custom YOLOv5 best.pt weights "
                         "(e.g. my_yolo_models/car_detector/weights/best.pt)")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="Max frames to process in video mode")
    args = ap.parse_args()

    if not args.image and not args.video:
        # Default: try the first dashboard frame as a test
        default = os.path.join(SCRIPT_DIR, "dashboard_frame_001.bmp")
        if os.path.exists(default):
            print(f"[INFO] No input specified — using default test image: {default}")
            args.image = default
        else:
            ap.print_help()
            print("\n[HINT] Pass --image <path> or --video <path>")
            sys.exit(1)

    counter = VehicleCounter(model_path=args.weights)

    if counter.torch is not None:
        print("[INFO] Using YOLOv5 deep detector (torch)")
        if counter._is_custom_model:
            print(f"[INFO] Custom weights: {args.weights}")
    else:
        print("[INFO] Using frame-diff + red-mask fallback (no torch)")

    if args.image:
        demo_image(args.image, counter)
    elif args.video:
        demo_video(args.video, counter, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
