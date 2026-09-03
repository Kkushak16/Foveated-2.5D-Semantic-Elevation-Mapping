"""
train_perception_models.py — One-shot trainer for camera tracking + vehicle info.
===============================================================================
Trains:
  1. camera_tracking_model.py (ndrplz/self-driving-car style steering CNN)
  2. vehicle_info_model.py    (ahmetozlu/vehicle_counting style counting+color)

Writes web/ui/vehicle_info.json + camera_tracking_report.json so the Teleop HUD
can display live-trained vehicle count / color / steering telemetry.

Usage:
    python train_perception_models.py --driving-data ./data/driving --epochs 10
    python train_perception_models.py --traffic-video ./data/traffic.mp4
    python train_perception_models.py --smoke   (no data needed, verifies pipeline)
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from camera_tracking_model import train as train_tracker
from vehicle_info_model import analyze_video, analyze_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driving-data", default="./data/driving")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--traffic-video", default=None)
    ap.add_argument("--vehicle-image", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="run without datasets; writes stub reports + sample HUD json")
    args = ap.parse_args()

    if args.smoke:
        tracking = {"status": "smoke-stub", "note": "pass --driving-data to train for real"}
        vehicle = {"total_vehicles": 0, "color_histogram": {},
                   "dominant_color": "unknown", "status": "smoke-stub"}
    else:
        tracking = train_tracker(args.driving_data, epochs=args.epochs)
        if args.traffic_video:
            vehicle = analyze_video(args.traffic_video)
        elif args.vehicle_image:
            vehicle = analyze_image(args.vehicle_image)
        else:
            vehicle = {"status": "no-input",
                       "hint": "pass --traffic-video traffic.mp4 or --vehicle-image car.jpg"}

    with open(os.path.join(SCRIPT_DIR, "camera_tracking_report.json"), "w") as f:
        json.dump(tracking, f, indent=2)
    with open(os.path.join(SCRIPT_DIR, "vehicle_info_report.json"), "w") as f:
        json.dump(vehicle, f, indent=2)

    # HUD-facing bundle (served statically next to index.html)
    hud_path = os.path.join(SCRIPT_DIR, "..", "web", "ui", "vehicle_info.json")
    hud_payload = {
        "vehicle_count": vehicle.get("total_vehicles", 0),
        "dominant_color": vehicle.get("dominant_color", "unknown"),
        "color_histogram": vehicle.get("color_histogram", {}),
        "tracking_status": tracking.get("status", "unknown"),
    }
    with open(os.path.normpath(hud_path), "w") as f:
        json.dump(hud_payload, f, indent=2)

    print(f"[OK] tracking={tracking.get('status')} vehicles={hud_payload['vehicle_count']} "
          f"color={hud_payload['dominant_color']}")
    print(f"[OK] HUD bundle -> {os.path.normpath(hud_path)}")


if __name__ == "__main__":
    main()
