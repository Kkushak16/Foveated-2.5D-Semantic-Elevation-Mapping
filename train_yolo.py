"""
train_yolo.py — Convenience wrapper for YOLOv5 training.
=========================================================
Clones the ultralytics/yolov5 repo if not already present, installs its
requirements, and launches training with sensible defaults.

All CLI flags are forwarded to yolov5/train.py, so you can override
anything (epochs, batch-size, image-size, etc.).

Usage:
    # Train with defaults (50 epochs, batch 16, yolov5s):
    python train_yolo.py

    # Override parameters:
    python train_yolo.py --epochs 100 --batch 8 --img 416

    # Use a different base config:
    python train_yolo.py --cfg yolov5m.yaml --epochs 30

After training, your best weights will be at:
    my_yolo_models/car_detector/weights/best.pt

Point VehicleCounter to them:
    counter = VehicleCounter(model_path='my_yolo_models/car_detector/weights/best.pt')
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR = os.path.join(SCRIPT_DIR, "yolov5")
DATASET_YAML = os.path.join(SCRIPT_DIR, "my_dataset", "data.yaml")


def ensure_yolov5_repo():
    """Clone yolov5 repo if it doesn't exist, then install requirements."""
    if os.path.isdir(YOLO_DIR) and os.path.isfile(os.path.join(YOLO_DIR, "train.py")):
        print(f"[INFO] YOLOv5 repo found at {YOLO_DIR}")
        return True

    print("[INFO] Cloning ultralytics/yolov5 ...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/ultralytics/yolov5.git", YOLO_DIR],
            check=True)
    except Exception as e:
        print(f"[ERROR] Failed to clone yolov5: {e}")
        print("[HINT] Install git and retry, or manually clone:")
        print(f"       git clone https://github.com/ultralytics/yolov5.git {YOLO_DIR}")
        return False

    # Install yolov5 requirements
    req_file = os.path.join(YOLO_DIR, "requirements.txt")
    if os.path.isfile(req_file):
        print("[INFO] Installing yolov5 requirements ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            check=True)

    return True


def check_dataset():
    """Verify the dataset folder has at least one image."""
    images_dir = os.path.join(SCRIPT_DIR, "my_dataset", "images")
    if not os.path.isdir(images_dir):
        print(f"[WARN] Images directory not found: {images_dir}")
        return False
    images = [f for f in os.listdir(images_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        print(f"[WARN] No images found in {images_dir}")
        print("[HINT] Add labelled images to my_dataset/images/ and")
        print("       corresponding .txt label files to my_dataset/labels/")
        return False
    print(f"[INFO] Found {len(images)} images in {images_dir}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Train a custom YOLOv5 vehicle detector",
        epilog="Any unknown flags are forwarded directly to yolov5/train.py")
    ap.add_argument("--data", default=DATASET_YAML,
                    help=f"Path to data.yaml (default: {DATASET_YAML})")
    ap.add_argument("--cfg", default="yolov5s.yaml",
                    help="YOLOv5 model config (default: yolov5s.yaml)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs (default: 50)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Batch size (default: 16)")
    ap.add_argument("--img", type=int, default=640,
                    help="Image size (default: 640)")
    ap.add_argument("--project", default="my_yolo_models",
                    help="Output project directory (default: my_yolo_models)")
    ap.add_argument("--name", default="car_detector",
                    help="Experiment name (default: car_detector)")
    args, extra = ap.parse_known_args()

    print("=" * 70)
    print("  YOLOV5 CUSTOM VEHICLE DETECTOR — TRAINING LAUNCHER")
    print("=" * 70)

    # Step 1: Ensure yolov5 is cloned
    if not ensure_yolov5_repo():
        sys.exit(1)

    # Step 2: Check dataset
    if not check_dataset():
        print("\n[ERROR] Cannot train without images. Add your labelled data to my_dataset/")
        print("        See my_dataset/data.yaml for format instructions.")
        sys.exit(1)

    # Step 3: Launch training
    train_script = os.path.join(YOLO_DIR, "train.py")
    cmd = [
        sys.executable, train_script,
        "--data", args.data,
        "--cfg", args.cfg,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--img", str(args.img),
        "--project", os.path.join(SCRIPT_DIR, args.project),
        "--name", args.name,
    ] + extra

    print(f"\n[INFO] Launching training:\n  {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=YOLO_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with exit code {e.returncode}")
        sys.exit(1)

    weights_path = os.path.join(SCRIPT_DIR, args.project, args.name, "weights", "best.pt")
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    if os.path.isfile(weights_path):
        print(f"  Best weights: {weights_path}")
        print(f"\n  To use in your project:")
        print(f"    counter = VehicleCounter(model_path='{weights_path}')")
        print(f"\n  Or run the demo:")
        print(f"    python demo_vehicle_detect.py --image test.jpg --weights {weights_path}")
    else:
        print(f"  [WARN] Expected weights not found at: {weights_path}")
        print(f"  Check the {args.project}/{args.name}/ directory for output.")
    print("=" * 70)


if __name__ == "__main__":
    main()
