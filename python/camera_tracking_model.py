"""
camera_tracking_model.py — Camera tracking / steering-angle model.
===============================================================
Approach adapted from ndrplz/self-driving-car (behavioral cloning):
Udacity-simulator style CNN that maps a front-camera frame -> steering angle.

Adaptations for this repo (Foveated 2.5D Perception HUD):
- Preprocessing matches the HUD's semantic spatial mask:
  sky top 35% + vehicle hood bottom 15% are cropped BEFORE resize,
  so the network trains on the same 50% ROI the dashboard runs live.
- Target input 66x200 (NVIDIA DAVE-2 style, as in ndrplz repo).
- PyTorch implementation with ONNX export (repo already ships model.onnx).
- Falls back to a NumPy-only stub predictor when torch is unavailable,
  so `train_perception_models.py --help` and the dashboard never crash.

Original repo: https://github.com/ndrplz/self-driving-car.git
See ATTRIBUTION.md for license/credit. This file is an original,
interface-compatible re-implementation (no vendored code).

Usage:
    python camera_tracking_model.py --data ./data/driving --epochs 10
    python camera_tracking_model.py --predict frame.jpg
"""

import argparse
import json
import os

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


# --- Preprocessing (shared by training + live inference) --------------------
SKY_CUTOFF = 0.35   # matches HUD semantic mask
HOOD_CUTOFF = 0.85  # matches HUD semantic mask
TARGET_W, TARGET_H = 200, 66


def preprocess_frame(pil_img):
    """Crop sky/hood, resize to 66x200, normalize to [-0.5, 0.5]."""
    w, h = pil_img.size
    top = int(h * SKY_CUTOFF)
    bottom = int(h * HOOD_CUTOFF)
    cropped = pil_img.crop((0, top, w, bottom))
    resized = cropped.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0 - 0.5
    # HWC -> CHW
    return np.transpose(arr, (2, 0, 1))


# --- Network (NVIDIA DAVE-2 / ndrplz-style) ----------------------------------
if HAS_TORCH:
    class SteeringCNN(nn.Module):
        """5 conv + 4 FC behavioral-cloning network (ndrplz-style)."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
                nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
                nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
                nn.Conv2d(48, 64, 3), nn.ELU(),
                nn.Conv2d(64, 64, 3), nn.ELU(),
                nn.Flatten(),
            )
            # infer flatten dim
            with torch.no_grad():
                n = self.conv(torch.zeros(1, 3, TARGET_H, TARGET_W)).shape[1]
            self.fc = nn.Sequential(
                nn.Linear(n, 100), nn.ELU(), nn.Dropout(0.5),
                nn.Linear(100, 50), nn.ELU(), nn.Dropout(0.5),
                nn.Linear(50, 10), nn.ELU(),
                nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.fc(self.conv(x))
else:
    SteeringCNN = None


class CameraTracker:
    """High-level wrapper: load weights (or stub) + predict steering."""

    def __init__(self, weights_path=None):
        self.device = None
        self.model = None
        if HAS_TORCH and weights_path and os.path.exists(weights_path):
            self.model = SteeringCNN()
            state = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            self.device = torch.device("cpu")

    def predict_steering(self, pil_img):
        """Return steering angle in [-1, 1]. Stub = lane-center heuristic."""
        if self.model is not None:
            x = preprocess_frame(pil_img)
            with torch.no_grad():
                t = torch.from_numpy(x).unsqueeze(0)
                return float(self.model(t).squeeze().item())
        # Stub: lateral offset of brightest road-region column from center.
        arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
        h, w = arr.shape
        roi = arr[int(h * SKY_CUTOFF):int(h * HOOD_CUTOFF), :]
        col_energy = roi.mean(axis=0)
        peak = int(np.argmax(col_energy))
        return float((peak / w - 0.5) * 2.0 * 0.5)

    def export_onnx(self, out_path="model_camera_tracking.onnx"):
        if not HAS_TORCH or self.model is None:
            raise RuntimeError("ONNX export needs torch + loaded weights.")
        dummy = torch.zeros(1, 3, TARGET_H, TARGET_W)
        torch.onnx.export(self.model, dummy, out_path,
                          input_names=["camera_frame"],
                          output_names=["steering_angle"])
        return out_path


def train(data_dir, epochs=10, out_weights="camera_tracking_weights.pt"):
    """Train on <data_dir>/driving_log.csv + IMG/ (Udacity format)."""
    if not HAS_TORCH:
        print("[camera_tracking] torch not installed — writing stub report.")
        report = {"status": "stub", "reason": "torch missing", "epochs": 0}
        with open("camera_tracking_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report
    import csv
    samples = []
    log_path = os.path.join(data_dir, "driving_log.csv")
    if not os.path.exists(log_path):
        print(f"[camera_tracking] no driving_log.csv in {data_dir} — stub report.")
        report = {"status": "no-data", "data_dir": data_dir}
        with open("camera_tracking_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report
    with open(log_path) as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                samples.append((row[0].strip(), float(row[3])))
    print(f"[camera_tracking] {len(samples)} samples, {epochs} epochs (ndrplz-style CNN).")
    model = SteeringCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for img_rel, angle in samples:
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(data_dir, img_rel)
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            x = torch.from_numpy(preprocess_frame(img)).unsqueeze(0)
            y = torch.tensor([[angle]], dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"  epoch {epoch + 1}/{epochs}  mse={total / max(n, 1):.4f}")
    torch.save(model.state_dict(), out_weights)
    report = {"status": "trained", "samples": len(samples),
              "epochs": epochs, "weights": out_weights}
    with open("camera_tracking_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[camera_tracking] saved {out_weights}")
    return report


def main():
    ap = argparse.ArgumentParser(description="ndrplz-style camera tracking trainer")
    ap.add_argument("--data", default="./data/driving")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--predict", default=None, help="image path for steering test")
    ap.add_argument("--weights", default="camera_tracking_weights.pt")
    args = ap.parse_args()
    if args.predict:
        tracker = CameraTracker(args.weights)
        img = Image.open(args.predict).convert("RGB")
        print(f"steering={tracker.predict_steering(img):.3f}")
    else:
        train(args.data, epochs=args.epochs, out_weights=args.weights)


if __name__ == "__main__":
    main()
