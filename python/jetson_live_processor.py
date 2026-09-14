"""
jetson_live_processor.py — Jetson-Native Perception Entry Point
================================================================
Reads from a local CSI/USB camera (and optionally a real LiDAR driver
publishing ROS 2 PointCloud2 topics) and feeds frames into the same
foveated perception pipeline used by camera_foveated_processor.py,
without requiring a phone or cloud tunnel.

This script is the Tier 2 (physical Jetson) counterpart to the
Colab-based demo in colab_deploy.ipynb.

Usage:
    # USB webcam (simplest)
    python3 python/jetson_live_processor.py --camera-type usb --device 0

    # CSI camera (Raspberry Pi Camera v2/HQ via MIPI CSI-2)
    python3 python/jetson_live_processor.py --camera-type csi --device 0

    # With ROS 2 LiDAR integration
    python3 python/jetson_live_processor.py --camera-type usb --lidar-topic /velodyne_points
"""

import argparse
import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("[jetson_live] WARNING: opencv-python not installed. "
          "Install with: pip3 install opencv-python", file=sys.stderr)


def gstreamer_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    """Build a Jetson-specific GStreamer pipeline for CSI cameras (nvarguscamerasrc).
    USB webcams do NOT need this — cv2.VideoCapture(device_index) works directly.

    This pipeline uses Jetson's hardware-accelerated video processing:
    - nvarguscamerasrc: Jetson camera driver (uses ISP for auto-exposure, AWB)
    - NVMM: Zero-copy GPU memory (avoids CPU-GPU memcpy)
    - nvvidconv: Hardware color space conversion
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={framerate}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def process_frame(frame, frame_count, width, height):
    """Apply the foveated pipeline to a single camera frame.

    This replicates the same processing as camera_foveated_processor.py:
    1. Semantic Spatial Masking (sky 35%, hood 15%)
    2. 3-Ring foveated resolution alignment
    3. Motion gating placeholder (requires frame history)
    """
    # --- 1. Semantic Spatial Masking ---
    sky_cutoff = int(height * 0.35)
    hood_cutoff = int(height * 0.85)
    # Zero out sky and hood regions
    masked = frame.copy()
    masked[:sky_cutoff, :] = 0      # Sky region
    masked[hood_cutoff:, :] = 0     # Hood region
    active_roi = masked[sky_cutoff:hood_cutoff, :]

    # --- 2. 3-Ring Foveated Resolution Alignment ---
    roi_h = active_roi.shape[0]

    # Near ring (bottom 30% of ROI) — full resolution
    near_start = int(roi_h * 0.70)
    near_crop = active_roi[near_start:, :]

    # Mid ring (middle 40% of ROI) — 0.5x downsampled
    mid_start = int(roi_h * 0.30)
    mid_crop = active_roi[mid_start:near_start, :]
    if mid_crop.shape[0] > 0 and mid_crop.shape[1] > 0:
        mid_h, mid_w = mid_crop.shape[:2]
        mid_ds = cv2.resize(mid_crop, (mid_w // 2, mid_h // 2))
    else:
        mid_ds = mid_crop

    # Far ring (top 30% of ROI) — 0.25x downsampled
    far_crop = active_roi[:mid_start, :]
    if far_crop.shape[0] > 0 and far_crop.shape[1] > 0:
        far_h, far_w = far_crop.shape[:2]
        far_ds = cv2.resize(far_crop, (far_w // 4, far_h // 4))
    else:
        far_ds = far_crop

    # --- Compute savings ---
    total_raw = width * height
    active_ratio = 1.0 - 0.35 - 0.15  # 50% retained after masking
    total_processed = (
        (0.30 * total_raw * active_ratio * 1.0) +     # Near: full res
        (0.40 * total_raw * active_ratio * 0.25) +     # Mid: 0.5x each dim = 0.25
        (0.30 * total_raw * active_ratio * 0.0625)     # Far: 0.25x each dim = 0.0625
    )
    savings_pct = (1.0 - total_processed / total_raw) * 100.0

    if frame_count % 30 == 0:  # Log every 30 frames
        print(f"[frame {frame_count}] Foveated savings: {savings_pct:.1f}% | "
              f"Near: {near_crop.shape} | Mid: {mid_ds.shape} | Far: {far_ds.shape}")

    return masked, savings_pct


def main():
    parser = argparse.ArgumentParser(
        description="Jetson-native foveated perception processor")
    parser.add_argument("--camera-type", choices=["usb", "csi"], default="usb",
                        help="Camera interface type (default: usb)")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Capture height (default: 720)")
    parser.add_argument("--framerate", type=int, default=30,
                        help="CSI camera framerate (default: 30)")
    parser.add_argument("--lidar-topic", default=None,
                        help="Optional ROS 2 topic for a real LiDAR driver "
                             "(e.g., /velodyne_points)")
    parser.add_argument("--display", action="store_true",
                        help="Show processed frames in a window (needs display)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without any display output")
    args = parser.parse_args()

    if cv2 is None:
        print("ERROR: OpenCV is required. Install with: pip3 install opencv-python",
              file=sys.stderr)
        sys.exit(1)

    # --- Open camera ---
    print("=" * 70)
    print(f"  JETSON LIVE FOVEATED PROCESSOR — {args.camera_type.upper()} Camera")
    print("=" * 70)

    if args.camera_type == "csi":
        pipeline = gstreamer_csi_pipeline(
            args.device, args.width, args.height, args.framerate)
        print(f"[INFO] CSI pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print(f"[INFO] Opening USB camera at /dev/video{args.device}")
        cap = cv2.VideoCapture(args.device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Could not open camera — check device index / CSI wiring",
              file=sys.stderr)
        print("  USB: Verify with 'ls /dev/video*'")
        print("  CSI: Check ribbon cable connection and 'nvarguscamerasrc' driver")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}x{actual_h}")

    # --- Optional: ROS 2 LiDAR integration ---
    if args.lidar_topic:
        print(f"[INFO] LiDAR topic requested: {args.lidar_topic}")
        print("[INFO] To use real LiDAR, run the ROS 2 bridge node separately:")
        print(f"       ros2 run foveated_grid_bridge lidar_bridge_node "
              f"--ros-args -p lidar_topic:={args.lidar_topic}")

    # --- Main processing loop ---
    frame_count = 0
    fps_start = time.time()
    fps_count = 0

    print(f"\n[RUNNING] Processing live camera feed... (Ctrl+C to stop)")
    if args.display:
        print("[INFO] Display mode: showing processed frames in window")
    elif not args.headless:
        print("[INFO] Headless mode: processing without display")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed — retrying...", file=sys.stderr)
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_count += 1

            # Process through foveated pipeline
            processed, savings = process_frame(
                frame, frame_count, actual_w, actual_h)

            # FPS calculation
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                if frame_count % 30 == 0:
                    print(f"[perf] {fps:.1f} FPS | Frame #{frame_count}")
                fps_start = time.time()
                fps_count = 0

            # Display if requested
            if args.display:
                cv2.imshow("Jetson Foveated Processor", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped after {frame_count} frames.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        print("[INFO] Camera released. Exiting.")


if __name__ == "__main__":
    main()
