## PHYSICAL TESTING & REMOTE GPU DEPLOYMENT GUIDE (v2 — Updated)

Zero-Hardware Prototyping Blueprint: Free Cloud GPUs & Smartphone Camera Integration
for Foveated 2.5D Elevation Mapping (SIH26053)

> **Important reconciliation note (read first):** This document assumes a GPU/CUDA/
> TensorRT/YOLOv8 deployment path. Earlier project planning deliberately chose a
> **CPU-first, lite pipeline** (Patchwork++/RANSAC ground segmentation, Euclidean
> clustering, Random Forest classification, no CUDA requirement in the grid engine)
> specifically so the system runs on a laptop with no GPU dependency. These two paths
> are **not in conflict** if treated as two deployment tiers:
> - **Tier 1 (baseline, always works):** the lite CPU pipeline. This is what you demo
>   from a laptop with zero hardware risk — see the master roadmap context.
> - **Tier 2 (accelerated, optional upgrade):** this document's GPU/Jetson/YOLOv8 path.
>   Use it only if a Jetson board or cloud GPU is actually available and tested ahead
>   of time. Never let Tier 2 become a single point of failure for the demo — Tier 1
>   is the fallback if Tier 2 breaks.
> The rest of this document adds the missing pieces to make Tier 2 a real, connectable
> deployment rather than just a cloud-notebook proof of concept.

---

## 1. Executive Summary & Virtual Testing Strategy

Deploying and evaluating real-time autonomous vehicle perception pipelines on dedicated
physical edge computing boards (such as the NVIDIA Jetson AGX Orin or Jetson Xavier)
usually presents significant hardware cost barriers ($1,000 to $2,500+ per unit).
However, physical hardware is not mandatory to perform rigorous, publication-grade
benchmarks or to deliver a compelling live demonstration for Smart India Hackathon
(SIH) evaluators.

By utilizing cloud-based GPU environments (Google Colab, RunPod, NVIDIA LaunchPad)
paired with smartphone cameras (via HTML5 WebRTC or RTSP video streaming), teams can
achieve 100% equivalent CUDA compilation, TensorRT INT8 quantization, and real-time
BEV mapping verification. This guide outlines the exact setup procedure to run and
demonstrate the system remotely at zero cost, **and** — new in this revision — the
exact steps to move from "cloud GPU simulation" to an actual physical Jetson board if
one becomes available before the demo.

| Deployment Path | GPU Architecture | Cost | Primary Advantage for Demo |
| --- | --- | --- | --- |
| Option 1: Google Colab | NVIDIA T4 / L4 (16GB VRAM) | 100% Free | Zero setup cost, instant CUDA build & public ngrok URL |
| Option 2: Laptop CPU (Tier 1 baseline) | None required | Free | Zero risk, matches the lite pipeline, always works |
| Option 3: Massed Compute / RunPod | NVIDIA RTX 4090 / T4 | ~$0.10–$0.20/hr | Persistent Linux container, dedicated RTSP streaming |
| Option 4: NVIDIA LaunchPad | Enterprise Jetson Sandbox | Free Trial / Grant | Native ARM64 / Jetson hardware emulation sandbox |
| Option 5: Physical Jetson board (new) | Jetson Orin Nano / Xavier NX | Owned/borrowed hardware | Real edge deployment, real sensor I/O, true latency numbers |

---

## 2. Remote GPU Infrastructure Setup (Options 1, 3, & 4)

### Option 1: Google Colab (100% Free NVIDIA T4/L4 GPU)

Google Colab provides free cloud access to NVIDIA T4 or L4 GPUs running Linux,
delivering identical CUDA SIMT architecture to NVIDIA Jetson edge boards. Follow these
step-by-step commands inside a Colab notebook:

```
Google Colab Deployment Script
# Step A: Enable GPU Runtime in Colab (Runtime -> Change runtime type -> T4 GPU)
# Step B: Clone Repo & Install Core Dependencies
!git clone https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping.git
%cd Foveated-2.5D-Semantic-Elevation-Mapping
!pip install -r requirements.txt pynvml pyngrok
# Step C: Build CMake CUDA SIMT Projection Engine
!mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
# Step D: Launch App Server with Ngrok Public Tunnel for Evaluators
from pyngrok import ngrok
public_url = ngrok.connect(8080)
print(f'LIVE DEMO PUBLIC DASHBOARD URL: {public_url}')
!python3 app.py --port 8080
```

### Option 3: Low-Cost Dedicated Cloud GPUs (RunPod / Massed Compute)

For non-stop 36-hour hackathon operations where Colab timeouts could disrupt live
judging, renting a dedicated cloud GPU instance (e.g., RunPod or Massed Compute) costs
approximately $0.10–$0.20/hr. This provides a persistent root SSH terminal,
unrestricted port forwarding for RTSP streams, and high-performance CUDA compilation.

### Option 4: NVIDIA LaunchPad & Student AI Grants

NVIDIA LaunchPad provides short-term free access to enterprise AI infrastructure,
including pre-configured Jetson Developer Kit sandboxes and DGX systems. Students
participating in official hackathons like SIH can apply for instant trial access to
test TensorRT INT8 model compilation directly on native ARM64 / Jetson execution
stacks.

---

## 3. Smartphone Camera Integration (Setup Options A & B)

To test live perception without an expensive physical camera sensor, your smartphone
can stream live HD video into the 2.5D mapping pipeline using either method below:

### Option A: Zero App Download (HTML5 Browser Camera Stream)

Because the codebase features an embedded WebGL dashboard and WebSocket server
(app.py / yolo_vision_server.py), you can turn any smartphone into a live camera feed
without installing any external application:

**Browser WebRTC Workflow**
1. Open the public server link (e.g., the ngrok URL generated by Colab) in Safari or
   Chrome on your phone.
2. Click 'Enable Camera' when prompted by the browser (uses HTML5
   `navigator.mediaDevices.getUserMedia`).
3. The browser captures 1280x720 video frames and streams them over WebSockets to the
   Python backend.
4. The backend runs spatial horizon gating, YOLOv8 object detection, and 2.5D CUDA
   elevation binning, rendering an AR Heads-Up Display directly back to your phone
   screen.

### Option B: Free IP Camera App (RTSP / HTTP Video Stream over Wi-Fi)

If you require full control over camera exposure, focus locking, or low-light ISO
settings during live physical testing, stream video via RTSP/HTTP using a free mobile
app:

**RTSP / IP Camera App Workflow**
1. Download App: Install IP Webcam (Android) or RTSP Camera (iOS).
2. Network Link: Connect your phone and laptop/board to the same Wi-Fi mobile hotspot.
3. Start Stream: Tap 'Start Server' in the app to display your stream address (e.g.,
   `http://192.168.1.50:8080/video`).
4. Launch Python Processor: Run the camera foveated processor pointing to the stream
   URL:

```
python python/camera_foveated_processor.py --source 'http://192.168.1.50:8080/video' --width 1280 --height 720
```

---

## 4. Live Telemetry & AR HUD Overlay Features (Cloud/Simulated Tier)

When running this live smartphone demo for SIH evaluators, the master dashboard
displays a multi-layered perception teleoperation interface:

- **Spatial Horizon Masking**: A translucent green/red HUD overlay demonstrates
  automatic horizon gating, cropping non-drivable regions (35% sky, 15% hood) to save
  GPU cycles.
- **YOLOv8 Dynamic Detection**: Pedestrians, vehicles, and obstacle categories are
  bounded in real time with depth-scaled distance tags in meters.
- **Concentric 2.5D BEV Map**: Side-by-side WebGL view showing the 3-Ring Elevation
  Grid (Near 5 cm, Mid 15 cm, Far 50 cm) updating dynamically.
- **Real-time Hardware Telemetry**: Live screen widgets displaying VRAM Usage (< 190
  MB) and Perception Latency (< 11.2 ms / ~90 FPS).

---

## 5. Evaluator Pitch Strategy (Answering Hardware Questions)

**Recommended Verbal Pitch Response for SIH Judges**

"To rigorously validate our system without physical hardware constraints, we deployed
and benchmarked our CUDA parallel SIMT kernels and TensorRT INT8 models in a cloud GPU
environment paired with a live smartphone WebGL teleoperation stream. This proves our
pipeline's real-time latency (< 11.2 ms) and micro-memory footprint (< 190 MB VRAM),
guaranteeing seamless zero-copy deployment on physical edge hardware such as the
NVIDIA Jetson AGX Orin."

*(Upgrade: if a physical Jetson was actually connected and tested before the demo —
see Section 6 — replace "guaranteeing seamless...deployment" with "as confirmed by our
own on-device benchmark on a Jetson Orin Nano, achieving X ms latency and Y W power
draw," which is a substantially stronger claim to a technical judging panel.)*

---

## 6. Missing Components Identified — Add These

The original document covers cloud simulation and phone-camera input well, but is
missing the pieces needed to actually **connect the software to a real Jetson board**
and to real sensors. The following components should be added to the project:

1. **A Jetson-targeted build path** (CMake toolchain file for aarch64 cross-compile,
   or native build instructions run directly on the board).
2. **A hardware I/O layer** for real sensors (USB/CSI camera, USB/Ethernet LiDAR) —
   the current doc only covers phone-camera streaming, which is a webcam substitute,
   not a path to a real LiDAR sensor.
3. **A JetPack/L4T environment setup script** — JetPack version, CUDA/cuDNN/TensorRT
   versions must match the board's L4T release; this isn't specified anywhere yet.
4. **A ROS 2 wrapper (optional but recommended)** if the grid engine is to integrate
   with a real vehicle's existing sensor stack, since most real LiDAR/camera drivers on
   Jetson boards publish via ROS 2 topics.
5. **On-device telemetry collection** (`tegrastats`, `jtop`) instead of only the
   `pynvml`-based VRAM reading used in the cloud/Colab path — `pynvml` does not work on
   Jetson's integrated GPU, a different telemetry source is required.
6. **A power/thermal test plan** — Jetson boards throttle under sustained load; a
   30–60 minute soak test should be run before the demo day, not just a quick check.
7. **A network fallback plan** for the physical demo (no reliance on venue Wi-Fi if the
   board is doing local inference — Section 9 covers this).

---

## 7. Connecting the Software to an NVIDIA Jetson Board

### 7a. Physical connection options

| Sensor / Link | Jetson Port Used | Notes |
| --- | --- | --- |
| USB webcam / USB LiDAR (e.g., RPLidar) | USB 3.0 port | Simplest option, works with almost any Jetson model, no driver compilation needed for UVC webcams |
| CSI camera (Raspberry Pi Camera v2/HQ, IMX219/IMX477) | MIPI CSI-2 connector | Lower latency and higher FPS than USB, but needs the Jetson-specific `nvarguscamerasrc` GStreamer pipeline |
| Ethernet LiDAR (e.g., Velodyne, Ouster) | RJ45 Ethernet port | Standard for real LiDAR units, uses UDP packet streams, typically ROS 2 driver already exists for these |
| Development access | USB-C / micro-USB (serial console) + SSH over Ethernet/Wi-Fi | Use for flashing, debugging, and remote development before it's running standalone |

### 7b. Software setup on the board (step by step)

```bash
# 1. Flash JetPack (do this on your laptop with NVIDIA SDK Manager, or use a
#    pre-flashed SD card image matching your Jetson model). This installs the L4T
#    OS + CUDA + cuDNN + TensorRT that match the board's hardware.

# 2. First boot — update and install base tooling
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip cmake git libopencv-dev

# 3. Confirm JetPack/CUDA versions actually match what your code expects
dpkg -l | grep nvidia-jetpack
nvcc --version

# 4. Clone the same repo used in the cloud path — no code changes needed here,
#    this is the point of building for aarch64 rather than x86_64
git clone https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping.git
cd Foveated-2.5D-Semantic-Elevation-Mapping
pip3 install -r requirements.txt

# 5. Build the C++/CUDA grid engine natively on the board (cross-compiling from a
#    laptop is possible but native build on-device is simpler and safer for a
#    hackathon timeline)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=aarch64
make -j$(nproc)
```

If cross-compiling from an x86_64 laptop instead of building natively on the board
(useful if the board is slow to compile on), add a toolchain file:

```cmake
# cmake/jetson-aarch64-toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```
then build with:
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/jetson-aarch64-toolchain.cmake
```

### 7c. If the software has no Jetson-facing entry point yet — add one

The cloud/Colab path launches `app.py` pointed at a phone-stream URL. For a real
board, add a dedicated entry script that reads from a local camera device and/or a
real LiDAR driver instead of a network video URL:

```python
# python/jetson_live_processor.py
"""
Jetson-native entry point: reads a local CSI/USB camera (and, if present, a LiDAR
driver publishing point clouds) and feeds frames into the same foveated pipeline
used by camera_foveated_processor.py, without requiring a phone or cloud tunnel.
"""
import argparse
import cv2

def gstreamer_csi_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    # Jetson-specific GStreamer pipeline for CSI cameras (nvarguscamerasrc).
    # USB webcams do not need this — cv2.VideoCapture(device_index) works directly.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={framerate}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-type", choices=["usb", "csi"], default="usb")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lidar-topic", default=None,
                         help="Optional ROS 2 topic for a real LiDAR driver")
    args = parser.parse_args()

    if args.camera_type == "csi":
        cap = cv2.VideoCapture(gstreamer_csi_pipeline(args.device), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.device)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera — check device index/CSI wiring")

    # from here, frames feed into the same foveated grid pipeline used elsewhere
    # in the codebase (horizon masking -> detection -> 2.5D grid insertion)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # process_frame(frame)  # hook into existing pipeline function
        # publish/update dashboard state here

if __name__ == "__main__":
    main()
```

For a real LiDAR sensor (not a webcam substitute), the simplest integration path on
Jetson is via ROS 2, since most LiDAR vendors already ship a ROS 2 driver:

```python
# ros2_ws/src/foveated_grid_bridge/foveated_grid_bridge/lidar_bridge_node.py
"""
Minimal ROS 2 node that subscribes to a real LiDAR driver's PointCloud2 topic and
feeds raw points into the existing grid-engine Python bindings.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class LidarBridgeNode(Node):
    def __init__(self):
        super().__init__('foveated_grid_lidar_bridge')
        self.subscription = self.create_subscription(
            PointCloud2, '/velodyne_points', self.on_cloud, 10)

    def on_cloud(self, msg: PointCloud2):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        # feed `points` into the existing grid engine insertion function here

def main():
    rclpy.init()
    node = LidarBridgeNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### 7d. Run it standalone (no cloud, no phone)

```bash
# On the Jetson board itself, over SSH or a locally attached monitor+keyboard:
python3 python/jetson_live_processor.py --camera-type csi --device 0
# In a second terminal, start the dashboard server pointed at localhost:
python3 app.py --port 8080 --source local
```
Judges viewing on a laptop connected to the same local hackathon-venue network (or
directly via the board's HDMI output) can then see the dashboard without depending on
Colab, ngrok, or any outside internet connection — an important reliability upgrade
over the cloud-only path in Sections 1–4.

---

## 8. What Upgrades Once a Real Jetson Is Connected

| Before (cloud/simulated) | After (real Jetson + real sensor) |
| --- | --- |
| Latency measured on a T4/L4 cloud GPU, not representative of edge hardware | True on-device latency (typically higher than cloud T4, since Jetson has less raw compute — report this honestly) |
| VRAM read via `pynvml` (only works on discrete NVIDIA GPUs) | Must switch to `tegrastats` or `jtop` (`jetson-stats` package) for shared-memory GPU telemetry |
| Phone camera as a webcam substitute (good FOV/exposure, but not a real automotive sensor) | Real CSI/USB camera and, ideally, a real LiDAR unit — real sensor noise, real FOV overlap, real calibration needed |
| No power/thermal constraint (cloud GPU has no thermal throttling in a 5-minute demo) | Power draw (Watts) and thermal throttling become real constraints — must soak-test |
| Network-dependent (ngrok tunnel, phone Wi-Fi) | Can run fully offline/local-network, removing a demo-day failure point |
| Simulated "real-time" numbers only prove the algorithm is efficient | Proves actual edge-deployability — the claim judges care most about for an autonomous-vehicle pitch |

---

## 9. Frontend Changes After Connecting the Board

The dashboard itself (Streamlit, per the earlier project decision) does **not** need a
rewrite — but a few additions are worth making once real hardware is in the loop:

1. **Telemetry source switch**: replace the `pynvml`-based VRAM widget with a
   `jetson-stats` (`jtop`) based reader when running on-device; keep both code paths
   behind a `--platform cloud|jetson` flag so the same dashboard file works in both
   demo tiers without duplicating UI code.
2. **New "Power Draw (W)" and "SoC Temp (°C)" metric cards** — judges evaluating an
   autonomous-vehicle project will ask about power budget; showing it live is a strong
   signal you tested for real deployment constraints, not just cloud benchmarks.
3. **Source indicator badge**: a small label on the dashboard ("LIVE: Jetson Orin Nano
   — CSI Camera" vs "SIMULATED: Colab T4 — SemanticKITTI replay") so judges always know
   which tier they're watching — this is good practice and prevents any appearance of
   overstating what's live vs. simulated.
4. **No change needed** to the 3-ring BEV grid visualization, the color-coding, or the
   comparison-table view from the earlier roadmap — those are platform-agnostic since
   they only consume the grid engine's output format, not the sensor source.

---

## 10. Recording a Hybrid Real-World + Simulation Demo

Since judges are in a room and the vehicle/LiDAR setup is likely a benchtop rig rather
than a moving car, the cleanest way to record and present a combined real+simulated
demo:

1. **Two synchronized recording sources:**
   - Screen-capture the dashboard (OBS Studio or `ffmpeg -f x11grab` / built-in Windows
     Game Bar) showing live telemetry and the 3-ring BEV map while the Jetson is
     actually processing a real camera/LiDAR feed on the bench.
   - A second phone/webcam recording of the **physical rig itself** (camera/LiDAR unit
     sitting on a desk, or mounted on a small cart/RC platform being moved by hand) so
     judges can see the real sensor and the dashboard reacting to it side by side.
2. **Picture-in-picture edit** (any simple video editor, or even OBS's built-in scene
   compositor while recording live) — physical rig footage as the main frame, dashboard
   as a corner inset, so both are visible in one video without needing judges to look
   back and forth between a screen and a desk.
3. **Always have the Tier-1 recorded fallback ready** (per the earlier roadmap
   decision) — if the Jetson, camera, or venue network fails live, the pre-recorded
   run on the golden test scene (SemanticKITTI clip) plays instantly with no visible
   scramble.
4. **Live demo order recommendation:**
   - Open with the analogy + Tier 1 (laptop, SemanticKITTI replay) — guaranteed to work,
     establishes the core idea.
   - Move to Tier 2 physical rig (if stable and tested the day before) as the "wow"
     closer — real camera, real Jetson, real numbers on screen.
   - Never open with the physical rig — if it has any hiccup in the first 30 seconds,
     it colors the rest of the demo; save it for after credibility is already
     established with Tier 1.
5. **Backup recording, not just backup live demo**: record a full successful Tier 2 run
   the day before presenting, even if you intend to do it live — this becomes your
   fallback video specifically for the physical-hardware portion, separate from the
   Tier 1 fallback already planned.
