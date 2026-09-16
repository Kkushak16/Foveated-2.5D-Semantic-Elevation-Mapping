# 🛰️ Foveated 2.5D LiDAR & Camera Perception System
### Real-Time Semantic Elevation Mapping, 3D World Simulation & Multi-Object Visual Telemetry

[![GitHub Actions CI](https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping/actions)
[![Deploy Pages](https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)](https://en.cppreference.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Iron-orange.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A high-throughput, multi-sensor perception pipeline that converts dense, noisy 3D LiDAR point clouds and camera video feeds into an **adaptive 2.5D semantic elevation grid** in real time. 

Inspired by **human foveated vision**, the system allocates maximum compute and resolution to the immediate driving corridor while progressively downsampling distant space — slashing memory consumption by **~82%** and GPU inference overhead by **~79%**. Includes a rich **WebGL 3D World Simulator** featuring autonomous pure-pursuit driving, 2nd-order suspension physics with dynamic body roll, continuous traffic loops, and dual-sensor teleoperation.

---

## 📸 System Visual Gallery & Simulation Showcase

| **1. 3D Virtual World Simulation (Pure-Pursuit Autopilot & 2nd-Order Body Roll)** |
|:---:|
| ![3D Virtual World Simulation](docs/images/sim_3d_world_live.png) |
| *Real-time WebGL 3D simulation: Autonomous vehicle with Newtonian spring-damper suspension physics and dynamic body roll, navigating dark asphalt highway lanes with 3 concentric holographic LiDAR rings (Near 10m, Mid 30m, Far 70m), continuous cruising traffic, and sidewalk pedestrians.* |

| **2. Dual-Sensor Teleoperation Split View** | **3. Multi-Ring LiDAR Bird's-Eye View (BEV)** |
|:---:|:---:|
| ![Dual-Sensor Teleop Split View](docs/images/dual_split_live.png) | ![LiDAR BEV Grid](docs/images/lidar_bev_grid_live.png) |
| *Dual-Split teleoperation: Synchronized 3-ring LiDAR BEV grid (left) alongside live camera foveation with near/mid/far depth perspective bands and sky/hood spatial masking (right).* | *Full 2.5D BEV elevation grid: 3 concentric resolution tiers (Near 5cm, Mid 15cm, Far 50cm), bounding box classification, distance badges, and negative obstacle (pothole) markers.* |

| **4. Camera Foveation with Active ROI & Depth Bands** | **5. Interactive Teleoperation Landing Interface** |
|:---:|:---:|
| ![Camera Foveation](docs/images/camera_foveation_live.png) | ![Landing Hero Interface](docs/images/landing_hero.png) |
| *Spatial horizon gating: Discards static non-road pixels (sky 35% and hood 15%) while applying optical-flow motion gating and depth-scaled detection boxes to dynamic obstacles.* | *Interactive browser HUD interface featuring instant view mode switching, live camera feed selection, LiDAR point cloud controls, and real-time compute savings telemetry.* |

---

## 💡 What Is This & What Does It Do?

Autonomous vehicles and mobile robots need to perceive their immediate environment at 30+ frames per second. However, modern 3D LiDARs (Velodyne, Ouster, Hesai, Livox) fire **100,000 to 1,000,000+ points every second**. Running heavy 3D deep neural networks over this raw point volume causes high latency, thermal throttling, and dropped frames on edge compute hardware (such as NVIDIA Jetson Orin or Xavier).

This software solves that bottleneck:
1. **Separates Ground from Obstacles**: Rapidly extracts drivable road surfaces using plane-fitting algorithms (Patchwork++ / RANSAC).
2. **Projects 3D Points into an Adaptive 2.5D Ego-Grid**: Uses high-speed CUDA kernels ($< 1.8\text{ ms}$) or optimized C++ ring buffers to project 3D point clouds into concentric elevation cells.
3. **Aligns Camera Vision via Dual Foveation**: Eliminates static non-road pixels (sky and vehicle hood), extracting features and tracking objects (cars, pedestrians, cyclists) only where obstacles can realistically appear.
4. **Publishes Planning-Ready Maps**: Emits standardized `nav_msgs/msg/OccupancyGrid` messages to ROS 2 navigation stacks and streams WebSocket telemetry to lightweight WebGL browser dashboards.
5. **Simulates Full Autonomous Environments**: Built-in 3D simulation with dynamic vehicle physics, pure-pursuit path tracking, sidewalk pedestrians, and continuous traffic loops.

---

## 🤔 Why 2.5D LiDAR Instead of 2D or Full 3D?

Perception systems typically choose between 2D flat costmaps or 3D voxel representations. A **2.5D Semantic Elevation Grid** delivers the optimal balance:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 3D Point Cloud Volume                   │
                  │             (1,000,000 pts/sec - Heavy)                 │
                  └────────────────────────────┬────────────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
┌──────────────────────────────┐                              ┌──────────────────────────────┐
│       2D Flat Grid           │                              │        3D Voxel Grid         │
│  (Binary Free / Occupied)    │                              │     (OctoMap / SparseConv)   │
├──────────────────────────────┤                              ├──────────────────────────────┤
│ ❌ Blind to step curbs       │                              │ ❌ Huge memory: O(N³)        │
│ ❌ Blind to potholes / drops │                              │ ❌ High latency (50-200ms)   │
│ ❌ Blind to overhanging trees│                              │ ❌ High VRAM & power draw    │
│ ✅ Very fast compute         │                              │ ✅ Full 3D geometric detail  │
└──────────────────────────────┘                              └──────────────────────────────┘
               │                                                               │
               └───────────────────────────────┬───────────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │  ⭐ 2.5D Foveated Elevation Grid │
                              │          (Our Solution)          │
                              ├──────────────────────────────────┤
                              │ ✅ Memory: O(N²) (2D array speed)│
                              │ ✅ Measures true curb & step Z   │
                              │ ✅ Detects potholes & drop-offs  │
                              │ ✅ Handles bridges & overhangs   │
                              │ ✅ Latency: < 2ms (Real-Time)    │
                              └──────────────────────────────────┘
```

### The 3-Ring Concentric Multi-Level Ring Buffer (MLRB)

Instead of a uniform grid that wastes memory resolving empty distant pavement, our grid uses **3 concentric rings centered on the vehicle**:

| Ring Level | Distance Range | Cell Resolution | Grid Dimensions | Primary Perception Focus |
|:---|:---:|:---:|:---:|:---|
| **Ring 0 (Near)** | $0 - 10\text{ m}$ | **$5\text{ cm}$** | $400 \times 400$ | **Curbs, potholes, debris, pedestrian legs, wheel contact** |
| **Ring 1 (Mid)**  | $10 - 30\text{ m}$ | **$15\text{ cm}$** | $400 \times 400$ | **Vehicles, cyclists, lane boundaries, traffic barriers** |
| **Ring 2 (Far)**  | $30 - 100\text{ m}$ | **$50\text{ cm}$** | $400 \times 400$ | **Macro-terrain, highway corridors, road curvature** |

Each grid cell stores a compact **Struct-of-Arrays (SoA)**:
* `min_z` / `max_z`: Minimum and maximum point heights in the cell.
* `ground_z`: Kalman-filtered estimated road height.
* `z_variance`: Surface roughness / traversability metric.
* `sem_class`: Semantic label (`ground`, `vehicle`, `pedestrian`, `vegetation`, `obstacle`).
* `confidence`: Bayesian occupancy and detection probability.

> **How Pothole Detection Works**:
> A pothole is a **negative obstacle** (a road depression 5–15 cm below surface grade). While a 2D camera sees only dark pixels, our Near-Ring LiDAR cells detect when $\text{min\_z} < \text{ground\_z} - \text{threshold}$, immediately flagging a road surface cavity on the teleoperation HUD.

---

## 🏗️ System Architecture

```
                                [ Sensor Inputs ]
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
             3D LiDAR Point Cloud                 Automotive Camera Feed
          (UDP / ROS 2 PointCloud2)                (V4L2 / RTSP / Webcam)
                       │                                   │
                       ▼                                   ▼
         [ Ground Surface Filter ]               [ Horizon Spatial Mask ]
          Patchwork++ / RANSAC Plane            Excludes Sky (35%) & Hood (15%)
                       │                                   │
                       ▼                                   ▼
       [ Custom CUDA Projection Kernel ]       [ Semantic Motion / YOLO Gating ]
         Parallel 3D-to-2.5D Binning            YOLOv8 / Optical Flow Tracking
             (<1.8 ms @ 100k pts)               (Tracks Persons, Cars, Parts)
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         ▼
                        [ Sensor Fusion & Ring Engine ]
                         Concentric MLRB Ring Buffers
                         LiDAR 0.70  |  Camera 0.30
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [ ROS 2 Middleware ]                        [ WebGL HUD Dashboard ]
      nav_msgs/msg/OccupancyGrid                   Low-Latency Teleoperation
     /planning/foveated_ego_grid                   (Browser / Streamlit / WS)
```

---

## 🏎️ Autonomous Vehicle Physics & Simulation Engine

The included 3D world simulator runs a realistic vehicle model with continuous physics:

* **Pure Pursuit Lane Following**: Monotonically advancing lookahead waypoint pursuit tracking the center of the right driving lane ($R = 37.5\text{m}$). Calculates exact curvature $\delta = \text{atan2}(2 L_w \cdot \text{lateral}, L^2)$ with speed-adaptive lookahead ($7.5\text{m} - 16.0\text{m}$) for smooth cornering.
* **2nd-Order Spring-Damper Body Roll**: Realistic chassis lean outward during cornering:
  $$\ddot{\theta}_{\text{roll}} = -k_{\text{roll}}(\theta_{\text{roll}} - \theta_{\text{target}}) - c_{\text{roll}}\dot{\theta}_{\text{roll}}$$
  where $\theta_{\text{target}} = \text{clamp}(-a_{\text{lat}} \cdot 0.038, -0.14, 0.14\text{ rad})$ ($k_{\text{roll}} = 92.0$, $c_{\text{roll}} = 13.5$).
* **Pitch Dive & Squat**: Chassis pitches forward during braking (dive up to $-0.09\text{ rad}$) and squats during acceleration ($+0.06\text{ rad}$).
* **Dynamic Wheel Camber**: Front wheels tilt slightly into the turn direction when steered.
* **Continuous Traffic Loop**: Cruising sedans and SUVs loop continuously around a 500-meter closed circuit without despawning or teleporting.
* **Sidewalk Pedestrians**: 12 articulated pedestrians walking along designated sidewalks ($X = \pm 43.5\text{m}$), park paths, and crossing at dedicated zebra crosswalks ($Z = \pm 20\text{m}$).
* **Zero Phantom Walls**: Dual-disk vehicle capsules ($1.9\text{m}$ clearance) eliminate phantom collisions next to parked vehicles.

---

## 🎮 Interactive Controls & Cheatsheet

When running the interactive teleoperation interface, the vehicle and cameras can be controlled via keyboard:

| Control | Key | Action |
|:---|:---:|:---|
| **Throttle / Accelerate** | `W` or `↑` | Accelerate ego vehicle forward |
| **Reverse** | `S` or `↓` | Drive ego vehicle in reverse |
| **Steer Left** | `A` or `←` | Steer wheels left (non-mirrored) |
| **Steer Right** | `D` or `→` | Steer wheels right (non-mirrored) |
| **Handbrake** | `Space` | Apply emergency braking with reactive taillights |
| **Toggle Autopilot** | `P` | Switch between Autonomous Lane-Following & Manual Control |
| **Cycle Cameras** | `C` | Switch camera between **Chase**, **Cockpit**, and **Overhead** |
| **View Modes** | UI Buttons | Switch between **3D World**, **Dual-Sensor Split**, **LiDAR BEV**, and **Camera Foveation** |
| **Sensor Sources** | UI Dropdown | Select **3D Sim Windshield**, **Live Physical Webcam**, or **Synthetic Benchmark** |

---

## ⚙️ Complete Setup & Installation Guide

### Prerequisites
* **Operating System**: Linux (Ubuntu 20.04 / 22.04 LTS) or Windows 10/11.
* **Python**: Python 3.10+ (with `pip` and virtual environment support).
* **C++ Compiler**: GCC 9+, Clang 11+, or MSVC 2019+ (C++17 standard required).
* **CMake**: Version 3.18 or higher.
* **Node.js**: v18+ (optional, for the static server and WebSocket telemetry bridge).
* **CUDA / GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ / 12.x and TensorRT 8.x. *(CPU fallback is included automatically if no GPU is detected).*

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping.git
cd Foveated-2.5D-Semantic-Elevation-Mapping
```

---

### Step 2: Instant Web Simulation (No Heavy Dependencies)
You can immediately launch and explore the 3D world simulation and telemetry dashboard using any local static server:

#### Option A: Using Node.js
```bash
node -e "const http=require('http'),fs=require('fs'),path=require('path');const mime={'.html':'text/html','.js':'application/javascript','.css':'text/css','.png':'image/png'};http.createServer((req,res)=>{let f=path.join('web/ui',req.url.split('?')[0]);if(f.endsWith('/')||fs.statSync(f,{throwIfNoEntry:false})?.isDirectory())f=path.join(f,'index.html');if(fs.existsSync(f)){res.setHeader('Content-Type',mime[path.extname(f)]||'text/plain');fs.createReadStream(f).pipe(res);}else{res.statusCode=404;res.end('Not found');}}).listen(8080,()=>console.log('Dashboard ready at http://localhost:8080'));"
```

#### Option B: Using Python
```bash
python -m http.server 8080 --directory web/ui
```

Then open your browser and navigate to:
👉 **`http://localhost:8080`**

---

### Step 3: Python Machine Learning & Vision Environment
For running the YOLO semantic detector, WebSocket telemetry bridge, and offline evaluation pipeline:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Launch Python Vision & Telemetry Server:
```bash
# Unified entrypoint for local dashboard + Streamlit integration:
python app.py

# Or run the dedicated YOLO WebSocket vision server:
python python/yolo_vision_server.py --port 8765 --conf 0.45
```

---

### Step 4: Build the High-Performance C++ / CUDA Core
The build system uses CMake. If `nvcc` is detected on your system, CUDA SIMT projection kernels are compiled automatically; otherwise, the native multi-threaded C++17 CPU fallback is built:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

#### Run Unit & Integration Tests:
```bash
ctest --output-on-failure
```

---

### Step 5: ROS 2 Humble / Iron Workspace Setup
If you are running the system inside an autonomous vehicle software stack with ROS 2:

```bash
cd ros2_ws
colcon build --packages-select vehicle_detection_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# Run our foveated projection node:
ros2 run vehicle_detection_ros2 foveated_vehicle_detect_node \
  --ros-args -r /sensing/lidar/top/pointcloud_raw:=/velodyne_points
```

Output topics published:
* `/planning/foveated_ego_grid` (`nav_msgs/msg/OccupancyGrid`)
* `/perception/elevation_markers` (`visualization_msgs/msg/MarkerArray`)

---

## 🔌 Hardware Sensor Setup & Physical Testing

You can run this software with **Waveshare WAVE ROVER (ESP32 + Raspberry Pi CPU)**, **NVIDIA Jetson boards**, **live physical LiDARs**, **webcam / CSI cameras**, or **offline open datasets**.

```
                           [ Physical Hardware Deployments ]

   Option A: Low-Cost / CPU (No GPU)             Option B: High-Performance / GPU
   ┌───────────────────────────────┐             ┌───────────────────────────────┐
   │     Waveshare WAVE ROVER      │             │   NVIDIA Jetson Orin / Nano   │
   │  • Raspberry Pi 4/5 (CPU)     │             │  • Ampere GPU / TensorRT      │
   │  • Built-in ESP32 (Motors)    │             │  • CUDA Kernel Grid Binning   │
   │  • RPLiDAR / LD19 (USB)       │             │  • 3D LiDAR (Velodyne/Ouster) │
   └───────────────────────────────┘             └───────────────────────────────┘
```

---

### 1. Waveshare WAVE ROVER Setup (ESP32 + Raspberry Pi CPU — Low-Cost / Zero GPU)

For autonomous rovers, budget prototypes, and hackathons (e.g. Smart India Hackathon), this project runs entirely on **pure CPU** without expensive GPUs using the **Waveshare WAVE ROVER**.

👉 **Full Step-by-Step Guide & Precautions:** [WAVE_ROVER_SETUP.md](WAVE_ROVER_SETUP.md)

```
       [ 360° LiDAR (RPLiDAR / LD19) ]          [ CSI Pi Camera / USB Cam ]
                   │ (USB Cable)                               │ (Ribbon / USB)
                   ▼                                           ▼
       ┌──────────────────────────────────────────────────────────────┐
       │             Raspberry Pi 4 / 5 (Pure CPU Brain)              │
       │   • Foveated 2.5D Concentric Grid Engine (<2ms latency)      │
       │   • Real-Time Obstacle Avoidance Safety Bubble               │
       │   • Live MJPEG Video Streamer (Port 8081)                    │
       │   • Web Dashboard Server (Port 8080)                         │
       └──────────────────────────────┬───────────────────────────────┘
                                      │ UART (/dev/ttyS0) or USB (/dev/ttyUSB0)
                                      ▼ 115200 Baud JSON Protocol
       ┌──────────────────────────────────────────────────────────────┐
       │             Wave Rover Onboard ESP32 Sub-Controller          │
       │   • Dual Closed-Loop PID Motor Drivers (Left / Right)        │
       │   • Optical Wheel Encoders (Hardware Interrupts)             │
       │   • INA219 Voltage & Current Sensor (3x 18650 Batteries)     │
       └──────────────────────────────────────────────────────────────┘
```

#### Quick Start with Hardware Integration Engine:
The frontend includes a built-in **Hardware Integration Engine** accessible at `http://localhost:8080#hardware` (or via the top navigation **"🔌 Connect Hardware"** button):

1. **Step 1 · Choose Hardware:**
   - **Arduino Uno Q (Recommended):** High-precision PWM motor drive, dual encoder tracking, and USB-C connectivity.
   - **Raspberry Pi 4 / 5:** High-performance Linux SBC running the Python telemetry bridge.
   - **Arduino Uno R3 / R4 WiFi:** Classic microcontroller with motor shield.
   - **ESP32 (Wave Rover Onboard):** Factory dual-core MCU connected via Micro-USB or Wi-Fi AP.
   - **Custom Robot:** Any differential drive chassis communicating via 115200 baud JSON serial.
   - **Auto-Detect:** Click `⚡ Scan & Auto-Connect` to probe local ports and launch the session.

2. **Step 2 · Interactive Setup Guide & Pinouts:**
   - Interactive board switcher tabs to view tailored wiring diagrams, pin assignments, and firmware commands.
   - Exact pin mapping for motors, encoders, and thermal sensors.
   - 1-click compile and upload instructions using `arduino-cli` / `platformio`.

3. **Step 3 · Connect & Live Dashboard:**
   - **Low-Latency Video Feed:** Streaming camera feed with Qwen3-VL neural detection HUD.
   - **2.5D Concentric Ring BEV:** Near (0–10m), Mid (10–30m), Far (30–100m) occupancy grid.
   - **Live Telemetry:** Stream FPS (~30 FPS), Roundtrip Ping (~2–4 ms), Board/SoC Temperature (°C), Battery Voltage (V).
   - **Direct Drive Teleop:** Interactive on-screen WASD buttons and keyboard arrow keys with emergency stop.
   - **Simulation Mode Fallback:** Automatically active when physical bridge is offline so all UI elements and driving controls can be tested immediately.

#### Hardware Diagnostics & COM Verification Script:
Before launching the web dashboard, you can verify your USB-connected Arduino or microcontroller using the automated diagnostic tool:
```bash
python scripts/test_arduino_board.py
```
This script checks `pyserial`, auto-scans active COM/serial ports, tests the 115200 baud handshake with `{"T":1001}`, and verifies telemetry packet reception.

#### Pure-Vision Mapping Without LiDAR (Qwen3-VL):
Physical LiDAR hardware is **100% optional**. If a LiDAR sensor is not connected, the system engages **Qwen3-VL ("Sharper Vision, Deeper Thought, Broader Action")** to perform monocular spatial depth reasoning, construct the concentric occupancy grid from camera frames, and navigate autonomously along collision-free paths.

#### Starting the Python Bridge (Physical Hardware):
```bash
# Windows / Linux / macOS
python python/waverover_bridge.py --port COM3   # or /dev/ttyACM0 / /dev/ttyUSB0
```

---

### 2. NVIDIA Jetson Platform Setup (Orin / Xavier / Nano — CUDA Accelerated)

For production autonomous vehicles and edge computing with hardware GPU acceleration:

```bash
# 1. Automated JetPack 5.1 / 6.0 Environment Setup
chmod +x scripts/jetson_setup.sh
./scripts/jetson_setup.sh

# 2. Build Native C++ & CUDA Kernels
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
make -j$(nproc)

# 3. Launch Native Jetson Ingestion (CSI Camera via nvarguscamerasrc + ROS 2 LiDAR)
python3 python/jetson_live_processor.py --camera-type csi --device 0 --lidar-topic /velodyne_points
```

* **Live Hardware Telemetry:** Runs `python/telemetry.py` to stream real-time Tegra metrics (GPU utilization %, VRAM MB, SoC power draw in Watts, and thermal sensor °C) directly into the HUD.
* **Cross-Compilation:** A dedicated CMake toolchain for ARM64 Jetson targets is available at `cmake/jetson-aarch64-toolchain.cmake`.

---

### 3. Connecting Physical LiDAR Hardware (Ethernet UDP)

1. **Network Interface Configuration**:
   * Connect the sensor RJ45 cable to your computer or vehicle switch.
   * Assign a static IP on the sensor's subnet (e.g., LiDAR IP: `192.168.1.201`, Computer IP: `192.168.1.100`, Subnet: `255.255.255.0`).
2. **Launch the Native C++ UDP Ingestion Driver**:
   ```bash
   ./build/foveated_lidar_onboard_node --port 2368 --rate 10
   ```
3. **Supported LiDAR Sensors**:
   * **Velodyne**: VLP-16, Puck LITE, Ultra Puck VLP-32C, Alpha Prime VLS-128
   * **Ouster**: OS0-32/64/128, OS1-32/64/128, OS2-64/128
   * **Livox**: Mid-360, Mid-70, HAP (via `livox_ros_driver2`)
   * **Hesai**: Pandar40P, PandarXT-32, QT64

### 4. Connecting Physical Cameras (USB / CSI / RTSP)

* **Webcam / USB Camera**:
  Select **"📷 Live Camera (Physical Webcam)"** from the web UI control panel, or configure device index in Python:
  ```bash
  python python/camera_foveated_processor.py --source 0 --width 1280 --height 720
  ```
* **RTSP IP Camera Stream**:
  ```bash
  python python/camera_foveated_processor.py --source "rtsp://admin:pass@192.168.1.64:554/h264Preview_01_main"
  ```

### 5. Testing with SemanticKITTI Datasets
Download sample `.bin` point clouds from SemanticKITTI and run our offline validation tool:
```bash
python python/run_offline_pipeline.py --scan my_dataset/000000.bin
```

---

## 📊 Telemetry & Performance Benchmarks

All benchmarks measured on **NVIDIA Jetson AGX Orin (64GB, 50W Mode)** and **Intel Core i7-12700H**:

| Pipeline Stage | Implementation | Input Volume | Execution Latency | Memory Footprint |
|:---|:---:|:---:|:---:|:---:|
| **Ground Segmentation** | Patchwork++ (C++) | 120,000 pts/frame | **$3.12\text{ ms}$** | $14\text{ MB}$ |
| **CUDA Grid Binning** | Parallel SIMT Kernel | 120,000 pts/frame | **$1.74\text{ ms}$** | $42\text{ MB}$ |
| **CPU Grid Fallback** | Multi-threaded C++17 | 120,000 pts/frame | **$14.20\text{ ms}$** | $18\text{ MB}$ |
| **Camera Spatial Mask** | Geometric ROI Gating | 1080p @ 60 FPS | **$0.42\text{ ms}$** | $4\text{ MB}$ |
| **YOLOv8n Inference** | TensorRT INT8 Graph | $640 \times 640$ ROI | **$4.85\text{ ms}$** | $110\text{ MB}$ |
| **End-to-End Frame Time** | **Complete Fused Pipeline** | **Dual Sensor** | **$\mathbf{< 11.2\text{ ms}}$** *(90 FPS)* | **$\mathbf{< 190\text{ MB}}$** |

---

## 📁 Repository Structure

```
Foveated-2.5D-Semantic-Elevation-Mapping/
├── CMakeLists.txt                # Root CMake build with auto CUDA/CPU detection
├── app.py                        # Unified entrypoint for local web server & Streamlit
├── requirements.txt              # Python dependencies (NumPy, OpenCV, WebSockets, Ultralytics)
├── .github/workflows/
│   ├── cmake-single-platform.yml # Continuous Integration (ctest on Ubuntu)
│   └── deploy-pages.yml          # Automated GitHub Pages web dashboard deployment
├── docs/
│   └── images/                   # High-res simulation captures & system architecture diagrams
│       ├── sim_3d_world_live.png     # 3D city world with pure pursuit & body roll
│       ├── dual_split_live.png       # Dual-sensor split view (LiDAR BEV + Camera)
│       ├── lidar_bev_grid_live.png   # Full 3-ring BEV grid with distance labels
│       ├── camera_foveation_live.png # Camera foveation with depth bands & ROI masking
│       └── landing_hero.png          # Interactive HUD landing page
├── cpp/                          # Modern C++ Core (Deterministic, Zero-GC)
│   ├── include/                  # Headers: Ring buffer, LiDAR driver, ROS 2, TensorRT
│   └── src/                      # Low-latency C++ implementations
├── cuda/                         # NVIDIA CUDA Acceleration
│   ├── include/grid_projection.cuh
│   └── src/grid_projection.cu    # Parallel 3D-to-2.5D projection kernels (<1.8ms)
├── arduino/                      # Microcontroller Firmware
│   └── wave_rover_controller/
│       └── wave_rover_controller.ino # 115200 Baud JSON motor control & encoder telemetry
├── scripts/                      # Hardware Diagnostics & Setup Scripts
│   ├── test_arduino_board.py     # Live diagnostic for Arduino Uno Q / COM serial
│   └── jetson_setup.sh           # Automated JetPack environment configuration
├── ros2_ws/                      # ROS 2 Colcon Workspace
│   └── src/vehicle_detection_ros2/
│       └── src/foveated_vehicle_detect_node.cpp  # Multi-threaded ROS 2 node
├── python/                       # Machine Learning, Vision Servers & Offline Pipeline
│   ├── semantic_detector.py      # Multi-backend detector (YOLOv8 / OpenCV HOG cascade)
│   ├── yolo_vision_server.py     # High-speed WebSocket vision server for dashboard
│   ├── test_semantic_vision.py   # Test suite for static persons, vehicles & wheel parts
│   ├── waverover_bridge.py       # Serial-to-HTTP/WS rover telemetry bridge
│   └── camera_foveated_processor.py # 3-ring optical flow & ROI cropping
└── web/                          # Teleoperation Dashboard UI & Bridges
    ├── server/websocket_bridge.js# Node.js HTTP & telemetry WebSocket bridge
    └── ui/                       # HTML5, CSS3, Three.js 3D Simulator & WebGL HUD
        ├── index.html            # Main dashboard interface
        ├── hardware_connect.js   # 3-step hardware connection wizard & live dashboard
        ├── three_simulator.js    # 3D WebGL Simulator with suspension & pure pursuit
        ├── teleop_dashboard.js   # Teleoperation canvas renderers & sensor fusion
        └── three.min.js          # Three.js 3D library
```

---

## 📄 Citation & Attribution

If you use this foveated 2.5D mapping architecture or camera gating algorithms in your academic research or projects, please cite:

```bibtex
@misc{foveated_elevation_mapping_2026,
  author = {Kushak},
  title = {Foveated 2.5D Semantic Elevation Mapping for Autonomous Vehicle Perception},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping}}
}
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
