# 🛰️ Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception

Convert dense 3D LiDAR point clouds into a lightweight, adaptive 2.5D grid
(elevation map + semantic layers) in real time, inspired by human foveated
vision — **high resolution near the vehicle, coarser resolution far away**.

---

## ⚡ Multi-Language Architecture & System Data Flow

```
[ Offline / Training Phase ]
 Python (PyTorch / NumPy) ──► Exports Model (.onnx)
                                    │
                                    ▼
[ Real-Time Onboard System ]
 LiDAR Driver (C++) ──► Point Cloud Pipeline (C++ / CUDA)
                                    │
                                    ▼
 TensorRT Engine (C++ / CUDA) ──► 3D Semantic Classification
                                    │
                                    ▼
 Custom CUDA Kernels (CUDA C++) ──► 3D-to-2.5D Parallel Projection
                                    │
                                    ▼
 Ring Buffer Logic (Modern C++) ──► Multi-Resolution Ego-Grid State
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
 ROS 2 Middleware (C++ rclcpp)               Browser/Desktop UI (JS/TS WebGL)
  (To Motion Planner / Controls)              (Remote Teleop Dashboard)
```

### Detailed Breakdown by Language

| Language / Domain | System Component | Concrete Use Case | Why This Language? | What It Improves & Solves |
|---|---|---|---|---|
| **Python** | Offline Machine Learning & Validation | Training 3D backbone (RandLA-Net / Cylinder3D), calculating loss functions, dataset augmentations, mIoU metrics, ONNX model export. | Rich AI ecosystem (PyTorch, PyTorch Geometric, Open3D-ML); dynamic scripting for fast experimentation. | **Accelerates R&D turnaround**: Modifying model layers or dataset paths takes seconds instead of requiring full C++ recompilation cycles. |
| **Modern C++ (C++17/20)** | Grid Engine Core, ROS 2 Middleware, Sensor Drivers | Maintaining Multi-Level Ring Buffer (MLRB), circular memory wrapping, coordinate transforms, vehicle ego-motion updates, low-latency ROS 2 nodes (`rclcpp`). | Manual deterministic memory layout, direct pointers, cache locality, absence of runtime Garbage Collector (GC) stalls. | **Guarantees deterministic latency**: Eliminates random 10–50 ms frame drops caused by Python GC; enables zero-copy sensor passing in $< 1\text{ ms}$. |
| **CUDA (CUDA C/C++)** | Hardware Acceleration & Parallel Grid Binning | Projecting 100,000–1,000,000 $(x,y,z)$ points simultaneously into 2.5D elevation cells; min/max bounds and atomic class voting. | GPU hardware direct access; thousands of parallel ALUs running single-instruction multi-thread (SIMT). | **Dramatically cuts latency**: Reduces point-to-grid projection time from $\sim 300\text{ ms}$ (CPU loops) to $< 2\text{ ms}$ (GPU kernels), enabling 30+ FPS operation. |
| **C++ TensorRT Runtime** | Deep Learning Inference Engine | Loading compiled network engines (`.plan` / `.engine`) onto onboard automotive hardware (e.g., NVIDIA Orin/Xavier). | Native hardware graph fusion, kernel autotuning, FP16/INT8 hardware quantization. | **Lowers VRAM & compute overhead**: Reduces model inference from $\sim 50\text{ ms}$ in vanilla PyTorch to $< 8\text{ ms}$ in INT8 precision without losing accuracy. |
| **TypeScript / JS (Optional UI)** | Teleoperation Dashboard / Remote Monitor | Web-based visualization (custom WebGL/Three.js interfaces) showing vehicle bird's-eye view live. | Runs in standard web browsers on any client laptop or operations center without local ROS/GPU toolchains installed. | **Decouples monitoring from vehicle compute**: Fleet managers inspect live road conditions over WebSocket streams without draining local car compute. |
| **CMake / Shell (Bash)** | Build & Infrastructure Orchestration | Cross-compiling C++/CUDA codebases, linking PCL/Eigen/TensorRT libraries, containerizing runtime in Docker. | Industry standard build systems for reproducible native binary builds on embedded Linux systems. | **Prevents dependency drift**: Ensures identical compilation flags (`-O3 -march=native -DCUDA_ARCH=87`) between dev laptops and onboard ECUs. |

---

## 📷 Dual-Sensor Foveation: Foveated Camera Processing

To achieve full narrative unity across both vision modalities, the **camera image processing pipeline** directly adopts the same 3-ring foveated geometry as the 3D LiDAR Grid:

1. **Semantic Spatial Masking**: Masks out sky (top 35%) and vehicle hood (bottom 15%) using horizon geometry before running detection algorithms, cutting **~30% of unnecessary pixel compute** with zero ML overhead.
2. **Motion-Compensated Optical Flow Gating**: Uses CPU-friendly Farnebäck optical flow (`cv2.calcOpticalFlowFarneback`) to isolate moving objects. Static background regions reuse cached obstacle classifications, boosting processing throughput by **~2.8x**.
3. **3-Ring Camera Crop Alignment**:
   - **Near Ring (0–10m)**: Processed at **Full Resolution (1.0x)** for critical obstacle detection (curbs, debris, pedestrians).
   - **Mid Ring (10–30m)**: Crop downsampled to **Moderate Resolution (0.5x)** for vehicle tracking.
   - **Far Ring (30–100m)**: Low-resolution glance (**0.25x**), re-triggered only when optical flow detects motion.

---

## 📂 Project Structure

```
Lidar Mapping/
├── CMakeLists.txt                # C++/CUDA build orchestrator
├── build.sh                      # Cross-platform build script
├── Docker/
│   └── Dockerfile                # NVIDIA L4T Docker container
├── python/                       # Offline ML & ONNX Exporting
│   ├── train_and_export_onnx.py  # PyTorch 3D model exporter (.onnx)
│   └── validate_onnx.py          # ONNX Runtime & mIoU validator
├── cpp/                          # Native C++ Real-time Core
│   ├── include/                  # Ring Buffer, LiDAR Driver, TensorRT, ROS 2 headers
│   └── src/                      # Low-latency C++ implementations
├── cuda/                         # CUDA C++ Acceleration
│   ├── include/grid_projection.cuh
│   └── src/grid_projection.cu    # Parallel 3D-to-2.5D projection kernels (<2ms)
├── web/                          # Teleoperation Dashboard (JS/TS WebGL)
│   ├── server/websocket_bridge.js # Node.js WebSocket telemetry bridge
│   └── ui/                       # HTML5/WebGL HUD interface
├── run_multilang_demo.py         # Multi-language pipeline integration runner
├── src/                          # Existing Python reference pipeline
└── README.md                     # ← you are here
```

---

## 🚀 Phase 1 — Ground Segmentation & Dataset Pipeline

### Setup

```bash
# 1. Create a virtual environment (recommended)
cd "Lidar Mapping"
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Patchwork++ for the fast C++ back-end
pip install pypatchworkpp
```

### 1 — Dataset Loader (`dataset_loader.py`)

Loads **SemanticKITTI** `.bin` point cloud files and `.label` files.

```python
from src.dataset_loader import SemanticKITTILoader

loader = SemanticKITTILoader("/path/to/semantickitti")

for seq_id, frames in loader.iter_sequences():
    for frame_id, points, labels in frames:
        # points: np.ndarray (N, 4) — x, y, z, intensity
        # labels: np.ndarray (N,)   — semantic label ID
        print(f"Seq {seq_id}, Frame {frame_id}: {points.shape[0]} points")
        break  # just peek at first frame
```

**CLI quick-test:**
```bash
python src/dataset_loader.py /path/to/semantickitti
```

### 2 — Ground Segmentation (`ground_segmentation.py`)

**Stable interface** consumed by Member B (grid engine) and Member C (dashboard):

```python
from src.ground_segmentation import segment_ground

mask = segment_ground(points)        # points: (N, 4) xyzi
ground_pts     = points[mask]
non_ground_pts = points[~mask]
```

**Back-ends:**

| Back-end       | Install                    | Speed     | Notes                          |
|----------------|----------------------------|-----------|--------------------------------|
| Patchwork++    | `pip install pypatchworkpp` | ~5–10 ms  | C++ via pybind11, preferred    |
| RANSAC         | (built-in, NumPy only)     | ~50–200 ms | Pure Python fallback, CPU-only |

The wrapper auto-detects which back-end is available. Force one with:
```python
mask = segment_ground(points, backend="ransac")
```

### 3 — Validation Script (`validate_ground_seg.py`)

#### Synthetic scene (no dataset needed):
```bash
cd src
python validate_ground_seg.py --synthetic
```

#### Real SemanticKITTI frame:
```bash
cd src
python validate_ground_seg.py /path/to/semantickitti --seq 00 --frame 000000
```

#### Skip visualisation (CI-friendly):
```bash
python validate_ground_seg.py --synthetic --no-vis
```

**Expected output (synthetic scene, RANSAC back-end):**
```
============================================================
  Ground Segmentation — Validation Report
============================================================
  Total points      :     80,000
  Predicted ground   :     44,xxx  (~55%)
  Predicted non-ground:     35,xxx  (~45%)
  Inference time     :        xx.x ms

  Ground-truth ground:     44,000  (55.0%)
  TP: ~43,xxx  FP: ~1,xxx  FN: ~xxx  TN: ~35,xxx
  Precision : 0.97xx
  Recall    : 0.98xx
  F1 Score  : 0.97xx
  IoU       : 0.95xx
============================================================
```

When Open3D is installed, a 3D viewer window opens showing:
- **With ground truth:** TP=green, FP=orange, FN=blue, TN=grey
- **Without ground truth:** Ground=green, Non-ground=red

---

## 🔧 Phase 2 — Obstacle Clustering & Classification

Takes non-ground points from Phase 1 and produces classified obstacle clusters.

### 1 — Clustering (`clustering.py`)

Two back-ends: **DBSCAN** (scikit-learn, preferred) and **voxel-based Euclidean** (pure NumPy fallback).

```python
from clustering import cluster_points

# non_ground_pts: (M, 3+) from points[~ground_mask]
cluster_ids = cluster_points(non_ground_pts, eps=0.5)
# cluster_ids: (M,) — -1 = noise, 0..K = cluster ID
```

### 2 — Feature Extraction (`feature_extraction.py`)

Extracts a **14-dimensional** feature vector per cluster:

| Feature | Description |
|---------|-------------|
| height, width, length | OBB extents (PCA-oriented in XY) |
| aspect_ratio_wl/hw | Shape ratios |
| volume, point_count, point_density | Size & density |
| z_mean, z_variance, z_range | Vertical statistics |
| intensity_mean, intensity_std | Reflectivity |
| linearity | PCA eigenvalue ratio (λ1-λ2)/λ1 |

### 3 — Training (`train_classifier.py`)

Trains a **Random Forest** on 4 target classes:

| ID | Class | SemanticKITTI examples |
|----|-------|------------------------|
| 0 | static_obstacle | buildings, fences, vegetation |
| 1 | dynamic_object | cars, pedestrians, cyclists |
| 2 | pole_wall | poles, traffic signs, walls |
| 3 | other | unlabelled, outliers |

```bash
# Synthetic training (no dataset needed):
python train_classifier.py --synthetic

# Real SemanticKITTI training:
python train_classifier.py /path/to/semantickitti --sequences 00 01 02
```

### 4 — Inference API (`classify_clusters.py`)

**Stable interface** consumed by Member B's grid engine:

```python
from classify_clusters import classify_clusters

results = classify_clusters(points, ground_mask)
for obj in results:
    print(f"{obj['class']} ({obj['confidence']:.2f}) — "
          f"{obj['points'].shape[0]} pts at {obj['centroid']}")
```

Each result dict contains: `cluster_id`, `class`, `class_id`, `confidence`, `points`, `centroid`, `bbox_min`, `bbox_max`.

If no trained model exists, a **rule-based heuristic** classifier is used automatically.

### 5 — Evaluation (`evaluate.py`)

```bash
# Synthetic evaluation:
python evaluate.py --synthetic

# Real dataset evaluation:
python evaluate.py /path/to/semantickitti --seq 08 --max-frames 100
```

Prints per-class precision/recall/F1/IoU + comparison vs. deep-learning baselines.

---

## 🏗️ Phase 3 — Foveated Ring-Buffer Grid Engine

Converts classified point clouds into a concentric 3-level 2.5D grid representation.

### 1 — Concentric 3-Ring Specification

| Ring | Range | Cell Size | Grid Size | Focus |
|---|---|---|---|---|
| Level 0 (Near) | 0 – 10 m | 0.05 m (5 cm) | 400 × 400 | Curbs, potholes, wheel contact |
| Level 1 (Mid) | 10 – 30 m | 0.15 m (15 cm) | 400 × 400 | Pedestrians, dynamic obstacles |
| Level 2 (Far) | 30 – 100 m | 0.50 m (50 cm) | 400 × 400 | Road boundaries, macro-terrain |

### 2 — Grid Engine Components

- **`grid_cell.py`**: Struct-of-Arrays (SoA) memory (`min_z`, `max_z`, `ground_z`, `z_variance`, `sem_class`, `sem_prob`, `point_count`, `confidence`, `overhang_flag`) and Multi-Level Surface Map (MLS) patch structure.
- **`ring_buffer.py`**: Modulo-wrap ring buffer indexing math and $O(1)$ ego-motion displacement offset shift tracking.
- **`grid_blending.py`**: Hysteresis boundary alpha-blending ($\alpha = \text{clamp}((r - (R_{\text{bound}}-w))/(2w), 0, 1)$) and speed-scaled temporal confidence decay ($\lambda = \lambda_0 (1 + k \cdot v_{\text{ego}})$).
- **`grid_engine.py`**: Main engine coordinating point insertion, Kalman ground filtering, class-weighted majority voting, overhang detection, and snapshot export.
- **`validate_grid_engine.py`**: Integration test suite verifying synthetic scenes, overhang detection, confidence decay, and ego-motion shifts.

```bash
# Run Grid Engine integration tests
python run.py validate_grid_engine
```

---

## 🔗 Interface Contracts

### Phase 1 → Phase 2
> **`segment_ground(points) → ground_mask`**

```python
def segment_ground(points, *, backend=None, sensor_height=1.73) -> np.ndarray[bool]
```

### Phase 2 → Phase 3 (Grid Engine)
> **`classify_clusters(points, ground_mask) → List[ClusterInfo]`**

```python
def classify_clusters(
    points: np.ndarray,      # (N, 3+) full point cloud
    ground_mask: np.ndarray,  # (N,) bool from segment_ground()
) -> List[dict]:              # [{cluster_id, class, confidence, points, ...}]
```

### Phase 3 → Phase 4 (Visualizer Dashboard API)
> **`engine.get_grid_snapshot(level) → Dict[str, np.ndarray]`**

```python
snapshot = engine.get_grid_snapshot(level=0)
# Returns 2D matrices (400, 400):
# snapshot["min_z"], snapshot["max_z"], snapshot["ground_z"],
# snapshot["sem_class"], snapshot["sem_prob"], snapshot["confidence"],
# snapshot["overhang"], snapshot["decayed_score"]
```

---

## 📊 Phase 4 — Visualization Dashboard

3-Ring Layout Visualizer with interactive / pure BMP snapshot fallback for zero-dependency execution:

```bash
# Test Phase 1 Raw Point Cloud Viewer
python run.py viewer_phase1 --synthetic

# Test Phase 2 Layout & HUD Stub
python run.py dashboard_phase2 --save-img recorded_demo/fallback_hud.bmp

# Test Phase 3 Live Dashboard Stream & Benchmark Loop
python run.py dashboard_phase3 --synthetic --frames 15
```

### 📸 Dashboard & Render Snapshots

| Phase 1 Point Cloud Viewer | Phase 2 HUD Visualization | Phase 3 Foveated Ring Grid |
| :---: | :---: | :---: |
| ![Viewer Phase 1](./viewer_p1.bmp) | ![Fallback HUD](./recorded_demo/fallback_hud.bmp) | ![Dashboard Phase 3](./dashboard_p3.bmp) |

| Live Stream Frame 1 | Live Stream Frame 2 | Live Stream Frame 3 |
| :---: | :---: | :---: |
| ![Frame 1](./dashboard_frame_001.bmp) | ![Frame 2](./dashboard_frame_002.bmp) | ![Frame 3](./dashboard_frame_003.bmp) |

---

## ⚡ Phase 5 — Benchmarking & Robustness Suite

Automated performance benchmarks evaluating RAM footprint, per-stage CPU throughput (FPS), semantic classification mIoU, and edge-case system robustness:

```bash
# Run Master Benchmark Suite (Memory + Latency + Accuracy + Robustness)
python run.py benchmark_suite

# Individual Benchmarks
python run.py benchmark_memory    # Compares 18.4MB MLRB vs 1,600MB Uniform Grid
python run.py benchmark_latency   # Measures per-stage execution times and FPS
python run.py benchmark_accuracy  # Evaluates class IoUs and mIoU vs ground truth
python run.py test_robustness     # Overhang, far-cell decay, and 100k-point stress tests
```

---

## 🎬 Phase 6 — Integration, Documentation & Hackathon Demo

Full technical documentation and dual-mode hackathon demo playbook:
- **`docs/demo_script.md`**: Live presentation script + zero-risk fallback walkthrough.
- **`docs/team-notes.md`**: Architectural trade-off analysis (2.5D vs 3D Voxel, CPU vs GPU, SoA layout).
- **`recorded_demo/`**: Archive containing pre-rendered BMP frame snapshots for instant presentation fallback.

---

## 🗺️ Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Dataset pipeline + ground segmentation + basic viewer | ✅ Done |
| **Phase 2** | Clustering + obstacle feature classifier + ring buffer core | ✅ Done |
| **Phase 3** | Foveated 2.5D Grid Engine + SoA ring buffers + temporal decay | ✅ Done |
| **Phase 4** | 3-Ring HUD visualization dashboard & frame playback | ✅ Done |
| **Phase 5** | Automated benchmarking suite vs uniform baseline & robustness tests | ✅ Done |
| **Phase 6** | Architecture documentation, hackathon script & pre-rendered demo archive | ✅ Done |

---

## 👥 Team

| Member | Focus Area |
|--------|---------------------------------------------------|
| A | Perception / ML (ground seg, clustering, classify) |
| B | Grid Engine / Backend (SoA ring buffer, 2.5D projection) |
| C | Visualization / Integration (dashboard, benchmarking, demo) |

---

## 📜 License

Apache-2.0 — see individual file headers for details.

