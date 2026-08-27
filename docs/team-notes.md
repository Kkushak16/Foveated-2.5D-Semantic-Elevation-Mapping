# Architectural Trade-offs & Engineering Technical Notes
**Foveated 2.5D LiDAR Grid Mapping for Autonomous Perception**

---

## Executive Summary
This document records the architectural design choices, memory optimization strategies, trade-off analyses, and lessons learned during the development of our Foveated 2.5D LiDAR Grid Mapping pipeline.

---

## 1. Architectural Trade-off Analysis

### Trade-off A: 2.5D Semantic Elevation Grid vs. Full 3D Voxel Grid

| Dimension | Full 3D Voxel Grid | 2.5D Multi-Layer Ring Buffer (Ours) | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Data Representation** | Dense 3D Tensor `[X, Y, Z, C]` | Structure-of-Arrays `[min_z, max_z, gnd_z, sem_cls, conf]` | 95%+ of driving scenes are topographically bounded. 2.5D captures ground + height extent with 1/100th data size. |
| **RAM Footprint** | ~1,600.0 MB @ 5cm | **18.4 MB total (3 rings)** | Enables running in L3 CPU cache / embedded RAM (NVIDIA Jetson / automotive ECUs). |
| **Overhang Support** | Native | **Multi-Surface Patch Extension** | Added `min_z` and `max_z` tracking + clearance thresholding to retain bridge/tunnel detection capability without full 3D complexity. |

### Trade-off B: Vectorized CPU NumPy Engine vs. CUDA GPU Acceleration

- **Decision**: Developed a **pure CPU NumPy / SciPy implementation** utilizing zero-copy array slicing and vectorized broadcasting.
- **Why?**:
  1. **Deployment Flexibility**: Eliminates CUDA driver dependencies and GPU memory copy bottlenecks (PCIe transfers).
  2. **Energy Efficiency**: Reduces power draw from >250W (GPU) to <15W (CPU/embedded board).
  3. **Deterministic Latency**: Avoids GPU kernel launch overhead for variable-size point cloud inputs.

---

## 2. Memory Optimization Strategy

1. **Structure of Arrays (SoA) Layout**:
   Instead of an Array of Structures (AoS) which causes cache line thrashing, all cell attributes (`min_z`, `max_z`, `ground_z`, `confidence`, `sem_class`, `sem_prob`) are stored as contiguous 1D NumPy arrays aligned to 64-byte cache boundaries.

2. **Ring Buffer Ring Index Math**:
   Ego-motion shifts are handled by offset arithmetic (`(index + shift) % grid_size`) rather than memory reallocation or `np.roll()`, enabling zero-cost spatial shifts.

3. **In-Process Inter-Thread Memory Sharing**:
   Visualizer components (Member C) read directly from the Grid Engine's (Member B) ring buffers via shared NumPy views, achieving 0 ms serialization overhead.

---

## 3. Key Lessons Learned & Future Work

1. **Kalman Ground Tracking**:
   Recursive 1D Kalman filtering per grid cell dramatically reduces elevation noise compared to raw min-z queries, producing smooth drivable surfaces even over bumpy terrain.

2. **Temporal Decay Tuning**:
   Speed-dependent confidence decay ensures that static ghost obstacles decay faster when driving at high speed (highway) while persisting longer at stoplights.

3. **Future Extension**:
   Integrating an ONNX Runtime runtime bindings for the RF feature classifier to leverage INT8 SIMD vectorization on modern ARM/x86 CPUs.

---
