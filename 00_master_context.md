# 00 — Master Context Prompt

Paste this FIRST in any new chat with an AI coding assistant (Claude, ChatGPT, Claude Code)
before using any of the other phase prompts. It primes the assistant with full project
context so later prompts can stay short.

---

```
<project>
Name: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Goal: Convert dense 3D LiDAR point clouds into a lightweight, adaptive 2.5D grid
(elevation map + semantic layers) in real time, inspired by human foveated vision —
high resolution near the vehicle, coarser resolution far away.
</project>

<architecture>
Pipeline: raw LiDAR point cloud -> ground segmentation -> obstacle clustering ->
classification -> foveated 2.5D grid projection -> real-time visualization.

We are using a LIGHTWEIGHT / LOW-GPU-DEPENDENCY variant (not full deep-learning
3D sparse-conv networks):
- Ground segmentation: Patchwork++ or RANSAC plane-fit (CPU, C++)
- Obstacle grouping: Euclidean Clustering (PCL) or DBSCAN (CPU)
- Classification: hand-crafted features + Random Forest (or PointPillars-lite
  if a deep-learning path is explicitly requested)
- Grid engine: Concentric Multi-Level Ring Buffers (MLRB), pure C++17/20 + Eigen,
  CUDA optional and never required for correctness
- Visualization: Open3D or Foxglove Studio
- Middleware: ROS 2 Humble (optional)
- Dataset: SemanticKITTI (subset) or nuScenes-mini

Grid design — 3 concentric ego-centric rings:
| Level | Range     | Cell size | Grid dims | Focus                              |
|-------|-----------|-----------|-----------|-------------------------------------|
| 0 Near| 0-10 m    | 5 cm      | 400x400   | curbs, potholes, wheel contact      |
| 1 Mid | 10-30 m   | 15 cm     | 400x400   | pedestrians, dynamic obstacles      |
| 2 Far | 30-100 m  | 50 cm     | 400x400   | road boundaries, macro-terrain      |

Per-cell state (Struct-of-Arrays): min_z, max_z, ground_z (Kalman-filtered),
z_variance, sem_class, sem_prob, point_count.

Boundary-artifact prevention: 10-15% ring overlap (hysteresis band),
distance-weighted alpha-blend at transitions, integer-aligned grid so coarse
cells map exactly to a k x k block of fine cells.

Overhang handling: store min_z AND max_z per column; flag overhang if
(max_z - ground_z) > vehicle_clearance with an empty gap between them.
Reference design (Triebel et al., Multi-Level Surface Maps): instead of one
min/max pair, a cell can hold a LIST of surface patches, each patch =
(mean height mu, variance sigma, depth d). Two height values belong to the
same patch if within a gap_size (~1.0m, tune to vehicle clearance); an
interval taller than thickness_tau (~10cm) is classified vertical, otherwise
horizontal. This naturally represents bridges/tunnels/overhangs as separate
patches in the same (x,y) cell instead of a single flagged min/max pair.
Traversability rule (borrowed from same paper): a patch is traversable only
if >=5 of its 8 neighbor cells have a patch AND height diff to each such
neighbor is <10cm.

Known open-source building blocks (checked directly, use instead of
reinventing where they fit):
- ANYbotics/grid_map — `grid_map_core` submodule has ZERO ROS dependency,
  pure C++17 + Eigen. Stores multi-layer 2D grid data as a 2D circular
  buffer specifically so the map can re-position with the robot in O(1)
  with no memory copy — this is functionally our MLRB ring-buffer offset
  idea, already implemented and tested. Member B should evaluate using
  grid_map_core as the base data structure instead of hand-rolling the
  ring buffer from scratch in Phase 1 — could save significant time; only
  hand-roll if grid_map_core's single-resolution-per-layer model can't
  cleanly support our 3-ring variable-resolution design.
- ANYbotics/elevation_mapping — ROS package built on grid_map, robot-centric
  elevation mapping from point cloud + pose (IMU/odometry), explicitly
  handles pose drift. Pure C++/ROS, no GPU. Good reference for the point
  cloud -> elevation layer ingest pipeline structure even if not used
  directly (our system may skip ROS entirely for the lite/CPU path).
- isl-org/Open3D-ML — has pretrained RandLA-Net and KPConv checkpoints for
  SemanticKITTI semantic segmentation, PyTorch/TensorFlow, built-in dataset
  loader and visualizer. NOT part of the lite pipeline by default, but
  useful as: (a) a quick dataset loader/visualizer utility, or (b) an
  optional DL-accuracy comparison baseline in Phase 4 benchmarking if a GPU
  is available for that one comparison run.
- charlesq34/pointnet — the original PointNet. AVOID for this project: it's
  TensorFlow 1.2-era code and the repo itself says GPU access is "highly
  recommended." Directly contradicts our lite/CPU-first constraint — do not
  adopt, even though it's a well-known point cloud network.

Confidence/decay model (reference: adaptive confidence-driven LiDAR-vision
fusion for UAV landing, Sade et al. 2026): don't just store point_count per
cell — maintain a confidence weight w per cell/patch that:
- accumulates on new observation: w_new = w_old + w_p (w_p typically 1)
- decays over time when NOT re-observed: w <- alpha * w (discrete), or
  continuous form C(t) = C0 * exp(-lambda * (t - t0))
- lambda (decay rate) can scale with ego speed: lambda = k * v — cells left
  behind faster (higher speed) lose trust faster since they're stale longer
  before possible re-observation
- confidence is a TEMPORAL VALIDITY signal, not a safety/occupancy
  probability — high confidence just means "recently and consistently
  observed," decoupled from whether the cell is safe/drivable
- final per-cell geometric score should be multiplied by mean confidence
  before being used in downstream fusion, so stale cells contribute less
  even if their last known geometry looked good
This directly generalizes our ring-buffer Kalman/EMA idea for sparse far
cells: use decaying confidence instead of a flat point_count, especially
important for Level 2 (far ring) where returns are sparse and stale data is
riskiest.

Cross-modal fusion weighting: default fixed weights (LiDAR 0.7 / image 0.3)
are fine as a baseline (matches published UAV LiDAR-vision fusion work), but
an entropy-based uncertainty weighting is a documented upgrade path: per
modality, confidence = exp(-entropy(prediction_distribution)); normalize
confidence across modalities and use as per-frame fusion weights instead of
a fixed split. Reference: MS-xMUDA (Sun et al. 2025) uses this pattern for
2D/3D fusion in point cloud segmentation. Not required for MVP — note as a
Phase 4 stretch goal if time allows.
</architecture>

<team>
3-member team:
- Member A: Perception / ML (ground segmentation, clustering, classification)
- Member B: Grid Engine / Backend (C++ ring buffer, projection, filtering)
- Member C: Visualization / Integration (dashboard, benchmarking glue, demo)
</team>

<roadmap>
Phase 1 (Days 1-5): dataset pipeline + ground segmentation + grid skeleton + basic viewer
Phase 2 (Days 6-12): clustering + classifier + ring buffer core + dashboard skeleton
Phase 3 (Days 13-19): full integration — classifier -> grid -> live dashboard
Phase 4 (Days 20-25): benchmarking vs. uniform-grid baseline, polish, docs, demo
</roadmap>

<instructions_for_assistant>
When I give you a task, assume the above context applies unless I say otherwise.
Ask before assuming libraries/versions not listed above. Keep code CPU-runnable
by default; only use CUDA/GPU-specific code if I explicitly ask for the
performance-optimized path. Match whichever teammate's task I specify (A/B/C).
</instructions_for_assistant>
```
