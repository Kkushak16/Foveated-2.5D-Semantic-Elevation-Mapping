# 03 — Foveated Ring-Buffer Grid Engine (Member B · Phases 1-3)

Use after 00_master_context.md. This is the core C++ deliverable — split into
3 sub-prompts for the 3 phases so the assistant doesn't try to do it all at once.

---

## 3a. Phase 1 — Skeleton & Coordinate Math

```
<task>
Set up the C++ project skeleton for our Multi-Level Ring Buffer (MLRB) grid
engine and implement the coordinate transform math only (no insertion logic yet).
</task>

<requirements>
0. BEFORE writing a custom ring buffer from scratch: evaluate ANYbotics'
   `grid_map_core` (from github.com/ANYbotics/grid_map) as a base. It's pure
   C++17 + Eigen, zero ROS dependency, and already implements a 2D circular
   buffer per layer for O(1) robot-relative re-positioning with no memory
   copy — the same mechanism our MLRB needs for ego-motion offset updates.
   If it cleanly supports 3 independent layers at 3 different resolutions
   (our near/mid/far rings), use it as the base and focus effort on the
   variable-resolution blending logic instead. If it only supports one
   resolution per map instance, note that limitation and fall back to a
   custom implementation, using grid_map_core's circular-buffer approach as
   the reference pattern either way.
1. C++17/20 project using CMake, Eigen for linear algebra. No CUDA.
2. Define the per-cell struct exactly as specified in project context
   (Struct-of-Arrays: min_z, max_z, ground_z, z_variance, sem_class, sem_prob,
   point_count) as separate contiguous arrays, not array-of-structs. Also add
   a `confidence` field per cell (float) — see the decay model in the master
   context; this replaces relying on point_count alone as a trust signal.
   If time allows in this phase, sketch (don't fully implement yet) how a
   cell could hold a small fixed-size list of patches (mean height, variance,
   depth, is_vertical flag) instead of a single min/max pair — this is the
   Multi-Level Surface Map pattern and makes bridges/overhangs a first-class
   case rather than a special-cased flag. Flag clearly if you scope this out
   of Phase 1 for time reasons.
3. Implement `pointToCellIndex(x, y, ring_level, offset_x, offset_y) -> (u, v)`
   with correct modulo wrap-around for ring-buffer indexing, matching the
   3-ring spec (0-10m/5cm, 10-30m/15cm, 30-100m/50cm, 400x400 each).
4. Write unit tests (Catch2 or GoogleTest) for the index math: known (x,y)
   inputs must produce exact expected (u,v), including edge cases at ring
   boundaries and after simulated ego-motion offset changes.
</requirements>

<deliverables>
- CMakeLists.txt
- include/grid_cell.hpp (the SoA struct)
- include/ring_buffer.hpp / src/ring_buffer.cpp (index math only)
- tests/test_ring_buffer.cpp
</deliverables>

<output_format>
Full code per file. Explain the modulo-wrap logic in a short comment block
above the function.
</output_format>
```

## 3b. Phase 2 — Ego-Motion Update & Boundary Blending

```
<task>
Extend the ring buffer with O(1) ego-motion offset updates and boundary
blending between ring levels.
</task>

<requirements>
1. Implement ego-motion offset update: on vehicle movement (dx, dy), update
   ring offsets in O(1) without shifting/copying the underlying array; clear
   only cells that wrap into newly-unobserved space.
2. Implement the 10-15% hysteresis overlap margin between adjacent rings.
3. Implement distance-weighted linear alpha-blend at ring transition zones
   using the formula: alpha = clamp((r - (R_bound - w)) / (2w), 0, 1);
   z_composite = (1-alpha)*z_fine + alpha*z_coarse.
4. Enforce integer grid alignment so a coarse cell maps exactly to a k x k
   block of fine cells (no diagonal jitter).
5. Implement per-cell confidence accumulate + decay per the master context
   model: on re-observation, w_new = w_old + w_p; each update cycle (or
   fixed time step) apply w <- alpha_decay * w for cells not re-observed.
   Expose a tunable decay rate; note in a comment that decay rate can later
   be scaled by ego speed (lambda = k * v) if the vehicle's speed is
   available to the grid engine.
6. Add unit tests: simulate ego motion past a boundary and verify no memory
   bug; verify alpha-blend output matches the formula at known test points;
   verify confidence decays correctly for an unobserved cell over N cycles
   and re-accumulates correctly when re-observed.
</requirements>

<deliverables>
- Updated ring_buffer.hpp/.cpp with ego-motion + blending + confidence logic
- tests/test_boundary_blend.cpp
- tests/test_ego_motion.cpp
- tests/test_confidence_decay.cpp
</deliverables>

<output_format>
Full code diffs/files. Flag any deviation from the formula above explicitly.
</output_format>
```

## 3c. Phase 3 — Point Insertion, Kalman Filtering, Semantic Voting

```
<task>
Implement full point-to-grid insertion pipeline, taking classified points
from Member A's classify_clusters() output (semantic label + xyz per point)
and inserting them into the appropriate ring buffer cells.
</task>

<requirements>
1. `insertPoints(points, semantic_labels, confidences)` — projects each point
   into its ring's cell, updates min_z/max_z, and marks overhang if
   (max_z - ground_z) > vehicle_clearance with an empty gap between them.
2. Recursive Kalman filter (or EMA fallback) per cell for ground_z, tuned
   for sparse far-range cells (fewer LiDAR returns at range).
3. Class-weighted majority voting per cell when multiple points/classes land
   in the same coarse cell — dynamic objects (pedestrians, vehicles) get
   higher weight so they aren't erased by downsampling.
4. Weight each cell's final geometric/safety score by its mean confidence
   (from 3b) before it's used downstream — a cell with decayed confidence
   should contribute less even if its last known geometry looked safe. This
   mirrors the UAV LiDAR-vision fusion pattern: score = base_score *
   mean_confidence.
5. Expose a clean API for Member C's visualizer: `getGridSnapshot(ring_level)
   -> flat arrays ready for color-coded rendering`, and include per-cell
   confidence in the snapshot so the dashboard can optionally show it.
6. Integration test: run one full synthetic frame (ground + obstacle + overhang
   scenario) end-to-end and verify no NaNs, correct overhang flagging, correct
   majority-vote class per cell. Add a second integration test: mark a region
   safe, stop observing it for N frames, verify its confidence has decayed
   below a threshold and it's no longer trusted without re-observation
   (same failure mode the UAV landing paper found in "fusion without decay"
   baselines — stale safe regions getting reused after conditions changed).
</requirements>

<deliverables>
- src/grid_insert.cpp / include/grid_insert.hpp
- src/kalman_filter.hpp
- src/semantic_voting.hpp
- include/grid_api.hpp (snapshot export for visualization)
- tests/test_insertion_integration.cpp
</deliverables>

<output_format>
Full code per file. Include a short doc comment on getGridSnapshot's exact
output format since Member C depends on it directly.
</output_format>
```
