"""
grid_engine.py — Complete Foveated 2.5D Ring Buffer Grid Engine
================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member B · Phase 3 (3c)

Main entry point for the Grid Engine. Coordinates:
  - 3-level concentric ring buffer memory (RingBufferSoA)
  - Point cloud & semantic classification insertion
  - Cell-level ground_z Kalman filtering
  - Class-weighted majority voting (dynamic objects prioritized)
  - Overhang / multi-level surface detection
  - Temporal confidence accumulation & decay
  - Visualizer export snapshot API: get_grid_snapshot(ring_level)

Usage:
    from grid_engine import FoveatedGridEngine

    engine = FoveatedGridEngine()
    engine.insert_points(points, semantic_classes, confidences)
    snapshot = engine.get_grid_snapshot(level=0)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ring_buffer import MultiLevelRingBuffer, RING_CONFIGS, RingConfig
from grid_blending import GridBlendingEngine
from grid_cell import SurfacePatch

logger = logging.getLogger(__name__)

# Class priority weights for downsampling majority vote
# Higher weight means the class dominates when multiple points land in one cell
CLASS_WEIGHTS: Dict[int, float] = {
    1: 3.0,  # dynamic_object (pedestrians, vehicles) -> Highest priority
    2: 2.0,  # pole_wall -> Medium-high priority
    0: 1.0,  # static_obstacle -> Normal priority
    3: 0.5,  # other -> Low priority
   -1: 0.0,  # unobserved
}


class FoveatedGridEngine:
    """Complete 3-Level Foveated 2.5D Ring-Buffer Grid Engine."""

    def __init__(
        self,
        vehicle_clearance: float = 2.0,
        overhang_gap_tau: float = 0.8,
        kalman_q: float = 0.01,
        kalman_r: float = 0.05,
    ):
        """
        Parameters
        ----------
        vehicle_clearance : float
            Clearance height in metres (e.g. 2.0m).
        overhang_gap_tau : float
            Min vertical gap between ground and roof to flag overhang.
        kalman_q : float
            Process variance for ground_z Kalman filter.
        kalman_r : float
            Measurement variance for ground_z Kalman filter.
        """
        self.mlrb = MultiLevelRingBuffer()
        self.blender = GridBlendingEngine()

        self.vehicle_clearance = vehicle_clearance
        self.overhang_gap_tau = overhang_gap_tau
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r

    def insert_points(
        self,
        points: np.ndarray,
        sem_classes: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        ground_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, int]:
        """Insert a classified point cloud into the foveated ring buffer grid.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3+)
            LiDAR point cloud XYZ[I...].
        sem_classes : np.ndarray[int], shape (N,)
            Semantic class per point (0: static, 1: dynamic, 2: pole, 3: other).
        confidences : np.ndarray[float], shape (N,)
            Classification confidence per point (0.0 to 1.0).
        ground_mask : np.ndarray[bool], shape (N,)
            Ground mask from segment_ground(). True = ground.

        Returns
        -------
        stats : dict
            Point insertion counts per ring level.
        """
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected (N, 3+) array, got {points.shape}")

        N = points.shape[0]
        if N == 0:
            return {0: 0, 1: 0, 2: 0}

        xyz = points[:, :3]

        if sem_classes is None:
            sem_classes = np.full(N, 3, dtype=np.int32)
        if confidences is None:
            confidences = np.ones(N, dtype=np.float32)
        if ground_mask is None:
            ground_mask = np.zeros(N, dtype=bool)

        stats = {}
        observed_masks = {}

        # Process each of the 3 ring levels
        for lvl in range(3):
            u, v, idx, valid_mask = self.mlrb.points_to_cells_vectorized(xyz, lvl)
            valid_indices = np.nonzero(valid_mask)[0]

            stats[lvl] = len(valid_indices)
            if len(valid_indices) == 0:
                continue

            ring_soa = self.mlrb.rings[lvl]
            cell_flat_indices = idx[valid_indices]

            # Track observed cells in this frame
            unique_cells, unique_counts = np.unique(cell_flat_indices, return_counts=True)
            frame_obs_mask = np.zeros(ring_soa.num_cells, dtype=bool)
            frame_obs_mask[unique_cells] = True
            observed_masks[lvl] = frame_obs_mask

            # 1. Update point counts & confidence accumulation
            ring_soa.point_count[unique_cells] += unique_counts.astype(np.int32)
            # Confidence accumulates up to max 10.0
            ring_soa.confidence[unique_cells] = np.minimum(
                ring_soa.confidence[unique_cells] + 1.0, 10.0
            )

            # 2. Vectorized min_z / max_z updates per cell
            z_vals = xyz[valid_indices, 2]
            np.minimum.at(ring_soa.min_z, cell_flat_indices, z_vals)
            np.maximum.at(ring_soa.max_z, cell_flat_indices, z_vals)

            # 3. Ground elevation update using Kalman filter
            is_ground = ground_mask[valid_indices]
            if np.any(is_ground):
                ground_pts_idx = cell_flat_indices[is_ground]
                ground_z_meas = z_vals[is_ground]

                # Group by cell for ground update
                ug_cells, ug_inv = np.unique(ground_pts_idx, return_inverse=True)
                ug_means = np.bincount(ug_inv, weights=ground_z_meas) / np.bincount(ug_inv)

                # Recursive Kalman Filter update for ground_z
                # K_gain = (P + Q) / (P + Q + R)
                p_old = ring_soa.z_variance[ug_cells]
                k_gain = (p_old + self.kalman_q) / (p_old + self.kalman_q + self.kalman_r)

                ring_soa.ground_z[ug_cells] += k_gain * (ug_means - ring_soa.ground_z[ug_cells])
                ring_soa.z_variance[ug_cells] = (1.0 - k_gain) * (p_old + self.kalman_q)

            # 4. Class-weighted semantic voting per cell
            # Priority: dynamic > pole/wall > static > other
            sub_classes = sem_classes[valid_indices]
            sub_confs = confidences[valid_indices]

            for c_idx in unique_cells:
                mask_c = (cell_flat_indices == c_idx)
                pts_cls = sub_classes[mask_c]
                pts_cnf = sub_confs[mask_c]

                # Weighted class scoring
                class_scores = np.zeros(4, dtype=np.float32)
                for cls_id in range(4):
                    m = (pts_cls == cls_id)
                    if np.any(m):
                        w_prio = CLASS_WEIGHTS.get(cls_id, 1.0)
                        class_scores[cls_id] = np.sum(pts_cnf[m]) * w_prio

                best_cls = int(np.argmax(class_scores))
                tot_score = np.sum(class_scores)
                best_prob = (class_scores[best_cls] / tot_score) if tot_score > 0 else 0.0

                ring_soa.sem_class[c_idx] = best_cls
                ring_soa.sem_prob[c_idx] = float(best_prob)

                # 5. Overhang Detection logic
                # Flag if (max_z - ground_z) > clearance AND min_z > ground_z + tau
                c_min = ring_soa.min_z[c_idx]
                c_max = ring_soa.max_z[c_idx]
                c_gnd = ring_soa.ground_z[c_idx]

                if (c_max - c_gnd) > self.vehicle_clearance and (c_min - c_gnd) > self.overhang_gap_tau:
                    ring_soa.overhang_flag[c_idx] = True

                    # Store multi-level surface patch
                    patch = SurfacePatch(
                        mean_z=float((c_max + c_min) / 2.0),
                        var_z=float((c_max - c_min) ** 2 / 12.0),
                        depth=float(c_max - c_min),
                        is_vertical=bool(c_max - c_min > 0.1),
                        sem_class=best_cls,
                        confidence=float(ring_soa.confidence[c_idx]),
                    )
                    ring_soa.patch_map[c_idx] = [patch]

        return stats

    def update_temporal_decay(self, dt: float, ego_speed: float = 0.0) -> None:
        """Decay confidence of unobserved cells over time."""
        self.blender.update_confidence_decay(self.mlrb, dt=dt, ego_speed=ego_speed)

    def update_ego_motion(self, dx: float, dy: float) -> None:
        """Update grid position with vehicle ego-motion shift."""
        self.mlrb.update_ego_motion(dx, dy)

    def get_grid_snapshot(self, level: int = 0) -> Dict[str, np.ndarray]:
        """Export flat arrays for Member C's visualizer and downstream planner.

        Output Dictionary Keys:
          - "min_z"        : (400, 400) float32 array
          - "max_z"        : (400, 400) float32 array
          - "ground_z"     : (400, 400) float32 array
          - "sem_class"    : (400, 400) int32 array (-1: unobserved, 0..3)
          - "sem_prob"     : (400, 400) float32 array
          - "confidence"   : (400, 400) float32 array
          - "overhang"     : (400, 400) bool array
          - "decayed_score": (400, 400) float32 geometric height weighted by confidence
          - "extent_m"     : float (half width of ring in metres)
          - "cell_size_m"  : float (resolution of cell in metres)

        Documentation for Member C:
        Reshape all 1D flat arrays into (400, 400) 2D grid matrices for fast matrix plot / image rendering.
        """
        cfg = RING_CONFIGS[level]
        soa = self.mlrb.rings[level]
        sz = cfg.grid_size

        # Geometric height score weighted by temporal confidence
        # decayed_score = (max_z - ground_z) * min(confidence, 1.0)
        valid_mask = (soa.max_z > -100) & (soa.min_z < 100)
        h_diff = np.zeros(soa.num_cells, dtype=np.float32)
        h_diff[valid_mask] = soa.max_z[valid_mask] - soa.ground_z[valid_mask]

        norm_conf = np.clip(soa.confidence / 1.0, 0.0, 1.0)
        decayed_score = h_diff * norm_conf

        return {
            "min_z": soa.min_z.reshape((sz, sz)),
            "max_z": soa.max_z.reshape((sz, sz)),
            "ground_z": soa.ground_z.reshape((sz, sz)),
            "sem_class": soa.sem_class.reshape((sz, sz)),
            "sem_prob": soa.sem_prob.reshape((sz, sz)),
            "confidence": soa.confidence.reshape((sz, sz)),
            "overhang": soa.overhang_flag.reshape((sz, sz)),
            "decayed_score": decayed_score.reshape((sz, sz)),
            "extent_m": cfg.extent,
            "cell_size_m": cfg.cell_size,
        }


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = FoveatedGridEngine()

    # Generate synthetic test points
    rng = np.random.default_rng(42)
    pts = rng.uniform(-15, 15, size=(5000, 3)).astype(np.float32)
    cls = rng.choice([0, 1, 2, 3], size=5000)
    cnf = rng.uniform(0.5, 1.0, size=5000).astype(np.float32)
    g_mask = (pts[:, 2] < -1.5)

    stats = engine.insert_points(pts, cls, cnf, g_mask)
    print("Point insertion stats per ring level:")
    for lvl, cnt in stats.items():
        cfg = RING_CONFIGS[lvl]
        print(f"  Level {lvl} ({cfg.min_range}-{cfg.max_range}m, {cfg.cell_size}m): {cnt} points inserted")

    snapshot = engine.get_grid_snapshot(level=0)
    obs_count = np.sum(snapshot["sem_class"] >= 0)
    print(f"Snapshot Level 0: {obs_count} cells observed out of {snapshot['sem_class'].size}")
    print("✅ Foveated Grid Engine verification complete!")
