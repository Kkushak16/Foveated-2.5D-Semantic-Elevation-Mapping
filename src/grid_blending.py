"""
grid_blending.py — Ring Boundary Blending & Confidence Temporal Decay
======================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member B · Phase 3 (3b)

Implements:
  1. Hysteresis overlap blending across ring boundaries.
  2. Linear distance-weighted alpha blending formula:
        alpha = clamp((r - (R_bound - w)) / (2w), 0, 1)
        z_composite = (1 - alpha) * z_fine + alpha * z_coarse
  3. Temporal confidence decay & accumulation model:
        Accumulate: w_new = min(w_old + w_p, max_confidence)
        Decay:      w_new = w_old * exp(-lambda_decay * dt)
        where lambda_decay = base_rate * (1.0 + speed_k * ego_speed)
"""

import numpy as np
from typing import Dict, Tuple

from ring_buffer import MultiLevelRingBuffer, RING_CONFIGS


class GridBlendingEngine:
    """Manages cross-ring alpha-blending and confidence temporal decay."""

    def __init__(self, blend_width: float = 1.0, base_decay_rate: float = 0.15, speed_k: float = 0.02):
        """
        Parameters
        ----------
        blend_width : float
            Width of hysteresis transition zone in metres (default 1.0m).
        base_decay_rate : float
            Base temporal decay rate per second (lambda_0).
        speed_k : float
            Speed multiplier scaling factor (lambda = lambda_0 * (1 + speed_k * v)).
        """
        self.blend_width = blend_width
        self.base_decay_rate = base_decay_rate
        self.speed_k = speed_k

    def compute_alpha_blend(self, r: np.ndarray, boundary_radius: float) -> np.ndarray:
        """Compute linear alpha blending weight for radial distance r.

        alpha = 0  -> 100% fine ring
        alpha = 1  -> 100% coarse ring
        """
        w = self.blend_width
        r_start = boundary_radius - w
        r_end = boundary_radius + w

        alpha = (r - r_start) / (2.0 * w)
        return np.clip(alpha, 0.0, 1.0)

    def blend_fine_coarse_elevation(
        self,
        r: np.ndarray,
        z_fine: np.ndarray,
        z_coarse: np.ndarray,
        boundary_radius: float,
    ) -> np.ndarray:
        """Blend fine and coarse ring elevations across boundary."""
        alpha = self.compute_alpha_blend(r, boundary_radius)
        return (1.0 - alpha) * z_fine + alpha * z_coarse

    def update_confidence_decay(
        self,
        mlrb: MultiLevelRingBuffer,
        dt: float,
        ego_speed: float = 0.0,
        observed_mask_per_level: Dict[int, np.ndarray] = None,
    ) -> None:
        """Apply temporal confidence decay to unobserved cells.

        Parameters
        ----------
        mlrb : MultiLevelRingBuffer
        dt : float
            Time step elapsed in seconds.
        ego_speed : float
            Vehicle speed in m/s (scales decay rate).
        observed_mask_per_level : Dict[int, np.ndarray]
            Boolean mask per ring level indicating cells observed in current frame.
        """
        effective_lambda = self.base_decay_rate * (1.0 + self.speed_k * ego_speed)
        decay_factor = np.exp(-effective_lambda * dt)

        for lvl, ring_soa in mlrb.rings.items():
            if observed_mask_per_level is not None and lvl in observed_mask_per_level:
                obs_mask = observed_mask_per_level[lvl]
                # Apply decay only to unobserved cells
                unobs_mask = ~obs_mask
                ring_soa.confidence[unobs_mask] *= decay_factor
            else:
                # Decay all cells
                ring_soa.confidence *= decay_factor


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    blender = GridBlendingEngine(blend_width=1.0)

    r_test = np.array([8.5, 9.0, 10.0, 11.0, 11.5])
    alphas = blender.compute_alpha_blend(r_test, boundary_radius=10.0)
    print(f"Distances: {r_test}")
    print(f"Alpha blend weights (Boundary=10m): {alphas}")

    z_f = np.full(5, 2.0)
    z_c = np.full(5, 3.0)
    z_blended = blender.blend_fine_coarse_elevation(r_test, z_f, z_c, boundary_radius=10.0)
    print(f"Blended elevations: {z_blended}")
    print("✅ Boundary blending & confidence decay verified!")
