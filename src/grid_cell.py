"""
grid_cell.py — Struct-of-Arrays (SoA) Grid Cell Data Structure
================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member B · Phase 3 (3a)

Defines contiguous Struct-of-Arrays (SoA) memory buffers for a single
400x400 ring level, as well as Multi-Level Surface Map patch structures
for handling overhangs/bridges.

Data Layout per Ring Level (400x400 = 160,000 cells):
    - min_z         : float32 [160000]
    - max_z         : float32 [160000]
    - ground_z      : float32 [160000] (Kalman/EMA filtered)
    - z_variance    : float32 [160000]
    - sem_class     : int32   [160000] (0: static, 1: dynamic, 2: pole/wall, 3: other, -1: unobserved)
    - sem_prob      : float32 [160000]
    - point_count   : int32   [160000]
    - confidence    : float32 [160000] (Temporal decay validity weight)
    - overhang_flag : bool    [160000] (True if multi-surface gap detected)

Usage:
    from grid_cell import RingBufferSoA, SurfacePatch

    ring_grid = RingBufferSoA(grid_size=400)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SurfacePatch:
    """Multi-Level Surface Map (MLS) patch representation (Triebel et al.).

    Used when a single 2D cell contains multiple vertical layers
    (e.g., bridge surface + ground below, or vehicle overhang).
    """
    mean_z: float          # Mean elevation of this patch
    var_z: float           # Variance of height
    depth: float           # Thickness of patch (max_z - min_z)
    is_vertical: bool      # True if depth > thickness_tau (~0.1m)
    sem_class: int = -1    # Semantic class ID
    confidence: float = 1.0


class RingBufferSoA:
    """Struct-of-Arrays (SoA) for a single concentric grid ring level."""

    def __init__(self, grid_size: int = 400):
        self.grid_size = grid_size
        self.num_cells = grid_size * grid_size

        self.reset()

    def reset(self) -> None:
        """Reset all cells to unobserved state."""
        self.min_z = np.full(self.num_cells, np.inf, dtype=np.float32)
        self.max_z = np.full(self.num_cells, -np.inf, dtype=np.float32)
        self.ground_z = np.zeros(self.num_cells, dtype=np.float32)
        self.z_variance = np.zeros(self.num_cells, dtype=np.float32)
        self.sem_class = np.full(self.num_cells, -1, dtype=np.int32)
        self.sem_prob = np.zeros(self.num_cells, dtype=np.float32)
        self.point_count = np.zeros(self.num_cells, dtype=np.int32)
        self.confidence = np.zeros(self.num_cells, dtype=np.float32)
        self.overhang_flag = np.zeros(self.num_cells, dtype=bool)

        # Dictionary mapping 1D cell index -> list of SurfacePatch objects (sparse allocation)
        self.patch_map: dict[int, list[SurfacePatch]] = {}

    def clear_cells(self, cell_indices: np.ndarray) -> None:
        """Clear specific cell indices when buffer wraps during ego-motion."""
        if cell_indices.size == 0:
            return

        self.min_z[cell_indices] = np.inf
        self.max_z[cell_indices] = -np.inf
        self.ground_z[cell_indices] = 0.0
        self.z_variance[cell_indices] = 0.0
        self.sem_class[cell_indices] = -1
        self.sem_prob[cell_indices] = 0.0
        self.point_count[cell_indices] = 0
        self.confidence[cell_indices] = 0.0
        self.overhang_flag[cell_indices] = False

        for idx in cell_indices:
            self.patch_map.pop(int(idx), None)


# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    grid = RingBufferSoA(grid_size=400)
    print(f"RingBufferSoA initialized with {grid.num_cells:,} cells.")
    print(f"Memory size of min_z array: {grid.min_z.nbytes / 1024:.1f} KB")
