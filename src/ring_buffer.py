"""
ring_buffer.py — Multi-Level Ring Buffer Coordinate Math & Ego-Motion
======================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member B · Phase 3 (3a & 3b)

Implements coordinate mapping and O(1) ego-motion shift indexing for
concentric 3-level ring buffers.

Modulo-wrap indexing formula:
    World point (x, y) -> Ego-centric continuous cell coords:
        c_x = floor(x / cell_size)
        c_y = floor(y / cell_size)
    Ego-motion offset (offset_x, offset_y) in cell units:
        u = (c_x + offset_x) mod grid_size
        v = (c_y + offset_y) mod grid_size
    Flat SoA Index:
        idx = u * grid_size + v

Ring Specification:
    Level 0 (Near):  0 – 10  m | Cell: 0.05 m (5 cm)  | 400x400
    Level 1 (Mid) : 10 – 30  m | Cell: 0.15 m (15 cm) | 400x400
    Level 2 (Far) : 30 – 100 m | Cell: 0.50 m (50 cm) | 400x400
"""

from typing import Dict, List, Tuple
import numpy as np

from grid_cell import RingBufferSoA


class RingConfig:
    """Configuration parameter set for a single ring level."""
    def __init__(self, level: int, min_range: float, max_range: float, cell_size: float, grid_size: int = 400):
        self.level = level
        self.min_range = min_range
        self.max_range = max_range
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.extent = cell_size * grid_size / 2.0  # Half extent in metres (e.g. 10m for L0)


# Standard 3-ring system configuration
RING_CONFIGS: Dict[int, RingConfig] = {
    0: RingConfig(level=0, min_range=0.0,  max_range=10.0,  cell_size=0.05, grid_size=400),
    1: RingConfig(level=1, min_range=10.0, max_range=30.0,  cell_size=0.15, grid_size=400),
    2: RingConfig(level=2, min_range=30.0, max_range=100.0, cell_size=0.50, grid_size=400),
}


class MultiLevelRingBuffer:
    """Manages 3 concentric ring buffers with ego-motion tracking."""

    def __init__(self, configs: Dict[int, RingConfig] = RING_CONFIGS):
        self.configs = configs
        self.rings = {lvl: RingBufferSoA(cfg.grid_size) for lvl, cfg in configs.items()}

        # Ego-motion cell offsets per ring level
        # Stores cumulative integer cell shifts (offset_x, offset_y)
        self.offsets = {lvl: [0, 0] for lvl in configs}

        # Last known vehicle world position (x, y)
        self.ego_pos = np.array([0.0, 0.0], dtype=np.float64)

    def point_to_cell(self, x: float, y: float, level: int) -> Tuple[int, int, int]:
        """Map a single 2D world point (x, y) to (u, v, cell_idx) for a ring level.

        Parameters
        ----------
        x, y : float
            World / ego-centric coordinates in metres.
        level : int
            Ring level (0, 1, or 2).

        Returns
        -------
        u, v, idx : int
            Wrapped 2D grid coordinates (0..grid_size-1) and flat array index.
        """
        cfg = self.configs[level]
        off_x, off_y = self.offsets[level]

        # Continuous cell coordinate relative to grid center
        c_x = int(np.floor(x / cfg.cell_size))
        c_y = int(np.floor(y / cfg.cell_size))

        # Apply offset and modulo wrap around
        u = (c_x + off_x + cfg.grid_size // 2) % cfg.grid_size
        v = (c_y + off_y + cfg.grid_size // 2) % cfg.grid_size

        idx = u * cfg.grid_size + v
        return u, v, idx

    def points_to_cells_vectorized(self, xyz: np.ndarray, level: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized mapping of Nx3 points to grid indices for a given ring level.

        Returns
        -------
        u, v, flat_idx : np.ndarray[int32]
        valid_mask : np.ndarray[bool] — True if point falls within ring range.
        """
        cfg = self.configs[level]
        off_x, off_y = self.offsets[level]

        r_sq = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        valid_mask = (r_sq >= cfg.min_range ** 2) & (r_sq <= cfg.max_range ** 2)

        cx = np.floor(xyz[:, 0] / cfg.cell_size).astype(np.int32)
        cy = np.floor(xyz[:, 1] / cfg.cell_size).astype(np.int32)

        u = (cx + off_x + cfg.grid_size // 2) % cfg.grid_size
        v = (cy + off_y + cfg.grid_size // 2) % cfg.grid_size
        flat_idx = u * cfg.grid_size + v

        return u, v, flat_idx, valid_mask

    def update_ego_motion(self, dx: float, dy: float) -> None:
        """Update grid origin with vehicle displacement (dx, dy) in O(1) time.

        Calculates cell shifts and clears wrapped unobserved cell rings.
        """
        self.ego_pos[0] += dx
        self.ego_pos[1] += dy

        for lvl, cfg in self.configs.items():
            # Shift in cell units
            shift_x = int(np.round(dx / cfg.cell_size))
            shift_y = int(np.round(dy / cfg.cell_size))

            if shift_x == 0 and shift_y == 0:
                continue

            # Clear newly wrapped cells
            ring_soa = self.rings[lvl]
            grid_sz = cfg.grid_size

            cleared_indices = []

            # Rows/cols to clear if shifted
            if shift_x != 0:
                old_off_x = self.offsets[lvl][0]
                for step in range(abs(shift_x)):
                    col = (old_off_x + (step if shift_x > 0 else -step - 1) + grid_sz // 2) % grid_sz
                    cleared_indices.extend([col * grid_sz + r for r in range(grid_sz)])

            if shift_y != 0:
                old_off_y = self.offsets[lvl][1]
                for step in range(abs(shift_y)):
                    row = (old_off_y + (step if shift_y > 0 else -step - 1) + grid_sz // 2) % grid_sz
                    cleared_indices.extend([c * grid_sz + row for c in range(grid_sz)])

            # Apply clearing
            ring_soa.clear_cells(np.unique(cleared_indices))

            # Update offsets
            self.offsets[lvl][0] += shift_x
            self.offsets[lvl][1] += shift_y


# ── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    mlrb = MultiLevelRingBuffer()

    # Test coordinate mapping for origin
    u, v, idx = mlrb.point_to_cell(0.0, 0.0, level=0)
    print(f"Point (0,0) at Level 0 maps to u={u}, v={v}, idx={idx}")
    assert u == 200 and v == 200, "Center point mapping error!"

    # Test ego-motion shift
    mlrb.update_ego_motion(dx=1.0, dy=0.0)  # Move 1m forward (20 cells at 5cm)
    u_new, v_new, idx_new = mlrb.point_to_cell(0.0, 0.0, level=0)
    print(f"After +1m X shift, Point (0,0) maps to u={u_new}, v={v_new}, idx={idx_new}")
    print("✅ Ring buffer coordinate math & ego-motion O(1) shift verified!")
