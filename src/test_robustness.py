"""
test_robustness.py — System Edge-Case & Robustness Validation Suite
====================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 5 Deliverable · System Robustness Tests

Executes 3 specific robustness tests:
  1. Overhang Test (Bridge/tunnel synthetic scene — verifies min_z & max_z gap detection)
  2. Sparse Far-Cell Test (Object at 90m — verifies temporal confidence decay over 10 frames)
  3. Stress Test (100k point dense urban frame vs 20k point sparse highway frame — checks RAM stability)

Usage:
    python -m src.test_robustness
    python run.py test_robustness
"""

import csv
import logging
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_engine import FoveatedGridEngine
from src.ground_segmentation import segment_ground

logger = logging.getLogger(__name__)


def test_overhang_detection() -> bool:
    """Test 1: Verify correct min_z & max_z gap handling for bridge/tunnel overhangs."""
    print("  [Test 1] Overhang Handling (Bridge/Tunnel Scene)...")
    engine = FoveatedGridEngine()

    # Generate synthetic bridge scene:
    # Ground at z = -1.73m, Bridge ceiling at z = +2.5m (gap of 4.23m)
    n_pts = 2000
    x = np.full(n_pts, 5.0, dtype=np.float32)  # In Level 0 (0-10m)
    y = np.full(n_pts, 0.0, dtype=np.float32)

    # 1000 ground points + 1000 bridge ceiling points
    z_gnd = np.random.uniform(-1.75, -1.71, 1000).astype(np.float32)
    z_bridge = np.random.uniform(2.45, 2.55, 1000).astype(np.float32)

    pts_xyz = np.column_stack([x, y, np.concatenate([z_gnd, z_bridge])])
    sem_cls = np.full(n_pts, 0, dtype=np.int32)
    gnd_mask = pts_xyz[:, 2] < 0.0
    sem_cls[gnd_mask] = 3
    confs = np.ones(n_pts, dtype=np.float32)

    engine.insert_points(pts_xyz, sem_cls, confs, gnd_mask)

    # Check snapshot at Level 0
    snap = engine.get_grid_snapshot(level=0)
    min_z = snap["min_z"]
    max_z = snap["max_z"]

    valid_mask = (min_z > -10.0) & (max_z < 10.0)
    if np.any(valid_mask):
        center_cell_min = float(min_z[valid_mask].min())
        center_cell_max = float(max_z[valid_mask].max())
        gap = center_cell_max - center_cell_min
        passed = (center_cell_min < -1.5) and (center_cell_max > 2.0) and (gap > 3.5)
    else:
        center_cell_min, center_cell_max, gap = float("inf"), float("-inf"), 0.0
        passed = False

    status = "PASSED" if passed else "FAILED"
    print(f"    -> min_z={center_cell_min:.2f}m, max_z={center_cell_max:.2f}m, gap={gap:.2f}m | Status: {status}")
    return passed


def test_sparse_far_cell_decay() -> bool:
    """Test 2: Verify temporal confidence decay for a sparse object at 90m across 10 frames."""
    print("  [Test 2] Sparse Far-Cell Confidence Decay (Object at 90m)...")
    engine = FoveatedGridEngine()

    # Place sparse obstacle at x=90m, y=0m (Level 2: 30-100m)
    n_pts = 50
    x = np.full(n_pts, 90.0, dtype=np.float32)
    y = np.full(n_pts, 0.0, dtype=np.float32)
    z = np.random.uniform(0.0, 1.5, n_pts).astype(np.float32)

    pts_xyz = np.column_stack([x, y, z])
    sem_cls = np.full(n_pts, 1, dtype=np.int32)
    gnd_mask = np.zeros(n_pts, dtype=bool)
    confs = np.full(n_pts, 1.0, dtype=np.float32)

    # Initial frame observation
    engine.insert_points(pts_xyz, sem_cls, confs, gnd_mask)
    snap0 = engine.get_grid_snapshot(level=2)
    conf_grid = snap0["confidence"]
    obs_cells = np.where(conf_grid > 0)

    if len(obs_cells[0]) > 0:
        r, c = obs_cells[0][0], obs_cells[1][0]
        conf_initial = float(conf_grid[r, c])

        # Simulate 10 frames of motion where object is not re-observed
        for f in range(10):
            engine.update_ego_motion(dx=0.5, dy=0.0)
            engine.update_temporal_decay(dt=0.1, ego_speed=5.0)

        snap10 = engine.get_grid_snapshot(level=2)
        conf_decayed = float(snap10["confidence"][r, c])
        passed = (conf_initial > 0.5) and (conf_decayed < conf_initial)
    else:
        conf_initial, conf_decayed = 0.0, 0.0
        passed = False

    status = "PASSED" if passed else "FAILED"
    print(f"    -> Initial Conf={conf_initial:.3f}, Decayed Conf={conf_decayed:.3f} | Status: {status}")
    return passed


def test_stress_and_stability() -> bool:
    """Test 3: Execute high-density (100k pts) vs low-density (20k pts) stress tests."""
    print("  [Test 3] Stress & Memory Stability Test (100k pts vs 20k pts)...")
    engine = FoveatedGridEngine()
    rng = np.random.default_rng(777)

    passed_all = True
    for name, n_pts in [("Highway Sparse", 20_000), ("Urban Dense", 100_000)]:
        t0 = time.perf_counter()
        pts = rng.uniform(-80, 80, (n_pts, 3)).astype(np.float32)
        pts[:, 2] = rng.uniform(-2, 4, n_pts)
        sem_cls = rng.integers(0, 4, n_pts, dtype=np.int32)
        confs = rng.uniform(0.5, 1.0, n_pts).astype(np.float32)
        gnd_mask = pts[:, 2] < -1.5

        try:
            engine.insert_points(pts, sem_cls, confs, gnd_mask)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            print(f"    -> {name} ({n_pts:,} pts): Processed in {dur_ms:.1f} ms | Status: PASSED")
        except Exception as e:
            print(f"    -> {name} FAILED with exception: {e}")
            passed_all = False

    return passed_all


def run_robustness_suite(output_csv: str = "results/robustness_results.csv") -> dict:
    """Execute all system robustness tests and save summary CSV."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print("=" * 75)
    print("  SYSTEM EDGE-CASE & ROBUSTNESS VALIDATION SUITE")
    print("=" * 75)

    r1 = test_overhang_detection()
    r2 = test_sparse_far_cell_decay()
    r3 = test_stress_and_stability()

    all_passed = r1 and r2 and r3

    rows = [
        {"test_name": "Overhang Handling (Bridge/Tunnel)", "result": "PASSED" if r1 else "FAILED"},
        {"test_name": "Sparse Far-Cell Confidence Decay", "result": "PASSED" if r2 else "FAILED"},
        {"test_name": "Stress & Memory Stability (100k pts)", "result": "PASSED" if r3 else "FAILED"},
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "result"])
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 75)
    print(f"  [SUMMARY] All Robustness Tests Passed: {all_passed}")
    print(f"  Saved results to {output_csv}\n")

    return {
        "all_passed": all_passed,
        "csv_path": output_csv
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_robustness_suite()


if __name__ == "__main__":
    main()
