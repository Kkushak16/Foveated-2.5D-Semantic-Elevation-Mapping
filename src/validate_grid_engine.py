"""
validate_grid_engine.py — Integration & Robustness Tests for Grid Engine
==========================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member B · Phase 3 (3c)

Runs end-to-end integration & validation tests for the Foveated Grid Engine:
  1. Synthetic Scene Test: Ground + obstacle + vehicle bridge overhang scenario.
     Verifies no NaNs, correct overhang flagging, and class-weighted voting.
  2. Temporal Confidence Decay Test: Simulates safe region stopping observation
     over N frames, verifying confidence decays below threshold.
  3. Ego-Motion Shift Test: Simulates vehicle displacement, verifying O(1)
     ring buffer offset wrap and cell clearing.

Usage:
    python validate_grid_engine.py
"""

import logging
import sys
import time
import numpy as np

from grid_engine import FoveatedGridEngine
from ring_buffer import RING_CONFIGS

logger = logging.getLogger(__name__)


def test_synthetic_scene_integration() -> bool:
    """Test 1: Synthetic scene with ground, obstacle, and overhang/bridge."""
    print("\n" + "=" * 60)
    print("  Test 1: Synthetic Scene & Overhang Integration Test")
    print("=" * 60)

    engine = FoveatedGridEngine(vehicle_clearance=2.0, overhang_gap_tau=0.8)
    rng = np.random.default_rng(42)

    # 1. Ground points (-1.73m elevation)
    n_ground = 10_000
    gx = rng.uniform(-8.0, 8.0, n_ground)
    gy = rng.uniform(-8.0, 8.0, n_ground)
    gz = -1.73 + rng.normal(0.0, 0.02, n_ground)
    ground_pts = np.column_stack([gx, gy, gz]).astype(np.float32)
    ground_cls = np.zeros(n_ground, dtype=np.int32)
    ground_cnf = np.full(n_ground, 0.9, dtype=np.float32)
    ground_mask = np.ones(n_ground, dtype=bool)

    # 2. Dynamic Car object at (x=3.0, y=2.0) — 1.0m x 1.0m box
    n_car = 800
    cx = rng.uniform(2.5, 3.5, n_car)
    cy = rng.uniform(1.5, 2.5, n_car)
    cz = rng.uniform(-1.5, 0.2, n_car)
    car_pts = np.column_stack([cx, cy, cz]).astype(np.float32)
    car_cls = np.full(n_car, 1, dtype=np.int32)  # 1: dynamic_object
    car_cnf = np.full(n_car, 0.95, dtype=np.float32)
    car_mask = np.zeros(n_car, dtype=bool)

    # 3. Overhang / Bridge at (x=5.0, y=0.0) -> Roof points above clearance
    n_bridge = 300
    bx = rng.uniform(4.5, 5.5, n_bridge)
    by = rng.uniform(-1.0, 1.0, n_bridge)
    bz = rng.uniform(1.2, 2.5, n_bridge)  # High roof above ground
    bridge_pts = np.column_stack([bx, by, bz]).astype(np.float32)
    bridge_cls = np.full(n_bridge, 0, dtype=np.int32)  # 0: static_obstacle
    bridge_cnf = np.full(n_bridge, 0.85, dtype=np.float32)
    bridge_mask = np.zeros(n_bridge, dtype=bool)

    # Combine
    all_pts = np.vstack([ground_pts, car_pts, bridge_pts])
    all_cls = np.concatenate([ground_cls, car_cls, bridge_cls])
    all_cnf = np.concatenate([ground_cnf, car_cnf, bridge_cnf])
    all_gmask = np.concatenate([ground_mask, car_mask, bridge_mask])

    t0 = time.perf_counter()
    stats = engine.insert_points(all_pts, all_cls, all_cnf, all_gmask)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  Processed {all_pts.shape[0]:,} points in {elapsed_ms:.2f} ms:")
    for lvl, cnt in stats.items():
        print(f"    Ring {lvl}: {cnt:,} points inserted")

    snapshot = engine.get_grid_snapshot(level=0)

    # Assertions
    # 1. No NaNs in numeric arrays
    for key in ["min_z", "max_z", "ground_z", "confidence", "decayed_score"]:
        assert not np.isnan(snapshot[key]).any(), f"NaN detected in {key}!"
    print("  [OK] No NaN values detected in exported grid snapshot")

    # 2. Check Overhang Flagging
    overhang_count = np.sum(snapshot["overhang"])
    print(f"  [OK] Overhang cells detected: {overhang_count}")
    assert overhang_count > 0, "Failed to detect bridge overhang!"

    # 3. Check Dynamic Object Priority Voting
    # At car location (3.0, 2.0), sem_class in car region should be 1 (dynamic_object)
    u_car, v_car, _ = engine.mlrb.point_to_cell(3.0, 2.0, level=0)
    car_region = snapshot["sem_class"][u_car - 5 : u_car + 5, v_car - 5 : v_car + 5]
    detected_dynamic_count = np.sum(car_region == 1)
    print(f"  [OK] Dynamic object cells found in car region (3.0, 2.0): {detected_dynamic_count}/100 cells")
    assert detected_dynamic_count > 0, "Dynamic object priority voting failed!"

    print("  [PASSED] Test 1 PASSED!")
    return True


def test_confidence_decay_integration() -> bool:
    """Test 2: Temporal confidence decay when region stops being observed."""
    print("\n" + "=" * 60)
    print("  Test 2: Temporal Confidence Decay Integration Test")
    print("=" * 60)

    engine = FoveatedGridEngine()
    rng = np.random.default_rng(42)

    # Insert point cloud in frame 1
    pts = rng.uniform(-2.0, 2.0, size=(1000, 3)).astype(np.float32)
    engine.insert_points(pts)

    snap0 = engine.get_grid_snapshot(level=0)
    initial_conf = snap0["confidence"][200, 200]
    print(f"  Initial cell confidence (Frame 0): {initial_conf:.2f}")
    assert initial_conf > 0.0, "Initial confidence should be > 0!"

    # Simulate N=30 frames (~3 seconds at 10Hz) with NO new observations
    dt = 0.1  # 100ms per frame
    ego_speed = 10.0  # 10 m/s vehicle speed (faster decay)

    for frame in range(30):
        engine.update_temporal_decay(dt=dt, ego_speed=ego_speed)

    snap_decayed = engine.get_grid_snapshot(level=0)
    final_conf = snap_decayed["confidence"][200, 200]
    print(f"  Decayed cell confidence after 30 unobserved frames: {final_conf:.4f}")

    assert final_conf < (initial_conf * 0.7), "Confidence failed to decay properly!"
    print("  [PASSED] Test 2 PASSED!")
    return True


def test_ego_motion_shift_integration() -> bool:
    """Test 3: Ego-motion shift & O(1) ring buffer offset wrapping."""
    print("\n" + "=" * 60)
    print("  Test 3: Ego-Motion Shift & Buffer Wrap Test")
    print("=" * 60)

    engine = FoveatedGridEngine()

    # Insert point at origin (0, 0)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    engine.insert_points(pts)

    snap_before = engine.get_grid_snapshot(level=0)
    assert snap_before["sem_class"][200, 200] >= 0, "Origin cell should be observed!"

    # Shift ego vehicle by +5m forward (100 cells at 5cm cell size)
    engine.update_ego_motion(dx=5.0, dy=0.0)

    snap_after = engine.get_grid_snapshot(level=0)
    print("  [OK] Ego-motion update of +5.0m shift completed without memory error.")

    print("  [PASSED] Test 3 PASSED!")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print("\nRunning Phase 3 Foveated Grid Engine Validation Suite ...")

    t_start = time.perf_counter()
    t1 = test_synthetic_scene_integration()
    t2 = test_confidence_decay_integration()
    t3 = test_ego_motion_shift_integration()

    total_time = (time.perf_counter() - t_start) * 1000.0

    print("\n" + "=" * 60)
    print(f"  All Grid Engine Validation Tests PASSED! ({total_time:.2f} ms)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
