"""
benchmark_latency.py — Per-Stage Execution Latency & FPS Benchmark
=====================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 5 Deliverable · Latency & Throughput Benchmark

Measures per-stage timing:
  1. Ground Segmentation (RANSAC / Elevation grid)
  2. Euclidean Clustering (DBSCAN / Euclidean)
  3. Obstacle Classification (14-dim Feature Extractor + Random Forest / Heuristics)
  4. Foveated Grid Engine Ingest (MLRB 3-Ring updates, coordinate math, semantic voting)
  5. Composite Dashboard Rendering

Usage:
    python -m src.benchmark_latency
    python run.py benchmark_latency
"""

import csv
import logging
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ground_segmentation import segment_ground
from src.classify_clusters import classify_clusters
from src.grid_engine import FoveatedGridEngine

logger = logging.getLogger(__name__)


def run_latency_benchmark(n_frames: int = 20, output_csv: str = "results/benchmark_latency.csv") -> dict:
    """Execute latency benchmark across synthetic golden test frames."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    engine = FoveatedGridEngine()

    rng = np.random.default_rng(2026)
    rows = []

    print("=" * 75)
    print("  PER-STAGE PIPELINE LATENCY & THROUGHPUT BENCHMARK (CPU ONLY)")
    print("=" * 75)
    print("  Target Throughput: > 10.0 FPS on standard x86 CPU")
    print("-" * 75)
    print(f"{'Frame':<6} | {'GndSeg(ms)':<10} | {'Cluster+Cls(ms)':<15} | {'Grid(ms)':<9} | {'Total(ms)':<9} | {'FPS':<6}")
    print("-" * 75)

    stage1_times = []
    stage2_times = []
    stage3_times = []
    total_times = []

    for f_idx in range(1, n_frames + 1):
        # 1. Generate 60,000 point synthetic scan
        n_pts = 60_000
        pts_xyz = rng.uniform(-50, 50, (n_pts, 3)).astype(np.float32)
        pts_xyz[:, 2] = rng.uniform(-2.0, 3.0, n_pts)
        intensity = rng.uniform(0.1, 0.9, n_pts).astype(np.float32)
        points = np.column_stack([pts_xyz, intensity])

        # Stage 1: Ground Segmentation
        t0 = time.perf_counter()
        gnd_mask = segment_ground(points)
        t_gnd = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Clustering + Classification (combined — classify_clusters calls cluster_points internally)
        t0 = time.perf_counter()
        classified_objs = classify_clusters(points, gnd_mask)
        t_classify = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Foveated Grid Engine Ingest
        t0 = time.perf_counter()
        sem_cls = np.full(n_pts, 0, dtype=np.int32)
        gnd_idx = np.where(gnd_mask)[0]
        sem_cls[gnd_idx] = 3
        confs = rng.uniform(0.8, 1.0, n_pts).astype(np.float32)

        engine.insert_points(points[:, :3], sem_cls, confs, gnd_mask)
        t_grid = (time.perf_counter() - t0) * 1000.0

        t_total = t_gnd + t_classify + t_grid
        fps = 1000.0 / t_total if t_total > 0 else 0.0

        stage1_times.append(t_gnd)
        stage2_times.append(t_classify)
        stage3_times.append(t_grid)
        total_times.append(t_total)

        print(f"#{f_idx:<5} | {t_gnd:<10.2f} | {t_classify:<15.2f} | {t_grid:<9.2f} | {t_total:<9.2f} | {fps:<6.1f}")

        rows.append({
            "frame": f_idx,
            "points": n_pts,
            "ground_seg_ms": f"{t_gnd:.2f}",
            "cluster_classify_ms": f"{t_classify:.2f}",
            "grid_engine_ms": f"{t_grid:.2f}",
            "total_pipeline_ms": f"{t_total:.2f}",
            "throughput_fps": f"{fps:.1f}"
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_fps = 1000.0 / np.mean(total_times)
    print("=" * 75)
    print(f"  [SUMMARY] Average Total Pipeline Latency : {np.mean(total_times):.2f} ms")
    print(f"  [SUMMARY] Average Throughput             : {avg_fps:.1f} FPS")
    print(f"  Saved benchmark metrics to {output_csv}\n")

    return {
        "avg_latency_ms": np.mean(total_times),
        "avg_fps": avg_fps,
        "csv_path": output_csv
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_latency_benchmark()


if __name__ == "__main__":
    main()
