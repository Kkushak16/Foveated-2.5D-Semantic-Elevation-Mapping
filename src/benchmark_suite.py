"""
benchmark_suite.py — Master Benchmark Suite & Headline Results Summary
========================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Phase 5 Deliverable · Master Benchmark Suite

Runs all benchmark modules sequentially and prints headline performance numbers:
  - Memory Footprint Reduction % vs 5cm Uniform Grid Baseline
  - Average CPU Throughput (FPS) & Pipeline Latency (ms)
  - Semantic Classification Accuracy & mIoU
  - Edge-case Robustness Validation Status

Usage:
    python -m src.benchmark_suite
    python run.py benchmark
"""

import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_memory import run_memory_benchmark
from src.benchmark_latency import run_latency_benchmark
from src.benchmark_accuracy import run_accuracy_benchmark
from src.test_robustness import run_robustness_suite

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "=" * 80)
    print("  FOVEATED 2.5D LIDAR GRID MAPPING — MASTER BENCHMARK SUITE")
    print("=" * 80 + "\n")

    t_start = time.perf_counter()

    # 1. Memory Benchmark
    res_mem = run_memory_benchmark(n_frames=10)

    # 2. Latency Benchmark
    res_lat = run_latency_benchmark(n_frames=10)

    # 3. Accuracy Benchmark
    res_acc = run_accuracy_benchmark(n_samples=5)

    # 4. Robustness Suite
    res_rob = run_robustness_suite()

    total_duration = time.perf_counter() - t_start

    # Executive Summary Headline Banner
    print("\n" + "=" * 80)
    print("  EXECUTIVE HEADLINE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  * Memory Footprint Saved : {res_mem['saved_percent']:.1f}% ({res_mem['mlrb_grid_mb']:.1f} MB MLRB vs {res_mem['baseline_uniform_mb']:.1f} MB Baseline)")
    print(f"  * Average Pipeline FPS   : {res_lat['avg_fps']:.1f} FPS ({res_lat['avg_latency_ms']:.1f} ms latency on CPU)")
    print(f"  * Semantic Accuracy      : {res_acc['accuracy_pct']:.1f}% Overall (mIoU: {res_acc['mIoU']:.4f})")
    print(f"  * System Robustness      : {'ALL TESTS PASSED' if res_rob['all_passed'] else 'ATTENTION REQUIRED'}")
    print(f"  * Total Suite Duration   : {total_duration:.2f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
