"""
validate_ground_seg.py — Visual + Numeric Sanity Check
=======================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 1

Quick validation script that:
  1. Loads a single SemanticKITTI frame (or uses a synthetic scene).
  2. Runs ``segment_ground()`` to produce a ground mask.
  3. Prints numeric stats (% ground, precision/recall vs GT if available).
  4. Opens an Open3D visualisation with ground (green) / non-ground (red).

Usage:
    # With a real SemanticKITTI dataset:
    python validate_ground_seg.py /path/to/semantickitti --seq 00 --frame 000000

    # Synthetic test (no dataset required):
    python validate_ground_seg.py --synthetic

    # Force RANSAC back-end:
    python validate_ground_seg.py --synthetic --backend ransac
"""

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np

# Local imports
from dataset_loader import SemanticKITTILoader, GROUND_LABEL_IDS
from ground_segmentation import segment_ground

logger = logging.getLogger(__name__)


# ── Colour palette ────────────────────────────────────────────────────
COLOR_GROUND     = np.array([0.18, 0.80, 0.25])   # green
COLOR_NONGROUND  = np.array([0.85, 0.20, 0.18])   # red
COLOR_TP         = np.array([0.10, 0.75, 0.10])   # bright green  (true positive)
COLOR_FP         = np.array([1.00, 0.55, 0.00])   # orange        (false positive)
COLOR_FN         = np.array([0.20, 0.20, 0.85])   # blue          (false negative)
COLOR_TN         = np.array([0.70, 0.70, 0.70])   # grey          (true negative)


def generate_synthetic_scene(
    n_points: int = 80_000,
    sensor_height: float = 1.73,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic LiDAR scene with known ground truth.

    Returns ``(points_xyzi, gt_ground_mask)``.
    """
    rng = np.random.default_rng(42)
    ground_z = -sensor_height

    # ── Ground plane (flat with slight noise) ──
    n_ground = int(n_points * 0.45)
    gx = rng.uniform(-60, 60, n_ground)
    gy = rng.uniform(-60, 60, n_ground)
    gz = ground_z + rng.normal(0, 0.03, n_ground)
    gi = rng.uniform(0.05, 0.35, n_ground)
    ground = np.column_stack([gx, gy, gz, gi]).astype(np.float32)

    # ── Sidewalk (slightly elevated ground) ──
    n_sidewalk = int(n_points * 0.10)
    sx = rng.uniform(-30, 30, n_sidewalk)
    sy = rng.uniform(8, 12, n_sidewalk)  # narrow strip
    sz = ground_z + 0.12 + rng.normal(0, 0.02, n_sidewalk)
    si = rng.uniform(0.1, 0.4, n_sidewalk)
    sidewalk = np.column_stack([sx, sy, sz, si]).astype(np.float32)

    # ── Non-ground: cars, pedestrians, trees ──
    n_objects = n_points - n_ground - n_sidewalk
    ox = rng.uniform(-40, 40, n_objects)
    oy = rng.uniform(-40, 40, n_objects)
    oz = rng.uniform(ground_z + 0.3, ground_z + 4.0, n_objects)
    oi = rng.uniform(0.1, 1.0, n_objects)
    objects = np.column_stack([ox, oy, oz, oi]).astype(np.float32)

    points = np.vstack([ground, sidewalk, objects])
    gt_mask = np.zeros(points.shape[0], dtype=bool)
    gt_mask[: n_ground + n_sidewalk] = True

    # Shuffle so ground isn't just the first rows
    order = rng.permutation(points.shape[0])
    return points[order], gt_mask[order]


def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    """Compute precision, recall, F1, IoU for ground segmentation."""
    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }


def print_report(
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    elapsed_ms: float = 0.0,
) -> None:
    """Print a formatted report to stdout."""
    total = pred_mask.shape[0]
    n_ground = int(pred_mask.sum())
    pct = n_ground / total * 100 if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("  Ground Segmentation — Validation Report")
    print("=" * 60)
    print(f"  Total points      : {total:>10,}")
    print(f"  Predicted ground   : {n_ground:>10,}  ({pct:.1f}%)")
    print(f"  Predicted non-ground: {total - n_ground:>10,}  ({100 - pct:.1f}%)")
    print(f"  Inference time     : {elapsed_ms:>10.1f} ms")

    if gt_mask is not None:
        gt_ground = int(gt_mask.sum())
        gt_pct = gt_ground / total * 100 if total > 0 else 0.0
        m = compute_metrics(pred_mask, gt_mask)
        print(f"\n  Ground-truth ground: {gt_ground:>10,}  ({gt_pct:.1f}%)")
        print(f"  TP: {m['TP']:,}  FP: {m['FP']:,}  FN: {m['FN']:,}  TN: {m['TN']:,}")
        print(f"  Precision : {m['Precision']:.4f}")
        print(f"  Recall    : {m['Recall']:.4f}")
        print(f"  F1 Score  : {m['F1']:.4f}")
        print(f"  IoU       : {m['IoU']:.4f}")

    print("=" * 60 + "\n")


def visualise_open3d(
    points: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    window_name: str = "Ground Segmentation Validation",
    point_size: float = 1.5,
) -> None:
    """Open an Open3D window showing ground vs. non-ground coloring.

    If *gt_mask* is provided, uses a 4-colour confusion-matrix palette:
    TP=green, FP=orange, FN=blue, TN=grey.
    Otherwise uses simple green/red.
    """
    try:
        import open3d as o3d
    except ImportError:
        print(
            "[WARN] open3d not installed — skipping visualisation.\n"
            "       Install with: pip install open3d"
        )
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

    colors = np.zeros((points.shape[0], 3), dtype=np.float64)

    if gt_mask is not None:
        # Confusion-matrix coloring
        tp = pred_mask & gt_mask
        fp = pred_mask & ~gt_mask
        fn = ~pred_mask & gt_mask
        tn = ~pred_mask & ~gt_mask
        colors[tp] = COLOR_TP
        colors[fp] = COLOR_FP
        colors[fn] = COLOR_FN
        colors[tn] = COLOR_TN
        print("  Visualisation legend:")
        print("    Green  = True Positive  (correctly identified ground)")
        print("    Orange = False Positive  (predicted ground, actually not)")
        print("    Blue   = False Negative  (missed ground)")
        print("    Grey   = True Negative   (correctly identified non-ground)")
    else:
        colors[pred_mask] = COLOR_GROUND
        colors[~pred_mask] = COLOR_NONGROUND
        print("  Visualisation legend:")
        print("    Green = Ground")
        print("    Red   = Non-ground")

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Downsample if huge cloud for smooth rendering
    if points.shape[0] > 500_000:
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        print(f"  (downsampled to {len(pcd.points):,} points for rendering)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.05, 0.05, 0.10])

    # Set a nice initial viewpoint (top-down-ish)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.4)
    ctr.set_front([0, -0.3, -1.0])
    ctr.set_up([0, 0, 1])

    print("\n  [Open3D] Window opened — close it to continue.\n")
    vis.run()
    vis.destroy_window()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ground segmentation on SemanticKITTI or synthetic data."
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Path to SemanticKITTI dataset root.",
    )
    parser.add_argument(
        "--seq", default="00",
        help="Sequence ID (default: 00).",
    )
    parser.add_argument(
        "--frame", default="000000",
        help="Frame ID (default: 000000).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic scene instead of a real dataset.",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Force back-end: 'patchworkpp' or 'ransac'.",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip Open3D visualisation.",
    )
    parser.add_argument(
        "--sensor-height", type=float, default=1.73,
        help="Sensor height above ground (m, default 1.73).",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # ── Load or generate data ──
    gt_mask: Optional[np.ndarray] = None

    if args.synthetic:
        print("Generating synthetic LiDAR scene ...")
        points, gt_mask = generate_synthetic_scene(
            sensor_height=args.sensor_height,
        )
        print(f"  → {points.shape[0]:,} points generated.")
    else:
        if args.dataset_root is None:
            parser.error(
                "Provide a dataset path or use --synthetic."
            )
        loader = SemanticKITTILoader(args.dataset_root)
        print(f"Loading seq {args.seq} / frame {args.frame} ...")
        points, labels = loader.load_frame(args.seq, args.frame)
        print(f"  → {points.shape[0]:,} points loaded.")

        if labels is not None:
            gt_mask = SemanticKITTILoader.ground_truth_mask(labels)
            print(f"  → Ground-truth labels available "
                  f"({gt_mask.sum():,} ground points).")

    # ── Run segmentation ──
    print(f"\nRunning ground segmentation "
          f"(backend={args.backend or 'auto'}) ...")
    t0 = time.perf_counter()
    pred_mask = segment_ground(
        points,
        backend=args.backend,
        sensor_height=args.sensor_height,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # ── Report ──
    print_report(pred_mask, gt_mask=gt_mask, elapsed_ms=elapsed_ms)

    # ── Visualisation ──
    if not args.no_vis:
        visualise_open3d(points, pred_mask, gt_mask=gt_mask)
    else:
        print("  (Visualisation skipped — use without --no-vis to view.)")


if __name__ == "__main__":
    main()
