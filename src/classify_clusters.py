"""
classify_clusters.py — Inference-Time Cluster Classification
==============================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 2

Provides the **stable public interface** consumed by Member B's grid engine:

    classify_clusters(points, ground_mask) -> List[ClusterInfo]

Each ClusterInfo dict contains:
    {
        "cluster_id": int,
        "class": str,          # "static_obstacle" | "dynamic_object" | "pole_wall" | "other"
        "class_id": int,       # 0 | 1 | 2 | 3
        "confidence": float,   # max class probability from RF
        "points": np.ndarray,  # (K, 3+) points belonging to this cluster
        "centroid": np.ndarray, # (3,) cluster centroid
        "bbox_min": np.ndarray, # (3,) axis-aligned bounding box min
        "bbox_max": np.ndarray, # (3,) axis-aligned bounding box max
    }

Usage:
    from classify_clusters import classify_clusters

    results = classify_clusters(points, ground_mask)
    for obj in results:
        print(f"Cluster {obj['cluster_id']}: {obj['class']} "
              f"({obj['confidence']:.2f}), {obj['points'].shape[0]} pts")
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
CLASS_NAMES = ["static_obstacle", "dynamic_object", "pole_wall", "other"]
DEFAULT_MODEL_PATH = Path(__file__).parent / "obstacle_classifier.joblib"

# Type alias for readability
ClusterInfo = Dict[str, Any]


# =====================================================================
# Public API  (STABLE — consumed by Member B's grid engine)
# =====================================================================

def classify_clusters(
    points: np.ndarray,
    ground_mask: np.ndarray,
    *,
    model_path: Optional[Union[str, Path]] = None,
    cluster_kwargs: Optional[dict] = None,
    return_noise: bool = False,
) -> List[ClusterInfo]:
    """Run the full clustering + classification pipeline on one frame.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3+)
        Full point cloud (ground + non-ground).
    ground_mask : np.ndarray[bool], shape (N,)
        Boolean mask from ``segment_ground()``. True = ground.
    model_path : str | Path | None
        Path to the trained ``.joblib`` classifier.
        ``None`` → uses ``DEFAULT_MODEL_PATH``.
        If the model file doesn't exist, falls back to a rule-based
        heuristic classifier.
    cluster_kwargs : dict | None
        Extra kwargs forwarded to ``cluster_points()``.
    return_noise : bool
        If True, include noise points (cluster_id = -1) as a single
        extra entry.

    Returns
    -------
    clusters : list[ClusterInfo]
        One dict per detected obstacle cluster.
    """
    from clustering import cluster_points
    from feature_extraction import extract_features

    t0 = time.perf_counter()

    # ── Step 1: extract non-ground points ──
    non_ground_pts = points[~ground_mask]
    if non_ground_pts.shape[0] < 5:
        logger.info("classify_clusters: no non-ground points.")
        return []

    # ── Step 2: cluster ──
    ck = cluster_kwargs or {}
    cluster_ids = cluster_points(non_ground_pts, **ck)

    n_clusters = cluster_ids.max() + 1 if cluster_ids.max() >= 0 else 0
    if n_clusters == 0:
        logger.info("classify_clusters: no clusters found.")
        return []

    # ── Step 3: load or create classifier ──
    clf = _load_classifier(model_path)

    # ── Step 4: classify each cluster ──
    results: List[ClusterInfo] = []

    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if mask.sum() == 0:
            continue

        cluster_pts = non_ground_pts[mask]
        features = extract_features(cluster_pts, has_intensity=(points.shape[1] >= 4))

        if clf is not None:
            probs = clf.predict_proba(features.reshape(1, -1))[0]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
        else:
            # Rule-based fallback
            class_id, confidence = _heuristic_classify(features)

        centroid = cluster_pts[:, :3].mean(axis=0)
        bbox_min = cluster_pts[:, :3].min(axis=0)
        bbox_max = cluster_pts[:, :3].max(axis=0)

        results.append({
            "cluster_id": cid,
            "class": CLASS_NAMES[class_id],
            "class_id": class_id,
            "confidence": confidence,
            "points": cluster_pts,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "classify_clusters: %d clusters classified in %.1f ms",
        len(results), elapsed_ms,
    )

    return results


# =====================================================================
# Internal helpers
# =====================================================================

_cached_clf = None
_cached_clf_path = None


def _load_classifier(
    model_path: Optional[Union[str, Path]] = None,
):
    """Load the trained model with caching. Falls back to None."""
    global _cached_clf, _cached_clf_path

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    model_path = Path(model_path)

    # Return cached if same path
    if _cached_clf is not None and _cached_clf_path == model_path:
        return _cached_clf

    if model_path.is_file():
        try:
            import joblib
            _cached_clf = joblib.load(model_path)
            _cached_clf_path = model_path
            logger.info("Loaded classifier from %s", model_path)
            return _cached_clf
        except Exception as exc:
            logger.warning("Failed to load classifier: %s — using heuristics.", exc)
            return None
    else:
        logger.info(
            "No trained model at %s — using rule-based heuristic classifier. "
            "Train one with: python train_classifier.py --synthetic",
            model_path,
        )
        return None


def _heuristic_classify(features: np.ndarray) -> tuple:
    """Simple rule-based classification when no trained model exists.

    Uses the 14-dim feature vector. Returns (class_id, confidence).
    """
    from feature_extraction import FEATURE_NAMES

    height = features[0]
    width = features[1]
    length = features[2]
    linearity = features[13]
    point_count = features[6]
    z_mean = features[8]

    # Pole/wall: tall & narrow with high linearity
    if linearity > 0.6 and height > 1.5 and max(width, length) < 1.0:
        return 2, 0.70

    # Dynamic object: car-like dimensions
    if 1.0 < height < 3.0 and 1.5 < length < 7.0 and 1.0 < width < 3.0:
        return 1, 0.55

    # Static obstacle: large objects
    if height > 2.0 or (length > 5.0 and width > 3.0) or point_count > 1000:
        return 0, 0.50

    # Default → other
    return 3, 0.40


# ── CLI quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    rng = np.random.default_rng(42)
    sensor_height = 1.73
    ground_z = -sensor_height

    # Build synthetic scene
    # Ground
    n_ground = 30_000
    gx = rng.uniform(-50, 50, n_ground)
    gy = rng.uniform(-50, 50, n_ground)
    gz = ground_z + rng.normal(0, 0.03, n_ground)
    gi = rng.uniform(0.05, 0.3, n_ground)
    ground = np.column_stack([gx, gy, gz, gi]).astype(np.float32)

    # Car-like object
    n_car = 300
    cx = rng.normal(5, 0.8, n_car)
    cy = rng.normal(3, 1.5, n_car)
    cz = rng.uniform(ground_z + 0.1, ground_z + 1.8, n_car)
    ci = rng.uniform(0.3, 0.6, n_car)
    car = np.column_stack([cx, cy, cz, ci]).astype(np.float32)

    # Pole
    n_pole = 60
    px = rng.normal(-10, 0.05, n_pole)
    py = rng.normal(-5, 0.05, n_pole)
    pz = rng.uniform(ground_z, ground_z + 4.0, n_pole)
    pi_ = rng.uniform(0.4, 0.8, n_pole)
    pole = np.column_stack([px, py, pz, pi_]).astype(np.float32)

    # Building wall
    n_wall = 800
    wx = rng.normal(-20, 0.1, n_wall)
    wy = rng.uniform(-15, 15, n_wall)
    wz = rng.uniform(ground_z, ground_z + 5.0, n_wall)
    wi = rng.uniform(0.1, 0.4, n_wall)
    wall = np.column_stack([wx, wy, wz, wi]).astype(np.float32)

    points = np.vstack([ground, car, pole, wall])
    ground_mask = np.zeros(points.shape[0], dtype=bool)
    ground_mask[:n_ground] = True

    print("Running classify_clusters on synthetic scene ...")
    results = classify_clusters(points, ground_mask)

    print(f"\nDetected {len(results)} clusters:")
    for obj in results:
        print(
            f"  Cluster {obj['cluster_id']:2d}: {obj['class']:18s} "
            f"(conf={obj['confidence']:.2f}), "
            f"{obj['points'].shape[0]:4d} pts, "
            f"centroid=({obj['centroid'][0]:+.1f}, "
            f"{obj['centroid'][1]:+.1f}, "
            f"{obj['centroid'][2]:+.1f})"
        )
