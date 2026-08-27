"""
feature_extraction.py — Per-Cluster Hand-Crafted Feature Computation
=====================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 2

Extracts a fixed-length feature vector from each cluster for the
Random Forest classifier.  Feature vector (14 dims):

    [ height, width, length,              # OBB extents
      aspect_ratio_wl, aspect_ratio_hw,   # shape ratios
      volume,                             # OBB volume
      point_count, point_density,         # density stats
      z_mean, z_variance, z_range,        # vertical stats
      intensity_mean, intensity_std,      # reflectivity
      linearity ]                         # PCA eigenvalue ratio

Usage:
    from feature_extraction import extract_features, FEATURE_NAMES

    features = extract_features(cluster_points)  # (14,) np.float32
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Ordered feature names — useful for model inspection / feature importance
FEATURE_NAMES: List[str] = [
    "height",             # 0  max_z - min_z
    "width",              # 1  OBB extent along 2nd principal axis
    "length",             # 2  OBB extent along 1st principal axis
    "aspect_ratio_wl",    # 3  width / length
    "aspect_ratio_hw",    # 4  height / length
    "volume",             # 5  height * width * length
    "point_count",        # 6  number of points
    "point_density",      # 7  point_count / volume
    "z_mean",             # 8  mean Z of the cluster
    "z_variance",         # 9  variance of Z
    "z_range",            # 10 max_z - min_z (same as height, kept for explicitness)
    "intensity_mean",     # 11 mean intensity (0 if not available)
    "intensity_std",      # 12 std intensity
    "linearity",          # 13 (λ1 - λ2) / λ1  from PCA eigenvalues
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(
    points: np.ndarray,
    *,
    has_intensity: bool = True,
) -> np.ndarray:
    """Extract a fixed-length feature vector from a single cluster.

    Parameters
    ----------
    points : np.ndarray, shape (K, 3) or (K, 4)
        Points belonging to one cluster. Columns: x, y, z [, intensity].
    has_intensity : bool
        If True and points have ≥4 columns, use column 3 as intensity.

    Returns
    -------
    features : np.ndarray, shape (14,)  dtype float32
    """
    K = points.shape[0]
    feat = np.zeros(NUM_FEATURES, dtype=np.float32)

    if K < 3:
        feat[6] = float(K)
        return feat

    xyz = points[:, :3].astype(np.float64)

    # ── Vertical extents ──
    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
    height = z_max - z_min
    z_mean = xyz[:, 2].mean()
    z_var = xyz[:, 2].var()

    feat[0] = height                 # height
    feat[8] = z_mean                 # z_mean
    feat[9] = z_var                  # z_variance
    feat[10] = height                # z_range

    # ── PCA on XY for oriented bounding box ──
    xy = xyz[:, :2]
    xy_centered = xy - xy.mean(axis=0)

    try:
        cov = np.cov(xy_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx_sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
    except np.linalg.LinAlgError:
        eigenvalues = np.array([1.0, 1.0])
        eigenvectors = np.eye(2)

    # Project XY onto principal axes
    projected = xy_centered @ eigenvectors
    p_min = projected.min(axis=0)
    p_max = projected.max(axis=0)
    extents = p_max - p_min
    length = max(extents[0], 1e-6)  # along 1st principal axis
    width = max(extents[1], 1e-6)   # along 2nd principal axis

    feat[1] = width
    feat[2] = length
    feat[3] = width / length                        # aspect_ratio_wl
    feat[4] = height / length if length > 0 else 0   # aspect_ratio_hw

    volume = height * width * length
    feat[5] = volume                                  # volume
    feat[6] = float(K)                                # point_count
    feat[7] = K / volume if volume > 1e-9 else 0      # point_density

    # ── 3D linearity from PCA eigenvalues ──
    # Full 3D PCA
    xyz_centered = xyz - xyz.mean(axis=0)
    try:
        cov3 = np.cov(xyz_centered, rowvar=False)
        evals3 = np.linalg.eigvalsh(cov3)
        evals3 = np.sort(evals3)[::-1]
        lam1 = max(evals3[0], 1e-12)
        lam2 = max(evals3[1], 1e-12)
        linearity = (lam1 - lam2) / lam1
    except (np.linalg.LinAlgError, IndexError):
        linearity = 0.0

    feat[13] = linearity

    # ── Intensity statistics ──
    if has_intensity and points.shape[1] >= 4:
        intensity = points[:, 3].astype(np.float64)
        feat[11] = intensity.mean()
        feat[12] = intensity.std()

    return feat


def extract_features_batch(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    has_intensity: bool = True,
) -> np.ndarray:
    """Extract features for all clusters in a labelled point cloud.

    Parameters
    ----------
    points : (M, 3+)
        Full (non-ground) point cloud.
    cluster_labels : (M,)
        Cluster IDs from ``cluster_points()``. ``-1`` = noise (skipped).
    has_intensity : bool
        Whether column 3 is intensity.

    Returns
    -------
    features : np.ndarray, shape (K, 14)
        One row per cluster (ordered by cluster ID 0..K-1).
    """
    if cluster_labels.max() < 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)

    n_clusters = cluster_labels.max() + 1
    features = np.zeros((n_clusters, NUM_FEATURES), dtype=np.float32)

    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        features[cid] = extract_features(
            points[mask], has_intensity=has_intensity,
        )

    return features


# ── CLI quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    # Simulate a "car" cluster
    car_pts = rng.normal([0, 0, 0], [2.0, 0.9, 0.7], size=(500, 3))
    car_pts[:, 2] += -0.5  # sits near ground
    car_int = rng.uniform(0.2, 0.6, size=(500, 1))
    car = np.hstack([car_pts, car_int]).astype(np.float32)

    # Simulate a "pole" cluster
    pole_pts = rng.normal([5, 5, 1.5], [0.1, 0.1, 1.5], size=(80, 3))
    pole_int = rng.uniform(0.3, 0.8, size=(80, 1))
    pole = np.hstack([pole_pts, pole_int]).astype(np.float32)

    print("Feature extraction test:")
    for name, pts in [("car", car), ("pole", pole)]:
        f = extract_features(pts)
        print(f"\n  {name} ({pts.shape[0]} pts):")
        for fname, val in zip(FEATURE_NAMES, f):
            print(f"    {fname:20s} = {val:.4f}")
