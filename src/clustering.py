from collections import deque

"""
clustering.py — Obstacle Point Cloud Clustering
=================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 2

Clusters non-ground points into candidate obstacle groups using
DBSCAN (scikit-learn) with an optional KD-tree accelerated Euclidean
clustering fallback.

Public API:
    cluster_points(points, **kwargs) -> labels_array

Usage:
    from clustering import cluster_points

    # points: (M, 3+) non-ground points
    cluster_ids = cluster_points(points)
    # cluster_ids: (M,) int array — -1 = noise, 0..K = cluster ID
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing sklearn for DBSCAN ──────────────────────────────────
_HAS_SKLEARN = False
try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    logger.warning(
        "scikit-learn not installed — only Euclidean clustering will be available. "
        "Install with: pip install scikit-learn"
    )


# =====================================================================
# Public API
# =====================================================================

def cluster_points(
    points: np.ndarray,
    *,
    backend: Optional[str] = None,
    eps: float = 0.5,
    min_samples: int = 10,
    min_cluster_size: int = 10,
    max_cluster_size: int = 50_000,
    max_points_for_dbscan: int = 80_000,
) -> np.ndarray:
    """Cluster non-ground points into candidate obstacle groups.

    Parameters
    ----------
    points : np.ndarray, shape (M, 3+)
        Non-ground XYZ[I…] point cloud.
    backend : str | None
        ``"dbscan"`` or ``"euclidean"``. ``None`` auto-selects DBSCAN if
        sklearn is available AND point count ≤ *max_points_for_dbscan*.
    eps : float
        Neighbourhood radius (metres). Default 0.5 m works well for
        urban LiDAR at ~10 Hz.
    min_samples : int
        DBSCAN core-point threshold.
    min_cluster_size : int
        Clusters smaller than this are relabelled as noise (-1).
    max_cluster_size : int
        Clusters larger than this are relabelled as noise (usually
        artefacts from wall merging).
    max_points_for_dbscan : int
        If point count exceeds this, fall back to the faster voxel-based
        Euclidean clustering even if sklearn is available.

    Returns
    -------
    labels : np.ndarray[int], shape (M,)
        Cluster ID per point.  ``-1`` = noise / unclustered.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected (M, 3+) array, got {points.shape}")

    M = points.shape[0]
    if M == 0:
        return np.array([], dtype=np.int32)

    # Auto-select backend
    if backend is None:
        if _HAS_SKLEARN and M <= max_points_for_dbscan:
            backend = "dbscan"
        else:
            backend = "euclidean"

    backend = backend.lower()
    t0 = time.perf_counter()

    if backend == "dbscan":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn required for DBSCAN backend")
        labels = _cluster_dbscan(
            points[:, :3], eps=eps, min_samples=min_samples,
        )
    elif backend == "euclidean":
        labels = _cluster_euclidean_voxel(
            points[:, :3], eps=eps, min_samples=min_samples,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    # Post-filter: remove too-small and too-large clusters
    labels = _filter_cluster_sizes(labels, min_cluster_size, max_cluster_size)

    # Re-number clusters to 0..K contiguously
    labels = _renumber_clusters(labels)

    elapsed = (time.perf_counter() - t0) * 1000
    n_clusters = labels.max() + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    logger.info(
        "Clustering (%s): %d points → %d clusters, %d noise (%.1f ms)",
        backend, M, n_clusters, n_noise, elapsed,
    )

    return labels


# =====================================================================
# Backend 1 — DBSCAN (scikit-learn)
# =====================================================================

def _cluster_dbscan(
    xyz: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> np.ndarray:
    """Run sklearn DBSCAN on the XYZ coordinates."""
    db = _DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="kd_tree",
        n_jobs=-1,
    )
    labels = db.fit_predict(xyz.astype(np.float64))
    return labels.astype(np.int32)


# =====================================================================
# Backend 2 — Voxel-based Euclidean Clustering (pure NumPy)
# =====================================================================

def _cluster_euclidean_voxel(
    xyz: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """Fast voxel-grid-based connected-component clustering.

    Steps:
      1. Voxelise the point cloud at resolution = eps.
      2. Build a hash-map of occupied voxels.
      3. BFS flood-fill among 26-connected neighbours.
      4. Map voxel cluster IDs back to point-level labels.

    This is O(N) on average and much faster than DBSCAN for large clouds.
    """
    voxel_size = eps
    # Quantise points to voxel indices
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int64)

    # Build voxel → list of point indices
    voxel_map: dict[tuple, list[int]] = {}
    for i in range(voxel_coords.shape[0]):
        key = (voxel_coords[i, 0], voxel_coords[i, 1], voxel_coords[i, 2])
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(i)

    # BFS flood-fill with 26-connectivity
    labels = np.full(xyz.shape[0], -1, dtype=np.int32)
    visited_voxels: set[tuple] = set()
    cluster_id = 0

    # Precompute 26 neighbour offsets
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))

    for voxel_key in voxel_map:
        if voxel_key in visited_voxels:
            continue

        # BFS
        queue = deque([voxel_key])
        visited_voxels.add(voxel_key)
        cluster_points_idx = []

        while queue:
            current = queue.popleft()
            cluster_points_idx.extend(voxel_map[current])

            for dx, dy, dz in offsets:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if nb in voxel_map and nb not in visited_voxels:
                    visited_voxels.add(nb)
                    queue.append(nb)

        if len(cluster_points_idx) >= min_samples:
            for idx in cluster_points_idx:
                labels[idx] = cluster_id
            cluster_id += 1

    return labels


# =====================================================================
# Post-processing helpers
# =====================================================================

def _filter_cluster_sizes(
    labels: np.ndarray,
    min_size: int,
    max_size: int,
) -> np.ndarray:
    """Relabel clusters outside [min_size, max_size] as noise (-1)."""
    out = labels.copy()
    if labels.max() < 0:
        return out

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    for cid, cnt in zip(unique, counts):
        if cnt < min_size or cnt > max_size:
            out[labels == cid] = -1
    return out


def _renumber_clusters(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster IDs to contiguous 0..K."""
    out = labels.copy()
    unique_valid = np.unique(labels[labels >= 0])
    mapping = {old: new for new, old in enumerate(unique_valid)}
    for old_id, new_id in mapping.items():
        out[labels == old_id] = new_id
    return out


# ── CLI quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)

    # Generate 5 synthetic clusters + noise
    clusters = []
    for i in range(5):
        center = rng.uniform(-20, 20, size=3)
        center[2] = rng.uniform(-1.0, 2.0)
        pts = center + rng.normal(0, 0.3, size=(200, 3))
        clusters.append(pts)

    noise = rng.uniform(-30, 30, size=(100, 3))
    all_pts = np.vstack(clusters + [noise]).astype(np.float32)

    print(f"Synthetic test — {all_pts.shape[0]} points, 5 expected clusters")
    labels = cluster_points(all_pts, eps=0.8, min_cluster_size=50)
    n_clusters = labels.max() + 1 if labels.max() >= 0 else 0
    print(f"  Found {n_clusters} clusters, {(labels == -1).sum()} noise points")
