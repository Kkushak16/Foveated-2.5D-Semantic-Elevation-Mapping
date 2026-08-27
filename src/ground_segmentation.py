"""
ground_segmentation.py — Ground Segmentation Wrapper
=====================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 1

Provides a **stable public interface**:

    segment_ground(points) -> ground_mask   (bool np.ndarray, shape (N,))

Two back-ends are supported:
  1. **Patchwork++** (``pypatchworkpp``)  — preferred, C++-fast
  2. **Multi-plane RANSAC**              — pure-NumPy fallback

The wrapper auto-detects whether ``pypatchworkpp`` is installed and falls
back transparently.  Member B's grid engine and Member C's dashboard consume
``segment_ground()`` directly — do NOT change its signature without
co-ordinating with the team.

Usage:
    from ground_segmentation import segment_ground

    mask = segment_ground(points)          # points: (N, 4) xyzi
    ground_pts     = points[mask]
    non_ground_pts = points[~mask]
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing Patchwork++ ─────────────────────────────────────────
_HAS_PATCHWORKPP = False
try:
    import pypatchworkpp
    _HAS_PATCHWORKPP = True
    logger.info("pypatchworkpp detected — using Patchwork++ back-end.")
except ImportError:
    logger.info(
        "pypatchworkpp not installed — falling back to RANSAC back-end. "
        "Install with: pip install pypatchworkpp"
    )


# =====================================================================
# Public API  (STABLE — do not change without team flag)
# =====================================================================

def segment_ground(
    points: np.ndarray,
    *,
    backend: Optional[str] = None,
    sensor_height: float = 1.73,
    **kwargs,
) -> np.ndarray:
    """Segment a point cloud into ground / non-ground.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3) or (N, 4)
        XYZ[I] point cloud.  Intensity column is optional.
    backend : str | None
        ``"patchworkpp"`` or ``"ransac"``.  ``None`` auto-selects the
        best available back-end.
    sensor_height : float
        Height of the LiDAR sensor above the ground plane (metres).
        Used by both back-ends for initial seed selection.
    **kwargs
        Forwarded to the chosen back-end's internal function.

    Returns
    -------
    ground_mask : np.ndarray[bool], shape (N,)
        ``True`` for points classified as ground.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Expected (N, 3+) array, got shape {points.shape}"
        )

    if backend is None:
        backend = "patchworkpp" if _HAS_PATCHWORKPP else "ransac"

    backend = backend.lower()
    if backend == "patchworkpp":
        if not _HAS_PATCHWORKPP:
            raise ImportError(
                "pypatchworkpp is not installed. "
                "Install with: pip install pypatchworkpp"
            )
        return _segment_patchworkpp(points, sensor_height=sensor_height, **kwargs)
    elif backend == "ransac":
        return _segment_ransac(points, sensor_height=sensor_height, **kwargs)
    else:
        raise ValueError(f"Unknown back-end: {backend!r}")


# =====================================================================
# Back-end 1 — Patchwork++
# =====================================================================

def _segment_patchworkpp(
    points: np.ndarray,
    *,
    sensor_height: float = 1.73,
    verbose: bool = False,
) -> np.ndarray:
    """Run Patchwork++ ground segmentation.

    Uses the ``pypatchworkpp`` Python bindings from
    https://github.com/url-kaist/patchwork-plusplus

    The API is:
        params = pypatchworkpp.Parameters()
        pw = pypatchworkpp.patchworkpp(params)
        pw.estimateGround(cloud_xyz)
        ground = pw.getGround()          # (M, 3) ground points
        nonground = pw.getNonground()    # (K, 3) non-ground points
    """
    # Ensure float64 contiguous array with at least 3 columns
    cloud = np.ascontiguousarray(points[:, :3], dtype=np.float64)

    params = pypatchworkpp.Parameters()
    params.sensor_height = sensor_height
    if verbose:
        params.verbose = True

    pw = pypatchworkpp.patchworkpp(params)
    pw.estimateGround(cloud)

    ground_pts = np.asarray(pw.getGround())     # (M, 3)
    # nonground_pts = np.asarray(pw.getNonground())  # (K, 3)

    # Map returned ground points back to a boolean mask over the
    # original input.  Patchwork++ returns the *actual points*, not
    # indices, so we need a nearest-neighbour match.
    ground_mask = _match_points_to_mask(points[:, :3], ground_pts)

    return ground_mask


def _match_points_to_mask(
    original: np.ndarray,
    subset: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """Create a boolean mask over *original* marking rows present in *subset*.

    Uses a hash-set approach for O(N+M) average-case performance.
    """
    if subset.shape[0] == 0:
        return np.zeros(original.shape[0], dtype=bool)

    # Quantise to avoid float precision issues, then use a set of tuples
    factor = 1.0 / tol
    subset_set = set(
        tuple(row)
        for row in np.round(subset[:, :3] * factor).astype(np.int64)
    )
    mask = np.array(
        [
            tuple(row) in subset_set
            for row in np.round(original[:, :3] * factor).astype(np.int64)
        ],
        dtype=bool,
    )
    return mask


# =====================================================================
# Back-end 2 — Multi-Plane RANSAC  (pure NumPy, CPU-only)
# =====================================================================

def _segment_ransac(
    points: np.ndarray,
    *,
    sensor_height: float = 1.73,
    max_distance: float = 0.15,
    num_iterations: int = 200,
    num_seeds: int = 3,
    z_range: Tuple[float, float] = (-3.0, 0.5),
    normal_z_thresh: float = 0.85,
    min_inlier_ratio: float = 0.02,
) -> np.ndarray:
    """Multi-plane RANSAC ground segmentation (pure NumPy fallback).

    Strategy:
      1. Filter candidate points by Z-range relative to sensor height.
      2. Run RANSAC to find the dominant horizontal plane.
      3. Mark inliers whose fitted-plane normal is roughly vertical
         (|n_z| > ``normal_z_thresh``) as ground.
      4. Repeat on remaining candidates to capture multi-level ground
         (e.g. road + sidewalk at different heights).

    Parameters
    ----------
    points : (N, 3+)
        Input point cloud.
    sensor_height : float
        LiDAR height above ground (m).
    max_distance : float
        RANSAC inlier distance threshold (m).
    num_iterations : int
        RANSAC iterations per plane extraction round.
    num_seeds : int
        How many planes to attempt extracting.
    z_range : (float, float)
        Acceptable Z-range *relative to -sensor_height* for candidate
        ground points.
    normal_z_thresh : float
        Minimum |n_z| for the fitted normal to be considered horizontal.
    min_inlier_ratio : float
        Minimum fraction of remaining candidates that must be inliers
        for a fitted plane to be accepted.

    Returns
    -------
    ground_mask : np.ndarray[bool], shape (N,)
    """
    xyz = points[:, :3].astype(np.float64)
    N = xyz.shape[0]
    ground_mask = np.zeros(N, dtype=bool)

    # Pre-filter: only consider points whose Z is plausibly ground
    expected_ground_z = -sensor_height
    z_lo = expected_ground_z + z_range[0]
    z_hi = expected_ground_z + z_range[1]
    candidate_mask = (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi)
    candidate_idx = np.nonzero(candidate_mask)[0]

    if candidate_idx.shape[0] < 10:
        logger.warning(
            "RANSAC: fewer than 10 candidate ground points (Z in [%.1f, %.1f]). "
            "Returning empty ground mask.",
            z_lo, z_hi,
        )
        return ground_mask

    remaining = candidate_idx.copy()

    rng = np.random.default_rng(seed=42)

    for seed_round in range(num_seeds):
        if remaining.shape[0] < 10:
            break

        best_inliers = np.array([], dtype=np.intp)
        best_normal = None

        pts_remaining = xyz[remaining]

        for _ in range(num_iterations):
            # Sample 3 random points
            sample_idx = rng.choice(pts_remaining.shape[0], size=3, replace=False)
            p1, p2, p3 = pts_remaining[sample_idx]

            # Fit plane via cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal /= norm_len

            # Ensure normal points upward (positive Z)
            if normal[2] < 0:
                normal = -normal

            # Check if plane is approximately horizontal
            if normal[2] < normal_z_thresh:
                continue

            # Distance of all remaining points to the plane
            d = np.abs(np.dot(pts_remaining - p1, normal))
            inlier_local = np.nonzero(d < max_distance)[0]

            if inlier_local.shape[0] > best_inliers.shape[0]:
                best_inliers = inlier_local
                best_normal = normal

        # Accept this plane?
        ratio = best_inliers.shape[0] / remaining.shape[0]
        if best_normal is not None and ratio >= min_inlier_ratio:
            global_inlier_idx = remaining[best_inliers]
            ground_mask[global_inlier_idx] = True
            # Remove these inliers from the remaining pool
            keep = np.ones(remaining.shape[0], dtype=bool)
            keep[best_inliers] = False
            remaining = remaining[keep]
            logger.debug(
                "RANSAC round %d: accepted plane (n_z=%.3f), %d inliers (%.1f%%)",
                seed_round,
                best_normal[2],
                best_inliers.shape[0],
                ratio * 100,
            )
        else:
            logger.debug(
                "RANSAC round %d: no acceptable plane found (best ratio=%.3f).",
                seed_round,
                ratio,
            )
            break

    return ground_mask


# ── CLI quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # Generate a synthetic point cloud with a flat ground plane + noise
    rng = np.random.default_rng(0)
    N = 50_000

    # Ground plane at z ≈ -1.73 (sensor height)
    ground = rng.uniform([-50, -50, -1.80], [50, 50, -1.65], size=(N // 2, 3))
    # Objects above ground
    objects = rng.uniform([-20, -20, -1.0], [20, 20, 3.0], size=(N // 2, 3))
    cloud = np.vstack([ground, objects])
    intensity = rng.uniform(0, 1, size=(cloud.shape[0], 1)).astype(np.float32)
    cloud_xyzi = np.hstack([cloud.astype(np.float32), intensity])

    mask = segment_ground(cloud_xyzi, backend="ransac")
    pct = mask.sum() / mask.shape[0] * 100
    print(f"\nSynthetic test — RANSAC back-end")
    print(f"  Total points : {mask.shape[0]:,}")
    print(f"  Ground points: {mask.sum():,}  ({pct:.1f}%)")
    print(f"  Expected ~50% ground for this synthetic scene.")
