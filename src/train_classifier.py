"""
train_classifier.py — Train & Save the Random Forest Classifier
=================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 2

Trains a Random Forest on SemanticKITTI labels mapped to our 4 target classes:
    0 — static_obstacle   (buildings, fences, vegetation, …)
    1 — dynamic_object     (cars, trucks, bicyclists, pedestrians, …)
    2 — pole_wall          (poles, traffic signs, walls, …)
    3 — other / unlabelled

Ground is excluded (handled by Phase 1's segment_ground).

The trained model is saved as ``obstacle_classifier.joblib``.

Usage:
    python train_classifier.py /path/to/semantickitti \\
        --sequences 00 01 02 03 04 05 06 07 09 10 \\
        --max-frames 50 \\
        --output obstacle_classifier.joblib

    # Quick synthetic training (no dataset needed):
    python train_classifier.py --synthetic
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Target class mapping ──────────────────────────────────────────────
# SemanticKITTI label → our 4-class target
# Reference: SemanticKITTI label definitions
# https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml

CLASS_NAMES = ["static_obstacle", "dynamic_object", "pole_wall", "other"]
NUM_CLASSES = len(CLASS_NAMES)

# SemanticKITTI label → target class ID
LABEL_TO_TARGET: Dict[int, int] = {
    # ── static_obstacle (0) ──
    10: 0,   # car (parked → static)
    13: 0,   # bus (parked)
    15: 0,   # motorcycle (parked)
    16: 0,   # on-rails
    18: 0,   # truck (parked)
    20: 0,   # other-vehicle (parked)
    50: 0,   # building
    51: 0,   # fence
    52: 0,   # other-structure
    70: 0,   # vegetation
    71: 0,   # trunk
    # ── dynamic_object (1) ──
    11: 1,   # bicycle
    30: 1,   # person
    31: 1,   # bicyclist
    32: 1,   # motorcyclist
    252: 1,  # moving-car
    253: 1,  # moving-bicyclist
    254: 1,  # moving-person
    255: 1,  # moving-motorcyclist
    256: 1,  # moving-on-rails
    257: 1,  # moving-bus
    258: 1,  # moving-truck
    259: 1,  # moving-other-vehicle
    # ── pole_wall (2) ──
    80: 2,   # pole
    81: 2,   # traffic-sign
    99: 2,   # other-object (often wall-like)
    # ── ground labels — EXCLUDED ──
    # 40: road, 44: parking, 48: sidewalk, 49: other-ground,
    # 60: lane-marking, 72: terrain
    # ── other / unlabelled (3) ──
    0: 3,    # unlabelled
    1: 3,    # outlier
}

# Ground label IDs (skip these entirely)
GROUND_LABELS = frozenset({40, 44, 48, 49, 60, 72})


def semantickitti_label_to_target(label: int) -> Optional[int]:
    """Map a SemanticKITTI label to our target class.

    Returns None for ground labels (should be excluded before clustering).
    """
    if label in GROUND_LABELS:
        return None
    return LABEL_TO_TARGET.get(label, 3)  # default → "other"


# ── Training data construction ────────────────────────────────────────

def build_training_data_from_clusters(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    ground_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (features, targets) for all clusters in one frame.

    Parameters
    ----------
    points : (N, 4)  full point cloud
    cluster_labels : (M,) cluster IDs for non-ground points (M = ~N-ground)
    gt_labels : (N,) SemanticKITTI semantic labels
    ground_mask : (N,) bool — True for ground

    Returns
    -------
    X : (K, 14) feature matrix
    y : (K,) target class IDs
    """
    from feature_extraction import extract_features, NUM_FEATURES

    non_ground_pts = points[~ground_mask]
    non_ground_gt = gt_labels[~ground_mask]

    if cluster_labels.max() < 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.array([], dtype=np.int32)

    n_clusters = cluster_labels.max() + 1
    X = np.zeros((n_clusters, NUM_FEATURES), dtype=np.float32)
    y = np.zeros(n_clusters, dtype=np.int32)

    valid_mask = np.ones(n_clusters, dtype=bool)

    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            valid_mask[cid] = False
            continue

        # Extract features
        cluster_pts = non_ground_pts[mask]
        X[cid] = extract_features(cluster_pts, has_intensity=True)

        # Majority-vote target class from GT labels
        cluster_gt = non_ground_gt[mask]
        target_votes = np.zeros(NUM_CLASSES, dtype=int)
        for lbl in cluster_gt:
            tgt = semantickitti_label_to_target(int(lbl))
            if tgt is not None:
                target_votes[tgt] += 1

        if target_votes.sum() == 0:
            valid_mask[cid] = False
        else:
            y[cid] = target_votes.argmax()

    return X[valid_mask], y[valid_mask]


def generate_synthetic_training_data(
    n_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for testing without a real dataset.

    Creates clusters with realistic feature distributions per class:
      - static_obstacle: large, boxy, high density
      - dynamic_object: medium, elongated, moderate density
      - pole_wall: tall & narrow, high linearity
      - other: random
    """
    from feature_extraction import NUM_FEATURES

    rng = np.random.default_rng(42)
    samples_per_class = n_samples // NUM_CLASSES

    X_list, y_list = [], []

    for cls_id in range(NUM_CLASSES):
        X_cls = np.zeros((samples_per_class, NUM_FEATURES), dtype=np.float32)

        if cls_id == 0:  # static_obstacle — large boxy
            X_cls[:, 0] = rng.uniform(1.5, 6.0, samples_per_class)   # height
            X_cls[:, 1] = rng.uniform(2.0, 10.0, samples_per_class)  # width
            X_cls[:, 2] = rng.uniform(2.0, 15.0, samples_per_class)  # length
            X_cls[:, 6] = rng.uniform(200, 5000, samples_per_class)  # point_count
            X_cls[:, 8] = rng.uniform(0.0, 3.0, samples_per_class)   # z_mean
            X_cls[:, 9] = rng.uniform(0.5, 3.0, samples_per_class)   # z_var
            X_cls[:, 11] = rng.uniform(0.1, 0.5, samples_per_class)  # intensity_mean
            X_cls[:, 13] = rng.uniform(0.0, 0.4, samples_per_class)  # linearity

        elif cls_id == 1:  # dynamic_object — medium, elongated
            X_cls[:, 0] = rng.uniform(1.0, 2.5, samples_per_class)   # height
            X_cls[:, 1] = rng.uniform(1.5, 2.5, samples_per_class)   # width
            X_cls[:, 2] = rng.uniform(3.0, 6.0, samples_per_class)   # length
            X_cls[:, 6] = rng.uniform(50, 800, samples_per_class)    # point_count
            X_cls[:, 8] = rng.uniform(-0.5, 1.5, samples_per_class)  # z_mean
            X_cls[:, 9] = rng.uniform(0.1, 1.5, samples_per_class)   # z_var
            X_cls[:, 11] = rng.uniform(0.2, 0.7, samples_per_class)  # intensity_mean
            X_cls[:, 13] = rng.uniform(0.1, 0.5, samples_per_class)  # linearity

        elif cls_id == 2:  # pole_wall — tall, narrow, high linearity
            X_cls[:, 0] = rng.uniform(2.0, 8.0, samples_per_class)   # height
            X_cls[:, 1] = rng.uniform(0.05, 0.5, samples_per_class)  # width (thin)
            X_cls[:, 2] = rng.uniform(0.05, 1.0, samples_per_class)  # length (thin)
            X_cls[:, 6] = rng.uniform(15, 200, samples_per_class)    # point_count
            X_cls[:, 8] = rng.uniform(0.5, 4.0, samples_per_class)   # z_mean
            X_cls[:, 9] = rng.uniform(1.0, 5.0, samples_per_class)   # z_var
            X_cls[:, 11] = rng.uniform(0.3, 0.9, samples_per_class)  # intensity_mean
            X_cls[:, 13] = rng.uniform(0.6, 0.99, samples_per_class) # linearity (high)

        else:  # other — random
            X_cls[:, 0] = rng.uniform(0.1, 3.0, samples_per_class)
            X_cls[:, 1] = rng.uniform(0.1, 5.0, samples_per_class)
            X_cls[:, 2] = rng.uniform(0.1, 5.0, samples_per_class)
            X_cls[:, 6] = rng.uniform(10, 500, samples_per_class)
            X_cls[:, 8] = rng.uniform(-1.0, 3.0, samples_per_class)
            X_cls[:, 9] = rng.uniform(0.0, 3.0, samples_per_class)
            X_cls[:, 11] = rng.uniform(0.0, 1.0, samples_per_class)
            X_cls[:, 13] = rng.uniform(0.0, 1.0, samples_per_class)

        # Derived features
        X_cls[:, 3] = X_cls[:, 1] / np.maximum(X_cls[:, 2], 1e-6)  # aspect_ratio_wl
        X_cls[:, 4] = X_cls[:, 0] / np.maximum(X_cls[:, 2], 1e-6)  # aspect_ratio_hw
        X_cls[:, 5] = X_cls[:, 0] * X_cls[:, 1] * X_cls[:, 2]      # volume
        X_cls[:, 7] = X_cls[:, 6] / np.maximum(X_cls[:, 5], 1e-9)  # point_density
        X_cls[:, 10] = X_cls[:, 0]  # z_range = height
        X_cls[:, 12] = rng.uniform(0.0, 0.3, samples_per_class)     # intensity_std

        # Add noise to all features
        noise = rng.normal(0, 0.05, X_cls.shape)
        X_cls = np.abs(X_cls + noise)  # keep positive

        X_list.append(X_cls)
        y_list.append(np.full(samples_per_class, cls_id, dtype=np.int32))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    order = rng.permutation(X.shape[0])
    return X[order], y[order]


# ── Training ──────────────────────────────────────────────────────────

def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple:
    """Train a Random Forest and return (model, metrics_dict).

    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
    metrics : dict with keys 'accuracy', 'report', 'confusion_matrix',
              'feature_importances'
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    logger.info(
        "Training Random Forest: %d train / %d test samples, %d features",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    from feature_extraction import FEATURE_NAMES
    fi = dict(zip(FEATURE_NAMES, clf.feature_importances_))

    metrics = {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importances": fi,
        "train_time_s": train_time,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
    }

    logger.info("Training done in %.2f s — accuracy: %.4f", train_time, acc)

    return clf, metrics


def save_model(model, path: str) -> None:
    """Save trained model to disk."""
    import joblib
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str):
    """Load a trained model from disk."""
    import joblib
    return joblib.load(path)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the obstacle Random Forest classifier."
    )
    parser.add_argument(
        "dataset_root", nargs="?", default=None,
        help="Path to SemanticKITTI root.",
    )
    parser.add_argument(
        "--sequences", nargs="+", default=None,
        help="Which sequences to use for training.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Max frames per sequence to use.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic training data (no dataset needed).",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of RF trees.",
    )
    parser.add_argument(
        "--output", "-o", default="obstacle_classifier.joblib",
        help="Output model file path.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if args.synthetic:
        print("Generating synthetic training data ...")
        X, y = generate_synthetic_training_data(n_samples=4000)
        print(f"  → {X.shape[0]} samples, {NUM_CLASSES} classes")
    else:
        if args.dataset_root is None:
            parser.error("Provide dataset path or use --synthetic.")

        from dataset_loader import SemanticKITTILoader
        from ground_segmentation import segment_ground
        from clustering import cluster_points

        loader = SemanticKITTILoader(
            args.dataset_root,
            sequences=args.sequences,
        )

        X_all, y_all = [], []
        for seq_id, frames in loader.iter_sequences():
            frame_count = 0
            for fid, points, labels in frames:
                if labels is None:
                    continue

                # Ground segmentation
                ground_mask = segment_ground(points, backend="ransac")
                non_ground = points[~ground_mask]

                if non_ground.shape[0] < 20:
                    continue

                # Cluster non-ground
                cluster_ids = cluster_points(non_ground)

                # Build training samples
                X_frame, y_frame = build_training_data_from_clusters(
                    points, cluster_ids, labels, ground_mask,
                )

                if X_frame.shape[0] > 0:
                    X_all.append(X_frame)
                    y_all.append(y_frame)

                frame_count += 1
                if frame_count >= args.max_frames:
                    break

            print(f"  Seq {seq_id}: {frame_count} frames processed")

        if not X_all:
            print("ERROR: No training data collected!")
            sys.exit(1)

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        print(f"\nTotal training data: {X.shape[0]} clusters")
        for c in range(NUM_CLASSES):
            print(f"  {CLASS_NAMES[c]:20s}: {(y == c).sum():,}")

    # Train
    print("\nTraining Random Forest ...")
    clf, metrics = train_random_forest(X, y, n_estimators=args.n_estimators)

    # Print results
    print("\n" + "=" * 60)
    print("  Classification Report")
    print("=" * 60)
    print(metrics["report"])
    print(f"  Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"  Training time:    {metrics['train_time_s']:.2f} s")

    print("\n  Feature importances (top 5):")
    fi_sorted = sorted(
        metrics["feature_importances"].items(),
        key=lambda x: x[1], reverse=True,
    )
    for fname, imp in fi_sorted[:5]:
        print(f"    {fname:20s} : {imp:.4f}")

    print("\n  Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("=" * 60)

    # Save
    save_model(clf, args.output)
    print(f"\n✅ Model saved to: {args.output}")


if __name__ == "__main__":
    main()
