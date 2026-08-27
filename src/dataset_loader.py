"""
dataset_loader.py — SemanticKITTI Dataset Loader
=================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member A · Phase 1

Loads SemanticKITTI .bin point cloud files and .label files, yielding
(points[N, 4], labels[N]) per frame, batched by sequence.

SemanticKITTI directory layout expected:
    dataset/
    └── sequences/
        ├── 00/
        │   ├── velodyne/
        │   │   ├── 000000.bin
        │   │   ├── 000001.bin
        │   │   └── ...
        │   └── labels/
        │       ├── 000000.label
        │       ├── 000001.label
        │       └── ...
        ├── 01/
        └── ...

Usage:
    loader = SemanticKITTILoader("/path/to/dataset")
    for seq_id, frames in loader.iter_sequences():
        for frame_id, points, labels in frames:
            # points: np.ndarray shape (N, 4)  — x, y, z, intensity
            # labels: np.ndarray shape (N,)    — semantic label (lower 16 bits)
            ...
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import (
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# ── SemanticKITTI ground-truth label IDs that correspond to "ground" ──
GROUND_LABEL_IDS = frozenset({
    40,   # road
    44,   # parking
    48,   # sidewalk
    49,   # other-ground
    60,   # lane-marking
    72,   # terrain
})


class SemanticKITTILoader:
    """Lazy, memory-efficient loader for the SemanticKITTI dataset.

    Parameters
    ----------
    dataset_root : str | Path
        Path to the SemanticKITTI root directory that contains a
        ``sequences/`` sub-folder.
    sequences : list[str] | None
        Which sequences to load (e.g. ``["00", "01"]``).
        ``None`` means auto-detect all available sequences.
    has_labels : bool
        If ``True`` (default), load corresponding ``.label`` files.
        Set to ``False`` for test sequences that lack labels.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        sequences: Optional[List[str]] = None,
        has_labels: bool = True,
    ) -> None:
        self.root = Path(dataset_root)
        self.seq_dir = self.root / "sequences"
        self.has_labels = has_labels

        if not self.seq_dir.is_dir():
            raise FileNotFoundError(
                f"SemanticKITTI sequences directory not found: {self.seq_dir}"
            )

        if sequences is not None:
            self.sequences = sorted(sequences)
        else:
            self.sequences = sorted(
                d.name
                for d in self.seq_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            )
        logger.info("SemanticKITTI loader initialised — sequences: %s", self.sequences)

    # ── public helpers ────────────────────────────────────────────────

    def num_sequences(self) -> int:
        """Return the number of sequences discovered."""
        return len(self.sequences)

    def frame_ids(self, seq: str) -> List[str]:
        """Return sorted list of frame IDs (stem names) in *seq*."""
        vel_dir = self.seq_dir / seq / "velodyne"
        if not vel_dir.is_dir():
            return []
        return sorted(p.stem for p in vel_dir.glob("*.bin"))

    # ── single-frame I/O ─────────────────────────────────────────────

    @staticmethod
    def load_points(bin_path: Union[str, Path]) -> np.ndarray:
        """Load a ``.bin`` point cloud file → ``(N, 4)`` float32 array.

        Columns are ``[x, y, z, intensity]``.
        """
        points = np.fromfile(str(bin_path), dtype=np.float32)
        return points.reshape(-1, 4)

    @staticmethod
    def load_labels(label_path: Union[str, Path]) -> np.ndarray:
        """Load a ``.label`` file → ``(N,)`` uint32 array of semantic IDs.

        SemanticKITTI packs *semantic_label* in the lower 16 bits and
        *instance_id* in the upper 16 bits.  This returns only the
        semantic label.
        """
        raw = np.fromfile(str(label_path), dtype=np.uint32)
        return (raw & 0xFFFF).astype(np.uint32)

    # ── generators ────────────────────────────────────────────────────

    def iter_frames(
        self,
        seq: str,
    ) -> Generator[Tuple[str, np.ndarray, Optional[np.ndarray]], None, None]:
        """Yield ``(frame_id, points, labels)`` for every frame in *seq*.

        Corrupt / missing frames are logged and skipped — the generator
        never raises on a single bad frame.
        """
        vel_dir = self.seq_dir / seq / "velodyne"
        lbl_dir = self.seq_dir / seq / "labels"

        for fid in self.frame_ids(seq):
            bin_path = vel_dir / f"{fid}.bin"
            try:
                points = self.load_points(bin_path)
            except Exception as exc:
                logger.warning(
                    "Skipping frame %s/%s — failed to load points: %s",
                    seq, fid, exc,
                )
                continue

            labels: Optional[np.ndarray] = None
            if self.has_labels:
                lbl_path = lbl_dir / f"{fid}.label"
                if lbl_path.is_file():
                    try:
                        labels = self.load_labels(lbl_path)
                        if labels.shape[0] != points.shape[0]:
                            logger.warning(
                                "Frame %s/%s — point/label count mismatch "
                                "(%d vs %d); discarding labels.",
                                seq, fid, points.shape[0], labels.shape[0],
                            )
                            labels = None
                    except Exception as exc:
                        logger.warning(
                            "Frame %s/%s — failed to load labels: %s",
                            seq, fid, exc,
                        )
                else:
                    logger.debug(
                        "Frame %s/%s — label file missing, yielding None.",
                        seq, fid,
                    )

            yield fid, points, labels

    def iter_sequences(
        self,
    ) -> Generator[
        Tuple[str, Generator[Tuple[str, np.ndarray, Optional[np.ndarray]], None, None]],
        None,
        None,
    ]:
        """Yield ``(seq_id, frame_generator)`` for every sequence."""
        for seq in self.sequences:
            yield seq, self.iter_frames(seq)

    # ── convenience: load single frame ────────────────────────────────

    def load_frame(
        self,
        seq: str,
        frame_id: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load a single frame by sequence + frame ID.

        Returns ``(points, labels)`` — labels may be ``None``.
        """
        bin_path = self.seq_dir / seq / "velodyne" / f"{frame_id}.bin"
        points = self.load_points(bin_path)

        labels: Optional[np.ndarray] = None
        if self.has_labels:
            lbl_path = self.seq_dir / seq / "labels" / f"{frame_id}.label"
            if lbl_path.is_file():
                labels = self.load_labels(lbl_path)

        return points, labels

    # ── ground-truth helpers ──────────────────────────────────────────

    @staticmethod
    def ground_truth_mask(labels: np.ndarray) -> np.ndarray:
        """Return a boolean mask that is True for ground-class points.

        Uses the SemanticKITTI ground label IDs:
        40 (road), 44 (parking), 48 (sidewalk), 49 (other-ground),
        60 (lane-marking), 72 (terrain).
        """
        mask = np.zeros(labels.shape, dtype=bool)
        for gid in GROUND_LABEL_IDS:
            mask |= labels == gid
        return mask


# ── CLI quick-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python dataset_loader.py <path/to/semantickitti>")
        sys.exit(1)

    loader = SemanticKITTILoader(sys.argv[1])
    print(f"Found {loader.num_sequences()} sequence(s): {loader.sequences}")

    for seq_id, frames in loader.iter_sequences():
        count = 0
        for fid, pts, lbl in frames:
            count += 1
            lbl_info = f", labels {lbl.shape}" if lbl is not None else ""
            print(f"  seq {seq_id} / frame {fid}: points {pts.shape}{lbl_info}")
            if count >= 3:
                print(f"  ... (showing first 3 frames of seq {seq_id})")
                break
