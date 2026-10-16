"""Reading the Kaggle ``asl-signs`` files (or the synthetic stand-in, which has the same layout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gesturesign import landmarks as lm
from gesturesign.types import Frames

PARQUET_COLUMNS = ("frame", "row_id", "type", "landmark_index", "x", "y", "z")
TRAIN_CSV_COLUMNS = ("path", "participant_id", "sequence_id", "sign")


class DataError(ValueError):
    """The dataset on disk does not have the expected layout."""


def read_train_csv(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train.csv"
    if not path.exists():
        raise DataError(
            f"{path} not found. Download Kaggle asl-signs into {data_dir} "
            "(or pass --data-dir), or run `gesturesign synth` for a synthetic stand-in."
        )
    table = pd.read_csv(path)
    missing = set(TRAIN_CSV_COLUMNS) - set(table.columns)
    if missing:
        raise DataError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    table["participant_id"] = table["participant_id"].astype(np.int64)
    table["sequence_id"] = table["sequence_id"].astype(np.int64)
    table["sign"] = table["sign"].astype(str)
    return table


def load_landmarks(path: Path) -> Frames:
    """One parquet file as [T, 543, 3] float32, frames in order, NaN where missing.

    Points are placed by (type, landmark_index) rather than by row order, so a
    file with shuffled rows or missing rows still loads correctly.
    """
    table = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    offsets = table["type"].map(lm.TYPE_OFFSETS)
    if offsets.isna().any():
        bad = sorted(set(table.loc[offsets.isna(), "type"].astype(str)))
        raise DataError(f"{path}: unknown landmark types {bad}")
    sizes = table["type"].map(lm.TYPE_SIZES).to_numpy(dtype=np.int64)
    index = table["landmark_index"].to_numpy(dtype=np.int64)
    if np.any((index < 0) | (index >= sizes)):
        raise DataError(f"{path}: landmark_index out of range for its type")
    frame_ids, frame_pos = np.unique(table["frame"].to_numpy(), return_inverse=True)
    out = np.full((len(frame_ids), lm.N_POINTS, 3), np.nan, dtype=np.float32)
    point = offsets.to_numpy(dtype=np.int64) + index
    out[frame_pos, point] = table[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return out
