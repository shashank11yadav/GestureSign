"""Reading the Kaggle ``asl-signs`` files (or the synthetic stand-in, which has the same layout).

The data can be a folder or the competition zip itself (``asl-signs.zip``,
40 GB compressed, 56 GB extracted). Reading parquet members straight out of
the zip avoids needing the extra disk space; each worker process opens its own
handle on the zip.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import pandas as pd

from gesturesign import landmarks as lm
from gesturesign.types import Frames

PARQUET_COLUMNS = ("frame", "row_id", "type", "landmark_index", "x", "y", "z")
TRAIN_CSV_COLUMNS = ("path", "participant_id", "sequence_id", "sign")
ZIP_NAME = "asl-signs.zip"


class DataError(ValueError):
    """The dataset on disk does not have the expected layout."""


class DataSource(Protocol):
    location: Path

    def exists(self, rel: str) -> bool: ...

    @contextmanager
    def open(self, rel: str) -> Iterator[BinaryIO]: ...


class FolderSource:
    def __init__(self, location: Path) -> None:
        self.location = location

    def exists(self, rel: str) -> bool:
        return (self.location / rel).exists()

    @contextmanager
    def open(self, rel: str) -> Iterator[BinaryIO]:
        with (self.location / rel).open("rb") as fh:
            yield fh


class ZipSource:
    def __init__(self, location: Path) -> None:
        self.location = location
        self._zip: zipfile.ZipFile | None = None

    @property
    def zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.location)
        return self._zip

    def exists(self, rel: str) -> bool:
        try:
            self.zip.getinfo(rel)
        except KeyError:
            return False
        return True

    @contextmanager
    def open(self, rel: str) -> Iterator[BinaryIO]:
        # Parquet readers seek, so hand them the member's bytes (a few hundred KB).
        yield io.BytesIO(self.zip.read(rel))


_SOURCES: dict[Path, DataSource] = {}


def open_source(data: Path) -> DataSource:
    """A folder with train.csv, the asl-signs zip, or a folder holding that zip.

    Sources are cached per process, so worker processes each keep one zip handle.
    """
    data = data.expanduser().resolve()
    if data in _SOURCES:
        return _SOURCES[data]
    source: DataSource
    if data.is_file() and zipfile.is_zipfile(data):
        source = ZipSource(data)
    elif (data / "train.csv").exists():
        source = FolderSource(data)
    elif (data / ZIP_NAME).is_file():
        source = ZipSource(data / ZIP_NAME)
    else:
        raise DataError(
            f"no train.csv or {ZIP_NAME} in {data}. Download Kaggle asl-signs there "
            "(or pass --data-dir), or run `gesturesign synth` for a synthetic stand-in."
        )
    _SOURCES[data] = source
    return source


def read_train_csv(data: Path | DataSource) -> pd.DataFrame:
    source = open_source(data) if isinstance(data, Path) else data
    if not source.exists("train.csv"):
        raise DataError(
            f"train.csv not found in {source.location}. Run `gesturesign synth` for a stand-in."
        )
    with source.open("train.csv") as fh:
        table = pd.read_csv(fh)
    missing = set(TRAIN_CSV_COLUMNS) - set(table.columns)
    if missing:
        raise DataError(f"train.csv is missing columns: {', '.join(sorted(missing))}")
    table["participant_id"] = table["participant_id"].astype(np.int64)
    table["sequence_id"] = table["sequence_id"].astype(np.int64)
    table["sign"] = table["sign"].astype(str)
    return table


def load_landmarks(path: Path | BinaryIO, name: str | None = None) -> Frames:
    """One parquet file as [T, 543, 3] float32, frames in order, NaN where missing.

    Points are placed by (type, landmark_index) rather than by row order, so a
    file with shuffled rows or missing rows still loads correctly.
    """
    label = name or str(path)
    table = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    offsets = table["type"].map(lm.TYPE_OFFSETS)
    if offsets.isna().any():
        bad = sorted(set(table.loc[offsets.isna(), "type"].astype(str)))
        raise DataError(f"{label}: unknown landmark types {bad}")
    sizes = table["type"].map(lm.TYPE_SIZES).to_numpy(dtype=np.int64)
    index = table["landmark_index"].to_numpy(dtype=np.int64)
    if np.any((index < 0) | (index >= sizes)):
        raise DataError(f"{label}: landmark_index out of range for its type")
    frame_ids, frame_pos = np.unique(table["frame"].to_numpy(), return_inverse=True)
    out = np.full((len(frame_ids), lm.N_POINTS, 3), np.nan, dtype=np.float32)
    point = offsets.to_numpy(dtype=np.int64) + index
    out[frame_pos, point] = table[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return out


def load_sequence(source: DataSource, rel: str) -> Frames:
    with source.open(rel) as fh:
        return load_landmarks(fh, name=rel)
