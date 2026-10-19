"""``prepare``: parquet files -> a compact, sharded ``.npz`` landmark cache.

Each clip is loaded, resampled to the model frame rate and squeezed to at most
``2 * max_frames`` frames (training squeezes again to ``max_frames`` after time
augmentation, exactly as the browser does). Only the 115 points any feature
set reads are kept (both hands, the full pose and 40 lip points), as float16
x/y. That turns ~40 GB of parquet into a cache that fits in memory.

Features are computed from this cache when batches are built, so that
augmentation can act on coordinates and the feature set can change without
re-reading the parquet files. Shards are written independently, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gesturesign import landmarks as lm
from gesturesign.config import Config
from gesturesign.data import DataError, load_sequence, open_source, read_train_csv
from gesturesign.resample import fit_length, resample
from gesturesign.types import Frames

KEEP_POINTS: tuple[int, ...] = tuple(
    sorted(
        set(range(lm.LEFT_HAND_START, lm.LEFT_HAND_START + lm.N_HAND))
        | set(range(lm.POSE_START, lm.POSE_START + lm.N_POSE))
        | set(range(lm.RIGHT_HAND_START, lm.RIGHT_HAND_START + lm.N_HAND))
        | set(lm.LIPS)
    )
)
COMPACT_HANDS = (
    slice(KEEP_POINTS.index(lm.LEFT_HAND_START), KEEP_POINTS.index(lm.LEFT_HAND_START) + lm.N_HAND),
    slice(
        KEEP_POINTS.index(lm.RIGHT_HAND_START), KEEP_POINTS.index(lm.RIGHT_HAND_START) + lm.N_HAND
    ),
)
CACHE_VERSION = 1
SHARD_SIZE = 2000
MANIFEST = "manifest.json"


def compact(frames: Frames) -> npt.NDArray[np.float16]:
    """[T, 543, 3] -> [T, 115, 2] float16."""
    return frames[:, KEEP_POINTS, :2].astype(np.float16)


def expand(points: npt.NDArray[np.floating]) -> Frames:
    """[T, 115, 2] -> [T, 543, 3] float32; unkept points and z are NaN."""
    out = np.full((points.shape[0], lm.N_POINTS, 3), np.nan, dtype=np.float32)
    out[:, KEEP_POINTS, :2] = points.astype(np.float32)
    return out


def check_participants(table: pd.DataFrame, cfg: Config) -> None:
    present = {int(p) for p in table["participant_id"].unique()}
    wanted = set(cfg.split.test_participants) | set(cfg.split.val_participants)
    missing = sorted(wanted - present)
    if missing:
        raise DataError(
            f"participants {missing} from the config are not in train.csv; "
            f"available: {sorted(present)}. Edit [split] in the config."
        )
    if not present - wanted:
        raise DataError("every participant is held out; nothing is left to train on")


def _shard_name(i: int) -> str:
    return f"shard-{i:04d}.npz"


def _process_shard(
    job: tuple[Path, Path, list[tuple[str, int, int, int]], float, float, int],
) -> int:
    data_dir, out_path, rows, src_fps, fps, keep_frames = job
    source = open_source(data_dir)  # one zip handle per worker process
    points: list[npt.NDArray[np.float16]] = []
    lengths, labels, participants, sequences = [], [], [], []
    for rel, label, participant, sequence in rows:
        frames = load_sequence(source, rel)
        frames = fit_length(resample(frames, src_fps, fps), keep_frames)
        if frames.shape[0] == 0:
            continue
        points.append(compact(frames))
        lengths.append(frames.shape[0])
        labels.append(label)
        participants.append(participant)
        sequences.append(sequence)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        points=np.concatenate(points) if points else np.zeros((0, len(KEEP_POINTS), 2), np.float16),
        lengths=np.asarray(lengths, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int32),
        participants=np.asarray(participants, dtype=np.int64),
        sequence_ids=np.asarray(sequences, dtype=np.int64),
    )
    tmp.replace(out_path)
    return len(lengths)


def prepare(
    cfg: Config,
    data_dir: Path,
    cache_dir: Path,
    *,
    workers: int = 8,
    limit: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    table = read_train_csv(data_dir)
    check_participants(table, cfg)
    labels = sorted(table["sign"].unique())
    index = {s: i for i, s in enumerate(labels)}
    table = table.sort_values("sequence_id").reset_index(drop=True)
    if limit is not None:
        table = table.groupby("participant_id", group_keys=False).head(limit)
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "version": CACHE_VERSION,
        "fps": cfg.fps,
        "source_fps": cfg.source_fps,
        "keep_frames": 2 * cfg.max_frames,
        "clips": len(table),
    }
    manifest_path = cache_dir / MANIFEST
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("settings") != settings:
            log("cache settings changed; rebuilding every shard")
            for shard in cache_dir.glob("shard-*.npz"):
                shard.unlink()

    rows = list(
        zip(
            table["path"],
            table["sign"].map(index),
            table["participant_id"],
            table["sequence_id"],
            strict=True,
        )
    )
    jobs = []
    for i in range(0, len(rows), SHARD_SIZE):
        out_path = cache_dir / _shard_name(i // SHARD_SIZE)
        if out_path.exists():
            continue
        chunk = [
            (str(p), int(lab), int(pid), int(sid)) for p, lab, pid, sid in rows[i : i + SHARD_SIZE]
        ]
        jobs.append((data_dir, out_path, chunk, cfg.source_fps, cfg.fps, 2 * cfg.max_frames))
    n_shards = -(-len(rows) // SHARD_SIZE)
    log(f"{len(rows)} clips in {n_shards} shards; {n_shards - len(jobs)} already cached")
    start = time.monotonic()
    done = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for n in pool.map(_process_shard, jobs):
                done += 1
                elapsed = time.monotonic() - start
                eta = elapsed / done * (len(jobs) - done)
                log(
                    f"  shard {done}/{len(jobs)}: {n} clips, {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

    counts = table.groupby("participant_id").size()
    manifest = {
        "settings": settings,
        "labels": labels,
        "points": list(KEEP_POINTS),
        "shards": [_shard_name(i) for i in range(n_shards)],
        "clips_per_participant": {str(k): int(v) for k, v in counts.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


@dataclass
class LandmarkCache:
    labels: list[str]
    points: npt.NDArray[np.float16]  # all clips concatenated
    offsets: npt.NDArray[np.int64]
    lengths: npt.NDArray[np.int32]
    targets: npt.NDArray[np.int32]
    participants: npt.NDArray[np.int64]
    sequence_ids: npt.NDArray[np.int64]

    @classmethod
    def load(cls, cache_dir: Path) -> LandmarkCache:
        manifest_path = cache_dir / MANIFEST
        if not manifest_path.exists():
            raise DataError(f"no cache at {cache_dir}; run `gesturesign prepare` first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["points"] != list(KEEP_POINTS):
            raise DataError("cache was built with a different point set; re-run prepare")
        parts: dict[str, list[Any]] = {
            k: [] for k in ("points", "lengths", "labels", "participants", "sequence_ids")
        }
        for name in manifest["shards"]:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
                raise DataError(f"cache manifest lists a bad shard name: {name!r}")
            with np.load(cache_dir / name, allow_pickle=False) as shard:
                for key, values in parts.items():
                    values.append(shard[key])
        lengths = np.concatenate(parts["lengths"]).astype(np.int32)
        offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)[:-1]]).astype(np.int64)
        return cls(
            labels=list(manifest["labels"]),
            points=np.concatenate(parts["points"]),
            offsets=offsets,
            lengths=lengths,
            targets=np.concatenate(parts["labels"]).astype(np.int32),
            participants=np.concatenate(parts["participants"]).astype(np.int64),
            sequence_ids=np.concatenate(parts["sequence_ids"]).astype(np.int64),
        )

    def __len__(self) -> int:
        return len(self.lengths)

    def compact_frames(self, i: int) -> npt.NDArray[np.float16]:
        start = int(self.offsets[i])
        return self.points[start : start + int(self.lengths[i])]

    def frames(self, i: int) -> Frames:
        return expand(self.compact_frames(i))

    def split(self, cfg: Config) -> dict[str, npt.NDArray[np.int64]]:
        test = np.isin(self.participants, cfg.split.test_participants)
        val = np.isin(self.participants, cfg.split.val_participants)
        return {
            "train": np.flatnonzero(~test & ~val),
            "val": np.flatnonzero(val),
            "test": np.flatnonzero(test),
        }
