"""Files the pipeline reads but did not necessarily write: checkpoints, cache shards, datasets.

A checkpoint or cache from somewhere else must not be able to run code, and
a dataset listing must not reach files outside the dataset.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from gesturesign import data
from gesturesign.config import Config
from gesturesign.prepare import KEEP_POINTS, MANIFEST, LandmarkCache
from gesturesign.train import build_model, load_checkpoint

RAN: list[str] = []


def _record(what: str) -> None:
    RAN.append(what)


class Payload:
    """Unpickling this calls _record(), standing in for arbitrary code."""

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_record, ("payload ran",))


def _save_checkpoint(path: Path, **extra: object) -> None:
    cfg = Config()
    model = build_model(cfg, 3)
    torch.save(
        {
            "model": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "labels": ["a", "b", "c"],
            "epoch": 1,
            "step": 10,
            "val_top1": 0.5,
            **extra,
        },
        path,
    )


def test_checkpoint_written_by_train_loads(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    _save_checkpoint(path)
    model, _, labels, state = load_checkpoint(path)
    assert labels == ["a", "b", "c"]
    assert state["epoch"] == 1
    assert model.num_classes == 3


def test_checkpoint_cannot_run_code(tmp_path: Path) -> None:
    path = tmp_path / "best.pt"
    _save_checkpoint(path, extra=Payload())
    RAN.clear()
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert RAN == []


def _write_cache(cache_dir: Path, shard_name: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"points": list(KEEP_POINTS), "labels": ["a"], "shards": [shard_name]}
    (cache_dir / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def test_cache_shard_names_stay_inside_the_cache(tmp_path: Path) -> None:
    np.savez(tmp_path / "outside.npz", points=np.zeros(1))
    _write_cache(tmp_path / "cache", "../outside.npz")
    with pytest.raises(data.DataError, match="shard"):
        LandmarkCache.load(tmp_path / "cache")


def test_cache_shards_cannot_hold_pickled_objects(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _write_cache(cache, "shard-000.npz")
    objects = np.array([Payload()], dtype=object)
    np.savez(
        cache / "shard-000.npz",
        points=objects,
        lengths=objects,
        labels=objects,
        participants=objects,
        sequence_ids=objects,
    )
    RAN.clear()
    with pytest.raises(ValueError, match="pickle"):
        LandmarkCache.load(cache)
    assert RAN == []


@pytest.mark.parametrize("rel", ["../secret.parquet", "/etc/passwd", "train/../../secret.parquet"])
def test_dataset_paths_stay_inside_the_dataset(tmp_path: Path, rel: str) -> None:
    root = tmp_path / "asl"
    root.mkdir()
    (tmp_path / "secret.parquet").write_bytes(b"not yours")
    source = data.FolderSource(root)
    with pytest.raises(data.DataError, match="outside"), source.open(rel):
        pass
    assert not source.exists(rel)
