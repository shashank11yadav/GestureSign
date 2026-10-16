from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from gesturesign import synthetic
from gesturesign.config import Config, SplitConfig
from gesturesign.data import DataError, load_landmarks, read_train_csv
from gesturesign.dataset import SignDataset, collate
from gesturesign.features import sequence_feature_dim
from gesturesign.prepare import LandmarkCache, compact, expand, prepare

PARTICIPANTS = [1, 2, 3, 4, 5, 6]


def cfg() -> Config:
    return Config(split=SplitConfig(test_participants=(5, 6), val_participants=(4,)))


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("asl")
    synthetic.write_dataset(out, participants=PARTICIPANTS, n_signs=4, clips_per_sign=2, seed=1)
    return out


@pytest.fixture(scope="module")
def cache_dir(dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cache")
    prepare(cfg(), dataset_dir, out, workers=1, log=lambda _: None)
    return out


def test_compact_round_trip_keeps_feature_points() -> None:
    frames = synthetic.render_sign(synthetic.sign_templates(1)[0], np.random.default_rng(0), T=6)
    back = expand(compact(frames))
    from gesturesign.prepare import KEEP_POINTS

    np.testing.assert_allclose(back[:, KEEP_POINTS, :2], frames[:, KEEP_POINTS, :2], atol=1e-3)
    assert len(KEEP_POINTS) == 21 + 33 + 21 + 40


def test_cache_holds_every_clip_with_labels(dataset_dir: Path, cache_dir: Path) -> None:
    cache = LandmarkCache.load(cache_dir)
    table = read_train_csv(dataset_dir)
    assert len(cache) == len(table)
    assert cache.labels == sorted(table["sign"].unique())
    first = int(np.argmin(cache.sequence_ids))
    row = table.sort_values("sequence_id").iloc[0]
    original = load_landmarks(dataset_dir / row["path"])
    assert cache.frames(first).shape[0] == original.shape[0]
    assert cache.labels[cache.targets[first]] == row["sign"]


def test_split_is_by_participant(cache_dir: Path) -> None:
    cache = LandmarkCache.load(cache_dir)
    parts = cache.split(cfg())
    assert set(cache.participants[parts["test"]]) == {5, 6}
    assert set(cache.participants[parts["val"]]) == {4}
    assert set(cache.participants[parts["train"]]) == {1, 2, 3}
    assert sum(len(v) for v in parts.values()) == len(cache)


def test_prepare_resumes_without_redoing_shards(dataset_dir: Path, cache_dir: Path) -> None:
    logs: list[str] = []
    prepare(cfg(), dataset_dir, cache_dir, workers=1, log=logs.append)
    assert "1 already cached" in logs[0]


def test_unknown_participant_in_config_fails(dataset_dir: Path, tmp_path: Path) -> None:
    bad = dataclasses.replace(cfg(), split=SplitConfig(test_participants=(99,)))
    with pytest.raises(DataError, match=r"\[99\]"):
        prepare(bad, dataset_dir, tmp_path, workers=1, log=lambda _: None)


def test_dataset_items_and_padding(cache_dir: Path) -> None:
    cache = LandmarkCache.load(cache_dir)
    config = cfg()
    ds = SignDataset(cache, cache.split(config)["train"], config, train=True, seed=0)
    batch = [ds[i] for i in range(4)]
    frames, mask, labels = collate(batch)
    assert frames.shape[0] == 4
    assert frames.shape[2] == sequence_feature_dim(config.feature_set)
    assert frames.shape[1] <= config.max_frames
    assert mask.dtype == torch.bool
    for i, (x, y) in enumerate(batch):
        assert int(mask[i].sum()) == max(1, x.shape[0])
        assert int(labels[i]) == y
    assert torch.isfinite(frames).all()


def test_prepare_reads_straight_from_the_zip(dataset_dir: Path, tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "asl-signs.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(dataset_dir).as_posix())
    prepare(cfg(), archive, tmp_path / "cache", workers=2, limit=3, log=lambda _: None)
    cache = LandmarkCache.load(tmp_path / "cache")
    assert len(cache) == 3 * len(PARTICIPANTS)
