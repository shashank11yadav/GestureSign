from __future__ import annotations

import json
from pathlib import Path

import pytest

from gesturesign import replay, synthetic
from gesturesign.config import Config, SplitConfig


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("asl")
    synthetic.write_dataset(out, participants=[1, 2, 3, 4], n_signs=8, clips_per_sign=2, seed=2)
    return out


def cfg() -> Config:
    return Config(split=SplitConfig(test_participants=(3, 4), val_participants=(2,)))


def test_real_clips_come_only_from_held_out_signers_and_shipped_signs(dataset: Path) -> None:
    signs = [t.name for t in synthetic.sign_templates(8, seed=2)][:5]
    clips = replay.asl_signs_clips(cfg(), dataset, signs, n=6)
    assert 1 <= len(clips) <= 5
    assert {c["sign"] for c in clips} <= set(signs)
    assert len({c["sign"] for c in clips}) == len(clips)
    table = __import__("pandas").read_csv(dataset / "train.csv")
    held_out = set(table[table["participant_id"].isin([3, 4])]["sequence_id"])
    assert all(int(c["id"].removeprefix("asl-")) in held_out for c in clips)


def test_real_clips_carry_the_cc_by_credit(dataset: Path, tmp_path: Path) -> None:
    clips = replay.asl_signs_clips(cfg(), dataset, None, n=3)
    path = replay.write_replay(cfg(), clips, "asl-signs", tmp_path / "clips.json")
    payload = json.loads(path.read_text())
    assert payload["source"] == "asl-signs"
    assert payload["credit"] == replay.ASL_CREDIT
    assert "CC BY 4.0" in payload["credit"]


def test_synthetic_clips_have_no_dataset_credit(tmp_path: Path) -> None:
    path = replay.write_replay(
        cfg(), replay.synthetic_clips(cfg()), "synthetic", tmp_path / "c.json"
    )
    payload = json.loads(path.read_text())
    assert payload["credit"] is None
    assert all(c["sign"] is None for c in payload["clips"])


def test_shipped_signs_reads_labels(tmp_path: Path) -> None:
    labels = {"signs": [{"sign": "a", "shipped": True}, {"sign": "b", "shipped": False}]}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    assert replay.shipped_signs(tmp_path / "labels.json") == ["a"]
    assert replay.shipped_signs(tmp_path / "missing.json") is None
