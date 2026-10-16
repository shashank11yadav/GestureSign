from __future__ import annotations

from pathlib import Path

import pytest

from gesturesign import config


def test_default_config_loads() -> None:
    cfg = config.load_config()
    assert cfg.feature_set in ("hands_pose", "hands_pose_lips")
    assert len(cfg.split.test_participants) == 3
    assert len(cfg.split.val_participants) == 2
    assert isinstance(cfg.train.warmup_epochs, float)


def test_every_shipped_config_loads() -> None:
    for path in sorted(config.CONFIG_DIR.glob("*.toml")):
        assert config.load_config(path).name == path.stem


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('name = "bad"\n[train]\nepochz = 3\n')
    with pytest.raises(ValueError, match=r"train\.epochz|epochz"):
        config.load_config(path)


def test_participant_in_two_splits_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[split]\ntest_participants = [1, 2]\nval_participants = [2]\n")
    with pytest.raises(ValueError, match="both test and val"):
        config.load_config(path)


def test_data_dir_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert config.default_data_dir() == tmp_path
    monkeypatch.delenv("DATA_DIR")
    assert config.default_data_dir() == config.REPO_ROOT.parent / "_datasets" / "asl-signs"
