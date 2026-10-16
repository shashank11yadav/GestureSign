"""Run configuration (``configs/*.toml``) and fixed repository paths."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, cast, get_args

from gesturesign.features import FEATURE_SETS, FeatureSet

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_ROOT.parent
CONFIG_DIR = PIPELINE_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
REPORTS_DIR = PIPELINE_ROOT / "reports"
RUNS_DIR = PIPELINE_ROOT / "runs"
WEB_PUBLIC = REPO_ROOT / "web" / "public"
WEB_MODELS_DIR = WEB_PUBLIC / "models"
REPLAY_PATH = WEB_PUBLIC / "replay" / "clips.json"
FIXTURES_DIR = REPO_ROOT / "shared" / "fixtures"


def default_data_dir() -> Path:
    """``$DATA_DIR`` if set, else ``../_datasets/asl-signs`` next to the repo."""
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env).expanduser()
    return REPO_ROOT.parent / "_datasets" / "asl-signs"


@dataclass(frozen=True)
class SplitConfig:
    # Signer-independent split: these participants never appear in training.
    test_participants: tuple[int, ...] = ()
    val_participants: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 192
    depth: int = 3
    heads: int = 4
    conv_kernel: int = 17
    ff_mult: int = 2
    embedding_dim: int = 128
    dropout: float = 0.1
    cosine_scale: float = 16.0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 2.0
    label_smoothing: float = 0.1
    patience: int = 10
    max_steps: int = 0  # 0 = no limit; the smoke run caps this
    grad_clip: float = 1.0
    device: str = "auto"


@dataclass(frozen=True)
class AugmentConfig:
    prob: float = 0.8
    rotate_deg: float = 15.0
    scale: float = 0.15
    aspect: float = 0.15
    shear: float = 0.1
    shift: float = 0.08
    time_stretch: float = 0.25
    frame_drop: float = 0.1
    hand_drop: float = 0.05


@dataclass(frozen=True)
class EvalConfig:
    # Spec 4.1: ship the largest sign set whose held-out top-1 stays >= 60%.
    ship_min_top1: float = 0.60
    # "Not sure" threshold: lowest confidence whose accepted validation
    # predictions are still at least this precise.
    not_sure_precision: float = 0.80
    confusion_pairs: int = 20
    # Teach-your-own episodes, matching the browser: up to 8 gestures, 3 takes each.
    few_shot_ways: int = 8
    few_shot_shots: int = 3
    few_shot_episodes: int = 1000


@dataclass(frozen=True)
class Config:
    name: str = "default"
    seed: int = 42
    # Decided by the A1 benchmark; see pipeline/reports/bench.md.
    feature_set: FeatureSet = "hands_pose_lips"
    fps: float = 30.0
    source_fps: float = 30.0
    min_frames: int = 8
    max_frames: int = 64
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    evaluate: EvalConfig = field(default_factory=EvalConfig)


def config_from_dict(raw: dict[str, Any], where: str = "config") -> Config:
    return cast(Config, _build(Config, raw, where))


def _build(cls: type[Any], raw: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in {where}: {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{where}.{name}")
        elif isinstance(default, tuple):
            kwargs[name] = tuple(value)
        elif isinstance(default, float) and isinstance(value, int):
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    config = config_from_dict(raw, path.name)
    if config.feature_set not in get_args(FeatureSet):
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {config.feature_set!r}")
    if not 1 <= config.min_frames <= config.max_frames:
        raise ValueError("need 1 <= min_frames <= max_frames")
    overlap = set(config.split.test_participants) & set(config.split.val_participants)
    if overlap:
        raise ValueError(f"participants in both test and val: {sorted(overlap)}")
    return config
