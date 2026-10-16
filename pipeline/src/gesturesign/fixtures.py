"""``export-fixtures``: the feature-parity fixture shared with the web app.

``shared/fixtures/features.json`` holds raw landmark sequences and the features
this package computes for them. ``tests/test_fixtures.py`` and
``web/src/ml/features.test.ts`` both recompute every case and must agree to
within ``tolerance``. Cases cover the train/serve skews that matter: a mirrored
(left-handed or selfie-flipped) copy of a clip, webcam frame rates other than
30 fps, clips longer than the model's window, missing parts, and both feature
sets.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from gesturesign import landmarks as lm
from gesturesign import synthetic
from gesturesign.config import Config
from gesturesign.dataset import model_input
from gesturesign.features import FEATURE_VERSION, FeatureSet, frame_features, mirror_landmarks
from gesturesign.prepare import KEEP_POINTS
from gesturesign.resample import resample
from gesturesign.types import Frames

TOLERANCE = 1e-4
FIXTURE_NAME = "features.json"


def raw_to_model_input(frames: Frames, src_fps: float, cfg: Config) -> npt.NDArray[np.float32]:
    """What the browser runs on a captured segment: resample -> squeeze -> features."""
    return model_input(resample(frames, src_fps, cfg.fps), cfg)


def _quantise(frames: Frames) -> Frames:
    """Round inputs so the JSON numbers convert to exactly these float32 values in JS."""
    return np.round(frames.astype(np.float64), 5).astype(np.float32)


def _encode_frames(frames: Frames) -> list[list[float | None]]:
    """x and y of the kept points per frame (features never read z)."""
    out = []
    for frame in frames[:, KEEP_POINTS, :2]:
        out.append([None if math.isnan(v) else float(v) for v in frame.reshape(-1).tolist()])
    return out


def _round(values: npt.NDArray[np.float32]) -> list[Any]:
    rounded: list[Any] = np.round(values.astype(np.float64), 5).tolist()
    return rounded


def _sequence_case(
    name: str, frames: Frames, src_fps: float, cfg: Config, note: str
) -> dict[str, Any]:
    frames = _quantise(frames)
    features = raw_to_model_input(frames, src_fps, cfg)
    return {
        "name": name,
        "note": note,
        "kind": "sequence",
        "featureSet": cfg.feature_set,
        "srcFps": src_fps,
        "fps": cfg.fps,
        "maxFrames": cfg.max_frames,
        "frames": _encode_frames(frames),
        "expected": {"length": int(features.shape[0]), "features": _round(features)},
    }


def _frame_case(name: str, frame: Frames, feature_set: FeatureSet, note: str) -> dict[str, Any]:
    frame = _quantise(frame[None])[0]
    return {
        "name": name,
        "note": note,
        "kind": "frame",
        "featureSet": feature_set,
        "frames": _encode_frames(frame[None]),
        "expected": {"features": _round(frame_features(frame, feature_set))},
    }


def build_fixture(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(99)
    templates = synthetic.sign_templates(8, seed=99)
    two_handed = next(t for t in templates if t.two_handed)
    one_handed = templates[0]
    base = synthetic.render_sign(two_handed, rng, T=16)
    partial = synthetic.render_sign(one_handed, rng, T=14)
    partial[4:6, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = np.nan
    partial[:, lm.pose(13)] = np.nan
    partial[10, lm.POSE_START : lm.POSE_START + lm.N_POSE] = np.nan
    long_clip = synthetic.render_sign(templates[2], rng, T=72)
    empty = np.full((6, lm.N_POINTS, 3), np.nan, dtype=np.float32)
    lips_cfg = dataclasses.replace(cfg, feature_set="hands_pose_lips")
    only_left = synthetic.render_sign(one_handed, rng, T=12, left_handed=True)[6]

    cases = [
        _sequence_case("two-hands", base, 30, cfg, "two-handed clip at the model rate"),
        _sequence_case(
            "two-hands-mirrored",
            mirror_landmarks(base),
            30,
            cfg,
            "the same clip flipped horizontally; must equal two-hands",
        ),
        _sequence_case("resample-25fps", base, 25, cfg, "25 fps source resampled to 30"),
        _sequence_case("resample-24fps-webcam", partial, 24, cfg, "slow webcam with dropouts"),
        _sequence_case("long-squeezed", long_clip, 30, cfg, "72 frames squeezed to 64"),
        _sequence_case("partial", partial, 30, cfg, "hand dropouts, a lost elbow, no pose once"),
        _sequence_case("empty", empty, 30, cfg, "nothing detected in any frame"),
        _sequence_case("lips", base, 30, lips_cfg, "hands_pose_lips feature set"),
        _frame_case("frame-only-left-hand", only_left, cfg.feature_set, "mirrored per frame"),
        _frame_case("frame-empty", empty[0], cfg.feature_set, "all NaN -> zeros"),
    ]
    return {
        "featureVersion": FEATURE_VERSION,
        "tolerance": TOLERANCE,
        "points": list(KEEP_POINTS),
        "cases": cases,
    }


def write_fixture(cfg: Config, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FIXTURE_NAME
    path.write_text(json.dumps(build_fixture(cfg), separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def decode_frames(points: list[int], frames: list[list[float | None]]) -> Frames:
    out = np.full((len(frames), lm.N_POINTS, 3), np.nan, dtype=np.float32)
    for t, flat in enumerate(frames):
        values = np.array([np.nan if v is None else v for v in flat], dtype=np.float32)
        out[t, points, :2] = values.reshape(len(points), 2)
    return out
