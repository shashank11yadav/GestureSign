from __future__ import annotations

import numpy as np

from gesturesign import landmarks as lm
from gesturesign.augment import augment, random_affine
from gesturesign.config import AugmentConfig
from tests.helpers import full_body_frames


def test_probability_zero_is_identity() -> None:
    frames = full_body_frames(10)
    out = augment(frames, np.random.default_rng(0), AugmentConfig(prob=0.0))
    assert out is frames


def test_same_seed_same_output() -> None:
    frames = full_body_frames(10)
    a = augment(frames, np.random.default_rng(5), AugmentConfig(prob=1.0))
    b = augment(frames, np.random.default_rng(5), AugmentConfig(prob=1.0))
    np.testing.assert_array_equal(a, b)


def test_affine_keeps_missing_points_missing() -> None:
    frames = full_body_frames(6)
    frames[:, lm.LEFT_HAND_START : lm.LEFT_HAND_START + lm.N_HAND] = np.nan
    out = random_affine(frames, np.random.default_rng(1), AugmentConfig())
    assert np.isnan(out[:, lm.LEFT_HAND_START, 0]).all()
    assert np.isfinite(out[:, lm.RIGHT_HAND_START, :2]).all()


def test_affine_preserves_rigid_shape_without_aspect_or_shear() -> None:
    frames = full_body_frames(2)
    cfg = AugmentConfig(scale=0.0, aspect=0.0, shear=0.0, shift=0.1, rotate_deg=30.0)
    out = random_affine(frames, np.random.default_rng(2), cfg)
    a, b = lm.right_hand(0), lm.right_hand(12)
    before = np.linalg.norm(frames[0, a, :2] - frames[0, b, :2])
    after = np.linalg.norm(out[0, a, :2] - out[0, b, :2])
    assert np.isclose(before, after, atol=1e-5)


def test_time_stretch_and_drops_keep_a_usable_sequence() -> None:
    frames = full_body_frames(30)
    cfg = AugmentConfig(prob=1.0, time_stretch=0.5, frame_drop=0.9, hand_drop=0.5)
    for seed in range(20):
        out = augment(frames, np.random.default_rng(seed), cfg)
        assert out.shape[1:] == frames.shape[1:]
        assert out.shape[0] >= 2
        assert out.dtype == np.float32
