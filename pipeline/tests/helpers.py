"""Hand-built landmark frames with known geometry, shared by the tests."""

from __future__ import annotations

import numpy as np

from gesturesign import landmarks as lm
from gesturesign.types import Frames


def blank_frames(T: int = 1) -> Frames:
    return np.full((T, lm.N_POINTS, 3), np.nan, dtype=np.float32)


def put_shoulders(frames: Frames, left: tuple[float, float], right: tuple[float, float]) -> None:
    frames[:, lm.pose(lm.POSE_LEFT_SHOULDER), :2] = left
    frames[:, lm.pose(lm.POSE_RIGHT_SHOULDER), :2] = right
    frames[:, lm.pose(lm.POSE_LEFT_SHOULDER), 2] = 0.0
    frames[:, lm.pose(lm.POSE_RIGHT_SHOULDER), 2] = 0.0


def hand_shape(wrist: tuple[float, float], size: float, rng: np.random.Generator) -> np.ndarray:
    """21 points: wrist plus a fan of fingers, with some jitter so points differ."""
    pts = np.zeros((lm.N_HAND, 3), dtype=np.float32)
    pts[0, :2] = wrist
    for finger in range(5):
        angle = -0.9 + finger * 0.45
        for joint in range(4):
            r = size * (0.5 + 0.35 * joint)
            idx = 1 + finger * 4 + joint
            pts[idx, 0] = wrist[0] + r * np.sin(angle) + rng.normal(0, size * 0.03)
            pts[idx, 1] = wrist[1] - r * np.cos(angle) + rng.normal(0, size * 0.03)
            pts[idx, 2] = rng.normal(0, 0.01)
    return pts


def full_body_frames(T: int, seed: int = 0) -> Frames:
    """A plausible upper body with both hands, face and lips, moving over T frames."""
    rng = np.random.default_rng(seed)
    frames = rng.uniform(0.3, 0.7, size=(T, lm.N_POINTS, 3)).astype(np.float32)
    put_shoulders(frames, (0.62, 0.62), (0.38, 0.62))
    for t in range(T):
        frames[t, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = hand_shape(
            (0.30 + 0.01 * t, 0.45 - 0.005 * t), 0.06, rng
        )
        frames[t, lm.LEFT_HAND_START : lm.LEFT_HAND_START + lm.N_HAND] = hand_shape(
            (0.70 - 0.004 * t, 0.55), 0.05, rng
        )
    return frames
