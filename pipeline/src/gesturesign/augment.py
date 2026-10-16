"""Training-time augmentation, applied to landmarks before features are computed.

Augmenting coordinates rather than features keeps every derived block
(hand shape, velocity, presence) consistent with the augmented geometry.
"""

from __future__ import annotations

import math

import numpy as np

from gesturesign import landmarks as lm
from gesturesign.config import AugmentConfig
from gesturesign.features import Anchor, compute_anchor
from gesturesign.resample import resample
from gesturesign.types import Frames

FULL_LAYOUT_HANDS = (
    slice(lm.LEFT_HAND_START, lm.LEFT_HAND_START + lm.N_HAND),
    slice(lm.RIGHT_HAND_START, lm.RIGHT_HAND_START + lm.N_HAND),
)


def random_affine(
    frames: Frames, rng: np.random.Generator, cfg: AugmentConfig, anchor: Anchor | None = None
) -> Frames:
    """Rotate, scale, stretch, shear and shift x/y about the shoulder centre.

    Separate x and y scaling also covers the unknown aspect ratio of the
    training videos versus a landscape webcam. ``frames`` is [T, P, C>=2]; pass
    ``anchor`` when the points are not in the full 543-point layout.
    """
    if anchor is None:
        anchor = compute_anchor(frames)
    theta = math.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    scale = 1.0 + rng.uniform(-cfg.scale, cfg.scale)
    aspect = 1.0 + rng.uniform(-cfg.aspect, cfg.aspect)
    shear = rng.uniform(-cfg.shear, cfg.shear)
    shift = rng.uniform(-cfg.shift, cfg.shift, size=2) * anchor.scale * 4
    c, s = math.cos(theta), math.sin(theta)
    matrix = (
        np.array([[c, -s], [s, c]])
        @ np.array([[1.0, shear], [0.0, 1.0]])
        @ np.diag([scale * aspect, scale / aspect])
    )
    centre = np.array([anchor.cx, anchor.cy])
    out = frames.copy()
    xy = frames[..., :2].astype(np.float64) - centre
    out[..., :2] = (xy @ matrix.T + centre + shift).astype(np.float32)
    return out


def augment(
    frames: Frames,
    rng: np.random.Generator,
    cfg: AugmentConfig,
    *,
    anchor: Anchor | None = None,
    hands: tuple[slice, slice] = FULL_LAYOUT_HANDS,
) -> Frames:
    """Randomly apply affine, time stretch, frame drop and hand drop (with probability ``prob``).

    Works on any [T, P, C] point layout; for anything but the full 543-point
    layout pass the sequence ``anchor`` and where the two ``hands`` sit.
    """
    if cfg.prob <= 0 or rng.random() >= cfg.prob:
        return frames
    out = random_affine(frames, rng, cfg, anchor)
    if cfg.time_stretch > 0 and out.shape[0] > 2:
        factor = rng.uniform(1 - cfg.time_stretch, 1 + cfg.time_stretch)
        out = resample(out, 1.0, factor)
    if cfg.frame_drop > 0 and out.shape[0] > 4:
        keep = rng.random(out.shape[0]) >= cfg.frame_drop
        if np.count_nonzero(keep) >= 2:
            out = out[keep]
    if cfg.hand_drop > 0:
        out = out.copy()
        for hand in hands:
            drop = rng.random(out.shape[0]) < cfg.hand_drop
            out[drop, hand] = np.nan
    return out.astype(np.float32, copy=False)
