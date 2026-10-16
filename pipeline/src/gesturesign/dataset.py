"""Torch dataset over the landmark cache: augment -> squeeze to max_frames -> features."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from gesturesign.augment import augment
from gesturesign.config import Config
from gesturesign.features import compute_anchor, sequence_features
from gesturesign.prepare import COMPACT_HANDS, LandmarkCache, expand
from gesturesign.resample import fit_length
from gesturesign.types import Frames


def model_input(frames: Frames, cfg: Config) -> npt.NDArray[np.float32]:
    """Already-resampled landmarks -> model features [T, F2], as the browser computes them."""
    return sequence_features(fit_length(frames, cfg.max_frames), cfg.feature_set)


class SignDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        cache: LandmarkCache,
        indices: npt.NDArray[np.int64],
        cfg: Config,
        *,
        train: bool,
        seed: int = 0,
    ) -> None:
        self.cache = cache
        self.indices = indices
        self.cfg = cfg
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        points = self.cache.compact_frames(idx).astype(np.float32)
        if self.train:
            # Augment the 115 cached points, not the mostly-empty 543-point layout.
            anchor = compute_anchor(expand(points))
            points = augment(points, self.rng, self.cfg.augment, anchor=anchor, hands=COMPACT_HANDS)
        features = model_input(expand(points), self.cfg)
        return torch.from_numpy(features), int(self.cache.targets[idx])


def collate(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to the longest clip in the batch -> (frames [B, T, F], mask [B, T], labels [B])."""
    T = max(1, max(x.shape[0] for x, _ in batch))
    F = batch[0][0].shape[1]
    frames = torch.zeros(len(batch), T, F)
    mask = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, (x, _) in enumerate(batch):
        frames[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    mask[:, 0] = True  # an empty clip still pools over one (all-zero) frame
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return frames, mask, labels
