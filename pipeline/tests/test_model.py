from __future__ import annotations

import pytest
import torch

from gesturesign.config import ModelConfig
from gesturesign.model import SignModel, count_parameters


def small() -> SignModel:
    torch.manual_seed(0)
    model = SignModel(
        in_dim=20, num_classes=7, max_frames=16, cfg=ModelConfig(dim=32, depth=2, heads=2)
    )
    return model.eval()


def test_outputs_logits_and_unit_embedding() -> None:
    model = small()
    logits, embedding = model(torch.randn(3, 10, 20))
    assert logits.shape == (3, 7)
    assert embedding.shape == (3, 128)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(3))


def test_padding_with_mask_matches_the_unpadded_sequence() -> None:
    model = small()
    x = torch.randn(1, 9, 20)
    alone_logits, alone_emb = model(x)
    padded = torch.zeros(2, 16, 20)
    padded[0, :9] = x[0]
    padded[1] = torch.randn(16, 20)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :9] = True
    mask[1] = True
    batch_logits, batch_emb = model(padded, mask)
    torch.testing.assert_close(batch_logits[0], alone_logits[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch_emb[0], alone_emb[0], atol=1e-5, rtol=1e-5)


def test_rejects_sequences_longer_than_max_frames() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        small()(torch.randn(1, 17, 20))


@pytest.mark.parametrize("in_dim", [299, 460])  # hands_pose, hands_pose_lips
def test_default_size_is_within_budget(in_dim: int) -> None:
    model = SignModel(in_dim=in_dim, num_classes=250, max_frames=64, cfg=ModelConfig())
    n = count_parameters(model)
    assert 1_000_000 <= n <= 3_000_000, n


def test_logits_are_scaled_cosines() -> None:
    model = small()
    logits, _ = model(torch.randn(4, 12, 20))
    assert logits.abs().max() <= model.cfg.cosine_scale + 1e-4
