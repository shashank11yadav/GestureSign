from __future__ import annotations

import numpy as np
import pytest

from gesturesign import evaluate as ev


def one_hot_logits(preds: list[int], C: int, strength: float = 5.0) -> np.ndarray:
    logits = np.zeros((len(preds), C), dtype=np.float32)
    logits[np.arange(len(preds)), preds] = strength
    return logits


def test_topk_accuracy() -> None:
    logits = np.array([[3, 2, 1], [1, 3, 2], [3, 1, 2]], dtype=np.float32)
    labels = np.array([0, 2, 1])
    assert ev.topk_accuracy(logits, labels, 1) == pytest.approx(1 / 3)
    assert ev.topk_accuracy(logits, labels, 2) == pytest.approx(2 / 3)
    assert ev.topk_accuracy(logits, labels, 3) == pytest.approx(1.0)


def test_topk_accuracy_of_nothing_is_nan() -> None:
    assert np.isnan(ev.topk_accuracy(np.zeros((0, 3), np.float32), np.zeros(0, np.int64), 1))


def test_ship_set_keeps_everything_when_accurate_enough() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    logits = one_hot_logits([0, 0, 1, 1, 2, 1], 3)
    shipped = ev.select_ship_set(logits, labels, 0.6)
    assert shipped.tolist() == [True, True, True]


def test_ship_set_drops_the_worst_signs_until_the_bar_is_met() -> None:
    # Sign 2 is always predicted as 1, sign 3 as 0; signs 0 and 1 are perfect.
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    logits = one_hot_logits([0, 0, 1, 1, 1, 1, 0, 0], 4)
    shipped = ev.select_ship_set(logits, labels, 0.9)
    assert shipped.tolist() == [True, True, False, False]
    acc = ev.restricted_accuracy(logits, labels, shipped)
    assert acc == pytest.approx(1.0)


def test_dropping_a_sign_can_rescue_signs_confused_with_it() -> None:
    # Sign 1 is often predicted as sign 2 (second choice is correct). Removing 2
    # from the candidates fixes sign 1 entirely.
    labels = np.array([1, 1, 1, 2, 2, 0, 0])
    logits = np.zeros((7, 3), np.float32)
    logits[0:3, 2] = 5
    logits[0:3, 1] = 4
    logits[3:5, 0] = 5  # sign 2 is wrong as well
    logits[5:7, 0] = 5
    shipped = ev.select_ship_set(logits, labels, 0.9)
    assert shipped.tolist() == [True, True, False]


def test_signs_without_examples_are_not_shipped() -> None:
    labels = np.array([0, 0, 1])
    logits = one_hot_logits([0, 0, 1], 3)
    assert ev.select_ship_set(logits, labels, 0.5).tolist() == [True, True, False]


def test_temperature_softens_overconfident_logits() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 400)
    logits = rng.normal(0, 1, (400, 5)).astype(np.float32)
    logits[np.arange(400), labels] += 1.0
    logits *= 8  # far too confident for ~55% accuracy
    assert ev.fit_temperature(logits, labels) > 2.0


def test_not_sure_threshold_meets_the_precision_target() -> None:
    conf = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([True, True, True, False, True, False, False])
    threshold = ev.not_sure_threshold(conf, correct, 0.8)
    accepted = conf >= threshold
    assert correct[accepted].mean() >= 0.8
    assert threshold == pytest.approx(0.6)  # 4 of the top 5 are right


def test_not_sure_threshold_is_one_when_target_is_unreachable() -> None:
    conf = np.array([0.9, 0.8])
    correct = np.array([False, False])
    assert ev.not_sure_threshold(conf, correct, 0.8) == 1.0


def test_confused_pairs_are_ranked_by_count() -> None:
    labels = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([1, 1, 0, 0, 2, 2])
    pairs = ev.confused_pairs(preds, labels, ["a", "b", "c"], 5)
    assert pairs[0] == {"true": "a", "predicted": "b", "count": 2, "rate": pytest.approx(2 / 3)}
    assert len(pairs) == 3


def test_few_shot_on_separable_embeddings_is_perfect() -> None:
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (10, 16))
    labels = np.repeat(np.arange(10), 8)
    emb = centres[labels] + rng.normal(0, 0.05, (80, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    result = ev.few_shot(emb.astype(np.float32), labels, rng, ways=5, shots=3, episodes=50)
    assert result["accuracy"] == pytest.approx(1.0)
    assert 0.0 < result["min_similarity"] <= 1.0
    assert 0.002 <= result["margin"] <= 0.1
