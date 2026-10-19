"""``evaluate``: held-out-signer accuracy, the shipped sign set and confidence calibration.

Everything the browser needs to behave honestly is decided here, from data the
model never trained on:

* **Ship set**: greedy backward elimination on the *validation*
  signers. Start with every sign; while top-1 restricted to the remaining signs
  is below ``ship_min_top1``, drop the sign with the lowest per-sign accuracy
  (ties: the one most often predicted wrongly, since removing it helps others).
  The *test* signers then measure the chosen set, so the reported number is not
  inflated by the selection.
* **Temperature and "Not sure" threshold**: a temperature fitted on validation
  logits, then the lowest confidence whose accepted validation predictions are
  at least ``not_sure_precision`` correct.
* **Teach-your-own threshold**: few-shot episodes on test-signer embeddings
  (cosine nearest prototype, as in the browser) give an accuracy and the
  similarity below which a match is reported as "Not sure".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader

from gesturesign.config import Config
from gesturesign.dataset import SignDataset, collate
from gesturesign.model import count_parameters
from gesturesign.prepare import LandmarkCache
from gesturesign.train import load_checkpoint, pick_device, predict

Logits = npt.NDArray[np.float32]
Labels = npt.NDArray[np.int64]
EVAL_FILE = "eval.json"


def topk_accuracy(logits: Logits, labels: npt.ArrayLike, k: int) -> float:
    y = np.asarray(labels)
    if len(y) == 0:
        return float("nan")
    k = min(k, logits.shape[1])
    top = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def _restrict(logits: Logits, shipped: npt.NDArray[np.bool_]) -> Logits:
    out = logits.copy()
    out[:, ~shipped] = -np.inf
    return out


def restricted_accuracy(
    logits: Logits, labels: npt.ArrayLike, shipped: npt.NDArray[np.bool_], k: int = 1
) -> float:
    """Top-k over shipped signs only, on clips whose true sign is shipped."""
    y = np.asarray(labels)
    keep = shipped[y]
    return topk_accuracy(_restrict(logits[keep], shipped), y[keep], k)


def select_ship_set(
    logits: Logits, labels: npt.ArrayLike, min_top1: float
) -> npt.NDArray[np.bool_]:
    y = np.asarray(labels)
    C = logits.shape[1]
    counts = np.bincount(y, minlength=C)
    shipped = counts > 0
    while shipped.sum() > 1:
        keep = shipped[y]
        pred = _restrict(logits[keep], shipped).argmax(1)
        truth = y[keep]
        correct = pred == truth
        if correct.mean() >= min_top1:
            break
        recall = np.bincount(truth, weights=correct, minlength=C) / np.maximum(counts, 1)
        false_pos = np.bincount(pred[~correct], minlength=C)
        candidates = np.flatnonzero(shipped)
        # Lowest recall first; among equals, the sign that steals the most predictions.
        order = np.lexsort((-false_pos[candidates], recall[candidates]))
        shipped[candidates[order[0]]] = False
    return shipped


def _nll(logits: Logits, labels: Labels, temperature: float) -> float:
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: Logits, labels: npt.ArrayLike) -> float:
    y = np.asarray(labels)
    if len(y) == 0:
        return 1.0
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
    losses = [_nll(logits, y, float(t)) for t in grid]
    return float(grid[int(np.argmin(losses))])


def softmax(logits: Logits, temperature: float = 1.0) -> npt.NDArray[np.float64]:
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: npt.NDArray[np.float64] = e / e.sum(axis=1, keepdims=True)
    return out


def not_sure_threshold(
    confidence: npt.ArrayLike, correct: npt.ArrayLike, precision: float
) -> float:
    """Lowest confidence c such that predictions with confidence >= c are >= precision correct."""
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    best = 1.0
    for c in np.unique(conf)[::-1]:
        accepted = conf >= c
        if ok[accepted].mean() >= precision:
            best = float(c)
    return best


def confused_pairs(
    preds: npt.ArrayLike, labels: npt.ArrayLike, names: list[str], n: int
) -> list[dict[str, Any]]:
    p, y = np.asarray(preds), np.asarray(labels)
    wrong = p != y
    counts: dict[tuple[int, int], int] = {}
    for t, q in zip(y[wrong].tolist(), p[wrong].tolist(), strict=True):
        counts[(t, q)] = counts.get((t, q), 0) + 1
    totals = np.bincount(y, minlength=len(names))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [
        {"true": names[t], "predicted": names[q], "count": c, "rate": c / int(totals[t])}
        for (t, q), c in ranked
    ]


def few_shot(
    embeddings: npt.NDArray[np.float32],
    labels: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    ways: int = 8,
    shots: int = 3,
    queries: int = 5,
    episodes: int = 1000,
) -> dict[str, float]:
    """Cosine nearest-prototype episodes, the same rule the browser's Teach mode uses."""
    y = np.asarray(labels)
    classes = [c for c in np.unique(y) if (y == c).sum() > shots]
    if len(classes) < 2:
        return {"accuracy": float("nan"), "min_similarity": 0.5, "margin": 0.04, "episodes": 0}
    ways = min(ways, len(classes))
    hits, total, own_sims, gaps = 0, 0, [], []
    for _ in range(episodes):
        chosen = rng.choice(classes, size=ways, replace=False)
        protos, query_sets = [], []
        for c in chosen:
            idx = rng.permutation(np.flatnonzero(y == c))
            support, query = idx[:shots], idx[shots : shots + queries]
            proto = embeddings[support].mean(axis=0)
            protos.append(proto / max(np.linalg.norm(proto), 1e-12))
            query_sets.append(query)
        P = np.stack(protos)
        for way, query in enumerate(query_sets):
            sims = embeddings[query] @ P.T
            correct = sims.argmax(axis=1) == way
            hits += int(correct.sum())
            total += len(query)
            own_sims.extend(sims[:, way].tolist())
            others = np.delete(sims, way, axis=1).max(axis=1)
            gaps.extend((sims[:, way] - others)[correct].tolist())
    # Accept a match only above the similarity 95% of true matches reach, and
    # only if it beats the runner-up by the gap 90% of correct matches have.
    margin = float(np.clip(np.percentile(gaps, 10), 0.002, 0.1)) if gaps else 0.04
    return {
        "accuracy": hits / max(total, 1),
        "min_similarity": float(np.percentile(own_sims, 5)),
        "margin": margin,
        "episodes": episodes,
        "ways": ways,
        "shots": shots,
    }


def evaluate(
    cfg: Config,
    cache_dir: Path,
    run_dir: Path,
    *,
    device_name: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    model, train_cfg, labels, state = load_checkpoint(run_dir / "best.pt")
    if train_cfg.feature_set != cfg.feature_set:
        raise ValueError("the checkpoint was trained with a different feature set than the config")
    device = pick_device(device_name or cfg.train.device)
    model = model.to(device)
    cache = LandmarkCache.load(cache_dir)
    split = cache.split(train_cfg)
    outputs = {}
    for name in ("val", "test"):
        loader = DataLoader(
            SignDataset(cache, split[name], train_cfg, train=False),
            batch_size=256,
            collate_fn=collate,
        )
        outputs[name] = predict(model, loader, device)
        log(f"{name}: {len(outputs[name][2])} clips")
    val_logits, _, val_y = outputs["val"]
    test_logits, test_emb, test_y = outputs["test"]
    ev = cfg.evaluate
    C = len(labels)

    shipped = select_ship_set(val_logits, val_y, ev.ship_min_top1)
    temperature = fit_temperature(
        _restrict(val_logits, shipped)[shipped[val_y]], val_y[shipped[val_y]]
    )
    val_keep = shipped[val_y]
    val_probs = softmax(_restrict(val_logits[val_keep], shipped), temperature)
    val_correct = val_probs.argmax(1) == val_y[val_keep]
    threshold = not_sure_threshold(val_probs.max(1), val_correct, ev.not_sure_precision)

    test_keep = shipped[test_y]
    test_probs = softmax(_restrict(test_logits[test_keep], shipped), temperature)
    test_conf = test_probs.max(1)
    test_correct = test_probs.argmax(1) == test_y[test_keep]
    accepted = test_conf >= threshold

    test_pred = test_logits.argmax(1)
    per_sign_correct = np.bincount(test_y, weights=test_pred == test_y, minlength=C)
    per_sign_n = np.bincount(test_y, minlength=C)
    rng = np.random.default_rng(cfg.seed)
    teach = few_shot(test_emb, test_y, rng, ways=ev.few_shot_ways, shots=ev.few_shot_shots,
                     episodes=ev.few_shot_episodes)  # fmt: skip

    metrics: dict[str, Any] = {
        "model": {
            "parameters": count_parameters(model),
            "featureSet": train_cfg.feature_set,
            "epoch": state["epoch"],
            "valTop1AtCheckpoint": state["val_top1"],
        },
        "data": {
            "signs": C,
            "testParticipants": list(train_cfg.split.test_participants),
            "valParticipants": list(train_cfg.split.val_participants),
            "clips": {k: len(v) for k, v in split.items()},
        },
        "test": {
            "top1": topk_accuracy(test_logits, test_y, 1),
            "top5": topk_accuracy(test_logits, test_y, 5),
            "clips": len(test_y),
        },
        "val": {
            "top1": topk_accuracy(val_logits, val_y, 1),
            "top5": topk_accuracy(val_logits, val_y, 5),
            "clips": len(val_y),
        },
        "shipped": {
            "minTop1": ev.ship_min_top1,
            "selectedOn": "val",
            "signs": int(shipped.sum()),
            "testTop1": restricted_accuracy(test_logits, test_y, shipped, 1),
            "testTop3": restricted_accuracy(test_logits, test_y, shipped, 3),
            "testClips": int(test_keep.sum()),
            "meetsBar": bool(
                restricted_accuracy(test_logits, test_y, shipped, 1) >= ev.ship_min_top1
            ),
        },
        "calibration": {
            "temperature": temperature,
            "notSureBelow": threshold,
            "targetPrecision": ev.not_sure_precision,
            "testAnswered": float(accepted.mean()) if len(accepted) else float("nan"),
            "testPrecisionWhenAnswered": float(test_correct[accepted].mean())
            if accepted.any()
            else float("nan"),
        },
        "teach": teach,
        "perSign": [
            {
                "sign": labels[c],
                "clips": int(per_sign_n[c]),
                "top1": float(per_sign_correct[c] / per_sign_n[c]) if per_sign_n[c] else None,
                "shipped": bool(shipped[c]),
            }
            for c in range(C)
        ],
        "confusedPairs": confused_pairs(test_pred, test_y, labels, ev.confusion_pairs),
    }
    (run_dir / EVAL_FILE).write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics


def metrics_markdown(metrics: dict[str, Any]) -> str:
    """README tables, built from metrics.json."""
    t, s, cal, teach = metrics["test"], metrics["shipped"], metrics["calibration"], metrics["teach"]
    d = metrics["data"]
    lines = [
        f"Held-out signers: {', '.join(map(str, d['testParticipants']))} "
        f"({t['clips']} clips, never seen in training or model selection).",
        "",
        "| Sign set | Signs | Top-1 | Top-5 / Top-3 |",
        "|---|---:|---:|---:|",
        f"| All signs | {d['signs']} | {t['top1']:.1%} | {t['top5']:.1%} (top-5) |",
        f"| Shipped in the demo | {s['signs']} | {s['testTop1']:.1%} "
        f"| {s['testTop3']:.1%} (top-3) |",
        "",
        f"The shipped set is the largest set found whose top-1 on the validation signers is at "
        f"least {s['minTop1']:.0%}; the test signers above did not take part in choosing it.",
        "",
        f'"Not sure" below {cal["notSureBelow"]:.2f} calibrated confidence: the demo answers '
        f"{cal['testAnswered']:.0%} of test clips, and those answers are "
        f"{cal['testPrecisionWhenAnswered']:.1%} correct.",
        "",
        f"Teach your own ({teach.get('ways', 0)}-way, {teach.get('shots', 0)}-shot, cosine "
        f"nearest prototype on test-signer embeddings): {teach['accuracy']:.1%}.",
        "",
        "| Most confused (true -> predicted) | Count | Share of true sign |",
        "|---|---:|---:|",
    ]
    for pair in metrics["confusedPairs"][:10]:
        lines.append(
            f"| {pair['true']} -> {pair['predicted']} | {pair['count']} | {pair['rate']:.0%} |"
        )
    return "\n".join(lines) + "\n"
