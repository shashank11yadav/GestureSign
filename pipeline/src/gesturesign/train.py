"""``train``: AdamW with warmup + cosine decay, label smoothing, augmentation, early stopping."""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gesturesign.config import Config, config_from_dict
from gesturesign.dataset import SignDataset, collate
from gesturesign.features import sequence_feature_dim
from gesturesign.model import SignModel, count_parameters
from gesturesign.prepare import LandmarkCache

CHECKPOINT = "best.pt"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - libraries that still use the global RNG
    torch.manual_seed(seed)


def build_model(cfg: Config, num_classes: int) -> SignModel:
    return SignModel(
        in_dim=sequence_feature_dim(cfg.feature_set),
        num_classes=num_classes,
        max_frames=cfg.max_frames,
        cfg=cfg.model,
    )


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        # Norms, biases and the positional table are not weight-decayed.
        if param.ndim < 2 or name == "pos":
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_factor(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict(
    model: SignModel, loader: DataLoader[Any], device: torch.device
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """(logits [N, C], embeddings [N, E], labels [N]) over a loader, in order."""
    model.eval()
    logits, embeddings, labels = [], [], []
    for frames, mask, y in loader:
        out, emb = model(frames.to(device), mask.to(device))
        logits.append(out.float().cpu().numpy())
        embeddings.append(emb.float().cpu().numpy())
        labels.append(y.numpy())
    if not logits:
        return (
            np.zeros((0, model.num_classes), np.float32),
            np.zeros((0, model.cfg.embedding_dim), np.float32),
            np.zeros(0, np.int64),
        )
    return np.concatenate(logits), np.concatenate(embeddings), np.concatenate(labels)


@dataclass
class TrainResult:
    checkpoint: Path
    best_val_top1: float
    epochs: int
    steps: int
    seconds: float


def load_checkpoint(path: Path) -> tuple[SignModel, Config, list[str], dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(state["config"], path.name)
    model = build_model(cfg, len(state["labels"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg, list(state["labels"]), state


def train(
    cfg: Config,
    cache_dir: Path,
    run_dir: Path,
    *,
    device_name: str | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    seed_everything(cfg.seed)
    device = pick_device(device_name or cfg.train.device)
    cache = LandmarkCache.load(cache_dir)
    split = cache.split(cfg)
    if len(split["train"]) == 0 or len(split["val"]) == 0:
        raise ValueError("training and validation splits must both be non-empty")
    num_classes = len(cache.labels)
    model = build_model(cfg, num_classes).to(device)
    log(
        f"device {device}; {count_parameters(model):,} parameters; "
        f"{len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test clips"
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds = SignDataset(cache, split["train"], cfg, train=True, seed=cfg.seed)
    val_ds = SignDataset(cache, split["val"], cfg, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate,
        generator=generator, drop_last=len(train_ds) > cfg.train.batch_size,
    )  # fmt: skip
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate)

    steps_per_epoch = max(1, len(train_loader))
    total = steps_per_epoch * cfg.train.epochs
    if cfg.train.max_steps:
        total = min(total, cfg.train.max_steps)
    warmup = int(cfg.train.warmup_epochs * steps_per_epoch)
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.train.weight_decay), lr=cfg.train.lr, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total))

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / CHECKPOINT
    history: list[dict[str, float]] = []
    best, stale, step, epoch = -1.0, 0, 0, 0
    start = time.monotonic()
    while epoch < cfg.train.epochs and step < total:
        epoch += 1
        model.train()
        running, seen = 0.0, 0
        for frames, mask, labels in train_loader:
            frames, mask, labels = frames.to(device), mask.to(device), labels.to(device)
            logits, _ = model(frames, mask)
            loss = F.cross_entropy(logits, labels, label_smoothing=cfg.train.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(labels)
            seen += len(labels)
            step += 1
            if step >= total:
                break
        val_logits, _, val_labels = predict(model, val_loader, device)
        val_top1 = float((val_logits.argmax(1) == val_labels).mean()) if len(val_labels) else 0.0
        elapsed = time.monotonic() - start
        history.append(
            {"epoch": epoch, "step": step, "loss": running / max(1, seen), "val_top1": val_top1}
        )
        eta = elapsed / epoch * (cfg.train.epochs - epoch)
        log(
            f"epoch {epoch}/{cfg.train.epochs}  loss {running / max(1, seen):.3f}  "
            f"val top-1 {val_top1:.3f}  ({elapsed:.0f}s, at most {eta:.0f}s left)"
        )
        if val_top1 > best:
            best, stale = val_top1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": dataclasses.asdict(cfg),
                    "labels": cache.labels,
                    "epoch": epoch,
                    "step": step,
                    "val_top1": val_top1,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= cfg.train.patience:
                log(f"no improvement for {stale} epochs; stopping")
                break
    seconds = time.monotonic() - start
    (run_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    return TrainResult(checkpoint, best, epoch, step, seconds)
