"""``export``: write ``model.onnx``, ``model-meta.json`` and ``labels.json`` for the browser.

ONNX I/O contract (identical for the stub and the trained model)::

    input   frames     float32 [1, T, F]   T in [min_frames, max_frames], F = featureDim
    output  logits     float32 [1, C]      scaled cosine logits, C = classes
    output  embedding  float32 [1, 128]    L2-normalised, used by Teach your own

``export --stub`` writes a randomly initialised model with that contract and
``metrics: null``, so the web app runs end to end before training; the page
then shows an "untrained preview model" notice.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from gesturesign import synthetic
from gesturesign.config import Config
from gesturesign.dataset import model_input
from gesturesign.evaluate import EVAL_FILE, few_shot
from gesturesign.features import FEATURE_VERSION, sequence_feature_dim
from gesturesign.model import SignModel
from gesturesign.signs import display_name, lookup_word
from gesturesign.train import build_model, load_checkpoint, seed_everything

OPSET = 17
INPUT = "frames"
OUTPUTS = ("logits", "embedding")
STUB_SEED = 20261016
# Untrained logits are near-uniform, so with this threshold the stub always says
# "Not sure" instead of naming a sign it cannot know.
STUB_NOT_SURE = 0.5


def write_onnx(model: SignModel, path: Path) -> None:
    model = model.eval().cpu()
    example = torch.zeros(1, model.max_frames // 2, model.in_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=[INPUT],
            output_names=list(OUTPUTS),
            dynamic_axes={INPUT: {1: "T"}},
            opset_version=OPSET,
            dynamo=False,
        )


def verify_onnx(
    model: SignModel, path: Path, lengths: tuple[int, ...], atol: float = 1e-4
) -> float:
    """Run ONNX Runtime against PyTorch at several lengths; return the max abs difference."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    with torch.no_grad():
        for T in lengths:
            x = rng.normal(size=(1, T, model.in_dim)).astype(np.float32)
            ref = [t.numpy() for t in model.eval().cpu()(torch.from_numpy(x))]
            got = session.run(None, {INPUT: x})
            worst = max(worst, *(float(np.abs(a - b).max()) for a, b in zip(ref, got, strict=True)))
    if worst > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:.2e} (> {atol:.0e})")
    return worst


def _model_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _meta(cfg: Config, model: SignModel, version: str, **extra: Any) -> dict[str, Any]:
    return {
        "version": version,
        "featureVersion": FEATURE_VERSION,
        "featureSet": cfg.feature_set,
        "fps": cfg.fps,
        "minFrames": cfg.min_frames,
        "maxFrames": cfg.max_frames,
        "featureDim": sequence_feature_dim(cfg.feature_set),
        "classes": model.num_classes,
        "embeddingDim": cfg.model.embedding_dim,
        "io": {"input": INPUT, "outputs": list(OUTPUTS), "opset": OPSET},
        **extra,
    }


def _labels(signs: list[str], shipped: list[bool], top1: list[float | None]) -> dict[str, Any]:
    return {
        "version": 1,
        "signs": [
            {
                "index": i,
                "sign": sign,
                "label": display_name(sign),
                "lookup": lookup_word(sign),
                "shipped": shipped[i],
                "heldOutTop1": top1[i],
            }
            for i, sign in enumerate(signs)
        ],
    }


def _stub_teach_thresholds(model: SignModel, cfg: Config) -> dict[str, float]:
    """Few-shot thresholds for the untrained model, measured on synthetic clips."""
    rng = np.random.default_rng(STUB_SEED)
    embeddings, labels = [], []
    with torch.no_grad():
        for label, template in enumerate(synthetic.sign_templates(12, seed=STUB_SEED)):
            for _ in range(6):
                frames = synthetic.render_sign(template, rng, left_handed=bool(rng.random() < 0.2))
                x = torch.from_numpy(model_input(frames, cfg))[None]
                embeddings.append(model(x)[1][0].numpy())
                labels.append(label)
    result = few_shot(np.stack(embeddings), np.asarray(labels), rng, episodes=200)
    return {
        "minSimilarity": round(result["min_similarity"], 4),
        "margin": round(result["margin"], 4),
    }


def export_stub(cfg: Config, out_dir: Path, labels: list[str]) -> list[Path]:
    seed_everything(STUB_SEED)
    model = build_model(cfg, len(labels)).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    write_onnx(model, onnx_path)
    verify_onnx(model, onnx_path, (cfg.min_frames, 33, cfg.max_frames))
    meta = _meta(
        cfg,
        model,
        f"stub-{_model_hash(onnx_path)}",
        trainedOn=None,
        metrics=None,
        calibration={"temperature": 1.0, "notSureBelow": STUB_NOT_SURE},
        teach=_stub_teach_thresholds(model, cfg),
    )
    _write_json(out_dir / "model-meta.json", meta)
    _write_json(
        out_dir / "labels.json", _labels(labels, [True] * len(labels), [None] * len(labels))
    )
    return [onnx_path, out_dir / "model-meta.json", out_dir / "labels.json"]


def export_trained(run_dir: Path, out_dir: Path) -> list[Path]:
    eval_path = run_dir / EVAL_FILE
    if not eval_path.exists():
        raise FileNotFoundError(f"{eval_path} not found; run `gesturesign evaluate` first")
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    model, cfg, labels, _ = load_checkpoint(run_dir / "best.pt")
    per_sign = {row["sign"]: row for row in evaluation["perSign"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    write_onnx(model, onnx_path)
    verify_onnx(model, onnx_path, (cfg.min_frames, 33, cfg.max_frames))
    test, shipped, data = evaluation["test"], evaluation["shipped"], evaluation["data"]
    meta = _meta(
        cfg,
        model,
        f"{cfg.name}-{_model_hash(onnx_path)}",
        trainedOn={
            "dataset": "asl-signs",
            "clips": data["clips"],
            "heldOutParticipants": data["testParticipants"],
        },
        metrics={
            "top1": test["top1"],
            "top5": test["top5"],
            "testClips": test["clips"],
            "shippedSigns": shipped["signs"],
            "shippedTop1": shipped["testTop1"],
            "shippedTop3": shipped["testTop3"],
            "teachAccuracy": evaluation["teach"]["accuracy"],
        },
        calibration={
            "temperature": evaluation["calibration"]["temperature"],
            "notSureBelow": evaluation["calibration"]["notSureBelow"],
        },
        teach={
            "minSimilarity": evaluation["teach"]["min_similarity"],
            "margin": evaluation["teach"]["margin"],
        },
    )
    _write_json(out_dir / "model-meta.json", meta)
    _write_json(
        out_dir / "labels.json",
        _labels(
            labels,
            [bool(per_sign[s]["shipped"]) for s in labels],
            [per_sign[s]["top1"] for s in labels],
        ),
    )
    return [onnx_path, out_dir / "model-meta.json", out_dir / "labels.json"]
