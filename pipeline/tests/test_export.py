from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from gesturesign import export
from gesturesign.config import Config, ModelConfig
from gesturesign.features import FEATURE_VERSION, sequence_feature_dim
from gesturesign.signs import ASL_SIGNS
from gesturesign.train import build_model


def small_cfg() -> Config:
    return dataclasses.replace(Config(), model=ModelConfig(dim=32, depth=1, heads=2))


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("stub")
    export.export_stub(Config(), out, labels=list(ASL_SIGNS))
    return out


def test_stub_has_the_onnx_io_contract(stub_dir: Path) -> None:
    session = ort.InferenceSession(str(stub_dir / "model.onnx"))
    (inp,) = session.get_inputs()
    assert inp.name == "frames"
    assert inp.shape == [1, "T", sequence_feature_dim("hands_pose")]
    assert [o.name for o in session.get_outputs()] == ["logits", "embedding"]
    for T in (8, 64):
        x = np.random.default_rng(T).normal(size=(1, T, 299)).astype(np.float32)
        logits, emb = session.run(None, {"frames": x})
        assert logits.shape == (1, 250)
        assert emb.shape == (1, 128)
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)


def test_stub_files_are_small_enough_for_pages(stub_dir: Path) -> None:
    assert (stub_dir / "model.onnx").stat().st_size < 10 * 1024 * 1024


def test_stub_meta_marks_the_model_untrained(stub_dir: Path) -> None:
    meta = json.loads((stub_dir / "model-meta.json").read_text())
    for key in (
        "version",
        "featureVersion",
        "fps",
        "featureDim",
        "classes",
        "trainedOn",
        "metrics",
    ):
        assert key in meta
    assert meta["metrics"] is None
    assert meta["trainedOn"] is None
    assert meta["featureVersion"] == FEATURE_VERSION
    assert meta["featureDim"] == 299
    assert meta["classes"] == 250
    assert meta["featureSet"] == "hands_pose"
    assert 0 < meta["calibration"]["notSureBelow"] <= 1
    assert -1 <= meta["teach"]["minSimilarity"] <= 1


def test_stub_labels_ship_every_sign(stub_dir: Path) -> None:
    labels = json.loads((stub_dir / "labels.json").read_text())["signs"]
    assert len(labels) == 250
    assert all(s["shipped"] for s in labels)
    thank = next(s for s in labels if s["sign"] == "thankyou")
    assert thank["label"] == "thank you"
    assert [s["index"] for s in labels] == list(range(250))


def test_stub_export_is_reproducible(stub_dir: Path, tmp_path: Path) -> None:
    export.export_stub(Config(), tmp_path, labels=list(ASL_SIGNS))
    assert (tmp_path / "model.onnx").read_bytes() == (stub_dir / "model.onnx").read_bytes()
    assert (tmp_path / "model-meta.json").read_text() == (stub_dir / "model-meta.json").read_text()


def test_trained_export_carries_metrics_and_ship_flags(tmp_path: Path) -> None:
    cfg = small_cfg()
    labels = ["a", "b", "c"]
    model = build_model(cfg, len(labels))
    run = tmp_path / "run"
    run.mkdir()
    torch.save(
        {
            "model": model.state_dict(),
            "config": dataclasses.asdict(cfg),
            "labels": labels,
            "epoch": 3,
            "step": 30,
            "val_top1": 0.5,
        },
        run / "best.pt",
    )
    evaluation = {
        "test": {"top1": 0.7, "top5": 0.9, "clips": 10},
        "shipped": {"signs": 2, "testTop1": 0.8, "testTop3": 0.95},
        "calibration": {"temperature": 0.7, "notSureBelow": 0.42},
        "teach": {"accuracy": 0.9, "min_similarity": 0.61, "margin": 0.04},
        "data": {"testParticipants": [5, 6, 7], "clips": {"train": 100, "val": 20, "test": 10}},
        "perSign": [
            {"sign": "a", "top1": 0.9, "shipped": True},
            {"sign": "b", "top1": 0.2, "shipped": False},
            {"sign": "c", "top1": 0.8, "shipped": True},
        ],
    }
    (run / "eval.json").write_text(json.dumps(evaluation))
    out = tmp_path / "models"
    export.export_trained(run, out)
    meta = json.loads((out / "model-meta.json").read_text())
    assert meta["metrics"]["top1"] == 0.7
    assert meta["metrics"]["shippedSigns"] == 2
    assert meta["calibration"] == {"temperature": 0.7, "notSureBelow": 0.42}
    assert meta["teach"]["minSimilarity"] == 0.61
    assert meta["trainedOn"]["dataset"] == "asl-signs"
    signs = json.loads((out / "labels.json").read_text())["signs"]
    assert [s["shipped"] for s in signs] == [True, False, True]
    assert signs[1]["heldOutTop1"] == 0.2
    session = ort.InferenceSession(str(out / "model.onnx"))
    x = np.zeros((1, 9, sequence_feature_dim(cfg.feature_set)), np.float32)
    assert session.run(None, {"frames": x})[0].shape == (1, 3)


def test_export_without_evaluation_refuses(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluate"):
        export.export_trained(tmp_path, tmp_path / "out")
