from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gesturesign import data, synthetic
from gesturesign import landmarks as lm


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("asl")
    synthetic.write_dataset(
        out, participants=[1, 2], n_signs=3, clips_per_sign=2, seed=3, left_handed=frozenset({2})
    )
    return out


def test_train_csv_has_kaggle_columns(tiny_dataset: Path) -> None:
    table = data.read_train_csv(tiny_dataset)
    assert tuple(table.columns) == data.TRAIN_CSV_COLUMNS
    assert len(table) == 2 * 3 * 2
    assert table["sequence_id"].is_unique
    assert (tiny_dataset / "sign_to_prediction_index_map.json").exists()


def test_parquet_has_kaggle_schema(tiny_dataset: Path) -> None:
    rel = data.read_train_csv(tiny_dataset)["path"].iloc[0]
    table = pd.read_parquet(tiny_dataset / rel)
    assert tuple(table.columns) == data.PARQUET_COLUMNS
    assert len(table) % lm.N_POINTS == 0
    per_frame = table.pivot_table(index="frame", columns="type", values="x", aggfunc="size")
    assert (per_frame["face"] == 468).all()
    assert (per_frame["left_hand"] == 21).all()
    assert (per_frame["pose"] == 33).all()
    assert (per_frame["right_hand"] == 21).all()
    first = table.iloc[0]
    assert first["row_id"] == f"{first['frame']}-face-0"
    # Hands are NaN before the sign starts, like real detections.
    hands = table[table["type"] == "right_hand"]
    assert hands["x"].isna().any()
    assert hands["x"].notna().any()


def test_load_landmarks_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    template = synthetic.sign_templates(4)[1]
    frames = synthetic.render_sign(template, rng, T=12)
    synthetic.frames_to_parquet_frame(frames, 40).to_parquet(tmp_path / "x.parquet")
    loaded = data.load_landmarks(tmp_path / "x.parquet")
    np.testing.assert_array_equal(loaded, frames)


def test_load_landmarks_places_points_by_type_not_row_order(tmp_path: Path) -> None:
    frames = synthetic.render_sign(synthetic.sign_templates(2)[0], np.random.default_rng(1), T=5)
    table = synthetic.frames_to_parquet_frame(frames, 0).sample(frac=1.0, random_state=0)
    table.to_parquet(tmp_path / "shuffled.parquet")
    np.testing.assert_array_equal(data.load_landmarks(tmp_path / "shuffled.parquet"), frames)


def test_load_landmarks_rejects_unknown_types(tmp_path: Path) -> None:
    frames = synthetic.render_sign(synthetic.sign_templates(1)[0], np.random.default_rng(1), T=3)
    table = synthetic.frames_to_parquet_frame(frames, 0)
    table.loc[0, "type"] = "tail"
    table.to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(data.DataError, match="unknown landmark types"):
        data.load_landmarks(tmp_path / "bad.parquet")


def test_missing_train_csv_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(data.DataError, match="gesturesign synth"):
        data.read_train_csv(tmp_path)


def test_left_handed_participant_signs_with_the_left_hand(tiny_dataset: Path) -> None:
    table = data.read_train_csv(tiny_dataset)
    for pid, expect_left in ((1, False), (2, True)):
        rel = table[table["participant_id"] == pid]["path"].iloc[0]
        frames = data.load_landmarks(tiny_dataset / rel)
        left = np.isfinite(frames[:, lm.LEFT_HAND_START, 0]).sum()
        right = np.isfinite(frames[:, lm.RIGHT_HAND_START, 0]).sum()
        assert (left > right) == expect_left


def test_synthetic_templates_are_distinct() -> None:
    names = [t.name for t in synthetic.sign_templates(24)]
    assert len(set(names)) == 24
