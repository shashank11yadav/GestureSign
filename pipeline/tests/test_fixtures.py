"""The committed parity fixture still matches these features; the web tests check the same file."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from gesturesign import config
from gesturesign.config import Config
from gesturesign.features import FEATURE_VERSION, frame_features
from gesturesign.fixtures import FIXTURE_NAME, decode_frames, raw_to_model_input

FIXTURE = config.FIXTURES_DIR / FIXTURE_NAME


def load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


def case_ids() -> list[str]:
    return [c["name"] for c in load()["cases"]]


def test_fixture_matches_the_current_feature_version() -> None:
    assert load()["featureVersion"] == FEATURE_VERSION


@pytest.mark.parametrize("name", case_ids())
def test_case_matches(name: str) -> None:
    fixture = load()
    case = next(c for c in fixture["cases"] if c["name"] == name)
    frames = decode_frames(fixture["points"], case["frames"])
    expected = np.asarray(case["expected"]["features"], dtype=np.float32)
    if case["kind"] == "frame":
        got = frame_features(frames[0], case["featureSet"])
    else:
        cfg = Config(feature_set=case["featureSet"], fps=case["fps"], max_frames=case["maxFrames"])
        got = raw_to_model_input(frames, case["srcFps"], cfg)
        assert got.shape[0] == case["expected"]["length"]
    np.testing.assert_allclose(got, expected, atol=fixture["tolerance"], rtol=0)


def test_mirrored_case_equals_the_original() -> None:
    cases = {c["name"]: c for c in load()["cases"]}
    a = np.asarray(cases["two-hands"]["expected"]["features"])
    b = np.asarray(cases["two-hands-mirrored"]["expected"]["features"])
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_fixture_covers_resampling_and_squeezing() -> None:
    cases = {c["name"]: c for c in load()["cases"]}
    assert cases["resample-25fps"]["srcFps"] == 25
    assert cases["long-squeezed"]["expected"]["length"] == 64
    assert len(cases["long-squeezed"]["frames"]) > 64
