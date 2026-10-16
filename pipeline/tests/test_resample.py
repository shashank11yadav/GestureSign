from __future__ import annotations

import numpy as np
import pytest

from gesturesign.resample import fit_length, resample, resampled_length


def ramp(T: int) -> np.ndarray:
    return np.arange(T, dtype=np.float32).reshape(T, 1, 1).repeat(2, axis=1)


def test_same_rate_is_identity() -> None:
    seq = ramp(10)
    np.testing.assert_array_equal(resample(seq, 30, 30), seq)


def test_halving_the_rate_keeps_every_other_frame() -> None:
    out = resample(ramp(9), 60, 30)
    assert out.shape[0] == 5
    np.testing.assert_allclose(out[:, 0, 0], [0, 2, 4, 6, 8])


def test_upsampling_interpolates_linearly() -> None:
    out = resample(ramp(3), 10, 20)
    np.testing.assert_allclose(out[:, 0, 0], [0, 0.5, 1, 1.5, 2])


def test_length_matches_duration() -> None:
    # 1 second at 25 fps (26 frames incl. both ends) -> 31 frames at 30 fps.
    assert resampled_length(26, 25, 30) == 31
    assert resample(ramp(26), 25, 30).shape[0] == 31


def test_nan_takes_the_nearest_frame() -> None:
    seq = ramp(3)
    seq[1] = np.nan
    out = resample(seq, 10, 40)  # positions 0, .25, .5, .75, 1, 1.25, ...
    vals = out[:, 0, 0]
    assert vals[0] == 0
    assert vals[1] == 0  # nearest is frame 0
    assert np.isnan(vals[2])  # nearest (rounding .5 up) is frame 1
    assert np.isnan(vals[3])
    assert np.isnan(vals[4])
    assert vals[7] == 2  # .75 between NaN frame 1 and frame 2 -> nearest is frame 2


@pytest.mark.parametrize("T", [0, 1])
def test_tiny_sequences_pass_through(T: int) -> None:
    assert resample(ramp(T), 30, 15).shape[0] == T


def test_fit_length_keeps_short_sequences() -> None:
    seq = ramp(20)
    assert fit_length(seq, 64) is seq


def test_fit_length_squeezes_long_sequences_and_keeps_endpoints() -> None:
    out = fit_length(ramp(127), 64)
    assert out.shape[0] == 64
    assert out[0, 0, 0] == 0
    assert out[-1, 0, 0] == pytest.approx(126)
    assert np.all(np.diff(out[:, 0, 0]) > 0)
