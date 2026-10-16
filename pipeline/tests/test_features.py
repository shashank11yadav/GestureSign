from __future__ import annotations

import numpy as np
import pytest

from gesturesign import features as ft
from gesturesign import landmarks as lm
from tests.helpers import blank_frames, full_body_frames, hand_shape, put_shoulders


@pytest.mark.parametrize(("feature_set", "dim"), [("hands_pose", 193), ("hands_pose_lips", 274)])
def test_frame_feature_dim(feature_set: ft.FeatureSet, dim: int) -> None:
    assert ft.frame_feature_dim(feature_set) == dim
    out = ft.frame_features(full_body_frames(1)[0], feature_set)
    assert out.shape == (dim,)
    assert out.dtype == np.float32


@pytest.mark.parametrize("feature_set", ["hands_pose", "hands_pose_lips"])
def test_all_nan_frame_gives_zeros_and_no_presence(feature_set: ft.FeatureSet) -> None:
    out = ft.frame_features(blank_frames(1)[0], feature_set)
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)
    layout = ft.feature_layout(feature_set)
    assert np.all(out[layout.presence] == 0.0)


def test_centres_on_shoulder_midpoint_and_scales_by_shoulder_width() -> None:
    frames = blank_frames(1)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))  # width 0.2, centre (0.5, 0.5)
    frames[0, lm.right_hand(0), :2] = (0.7, 0.3)
    frames[0, lm.RIGHT_HAND_START + 1 : lm.RIGHT_HAND_START + lm.N_HAND, :2] = (0.7, 0.3)
    out = ft.frame_features(frames[0], "hands_pose")
    layout = ft.feature_layout("hands_pose")
    dom = out[layout.dominant_hand].reshape(lm.N_HAND, 2)
    np.testing.assert_allclose(dom[0], (1.0, -1.0), atol=1e-6)
    pose = out[layout.pose].reshape(len(lm.POSE_UPPER), 2)
    left_sh = lm.POSE_UPPER.index(lm.POSE_LEFT_SHOULDER)
    np.testing.assert_allclose(pose[left_sh], (0.5, 0.0), atol=1e-6)


def test_presence_flags_follow_detected_parts() -> None:
    frames = blank_frames(1)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))
    frames[0, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = hand_shape(
        (0.3, 0.4), 0.05, np.random.default_rng(0)
    )
    out = ft.frame_features(frames[0], "hands_pose_lips")
    layout = ft.feature_layout("hands_pose_lips")
    # dominant hand, other hand, pose, lips
    np.testing.assert_array_equal(out[layout.presence], [1.0, 0.0, 1.0, 0.0])


def test_partly_missing_part_is_zeroed_not_nan() -> None:
    frames = full_body_frames(1)
    frames[0, lm.pose(13), :] = np.nan  # one elbow lost
    out = ft.frame_features(frames[0], "hands_pose")
    assert np.all(np.isfinite(out))


def test_only_left_hand_is_mirrored_into_the_dominant_slot() -> None:
    rng = np.random.default_rng(3)
    right_only = blank_frames(1)
    put_shoulders(right_only, (0.6, 0.5), (0.4, 0.5))
    right_only[0, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = hand_shape(
        (0.35, 0.4), 0.05, rng
    )
    left_only = ft.mirror_landmarks(right_only)
    assert np.all(np.isnan(left_only[0, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND]))
    a = ft.frame_features(right_only[0], "hands_pose")
    b = ft.frame_features(left_only[0], "hands_pose")
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("feature_set", ["hands_pose", "hands_pose_lips"])
def test_mirrored_sequence_gives_identical_features(feature_set: ft.FeatureSet) -> None:
    frames = full_body_frames(12, seed=5)
    # Make the left hand dominant: drop the right hand from most frames.
    frames[3:, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = np.nan
    mirrored = ft.mirror_landmarks(frames)
    a = ft.sequence_features(frames, feature_set)
    b = ft.sequence_features(mirrored, feature_set)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_mirror_landmarks_is_an_involution() -> None:
    frames = full_body_frames(4, seed=2)
    twice = ft.mirror_landmarks(ft.mirror_landmarks(frames))
    np.testing.assert_allclose(twice, frames, atol=1e-6)


def test_hand_local_shape_is_scale_and_position_invariant() -> None:
    rng = np.random.default_rng(7)
    small = blank_frames(1)
    put_shoulders(small, (0.6, 0.5), (0.4, 0.5))
    shape = hand_shape((0.3, 0.4), 0.05, rng)
    small[0, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = shape
    big = small.copy()
    moved = shape.copy()
    moved[:, :2] = (shape[:, :2] - shape[0, :2]) * 2.0 + (0.2, 0.3)
    big[0, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = moved
    layout = ft.feature_layout("hands_pose")
    a = ft.frame_features(small[0], "hands_pose")[layout.dominant_local]
    b = ft.frame_features(big[0], "hands_pose")[layout.dominant_local]
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_anchor_uses_median_of_frames_with_shoulders() -> None:
    frames = blank_frames(3)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))
    frames[1, lm.pose(lm.POSE_LEFT_SHOULDER), :2] = (0.9, 0.5)  # outlier frame
    frames[1, lm.pose(lm.POSE_RIGHT_SHOULDER), :2] = (0.5, 0.5)
    frames[2, lm.pose(lm.POSE_LEFT_SHOULDER), :] = np.nan  # no shoulders here
    anchor = ft.compute_anchor(frames)
    # Valid frames: centre 0.5 width 0.2, and centre 0.7 width 0.4 -> medians are the means.
    assert (anchor.cx, anchor.cy, anchor.scale) == pytest.approx((0.6, 0.5, 0.3))


def test_anchor_falls_back_to_default_without_shoulders() -> None:
    assert ft.compute_anchor(blank_frames(2)) == ft.DEFAULT_ANCHOR


def test_sequence_features_shape_and_velocity() -> None:
    frames = blank_frames(3)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))
    shape = hand_shape((0.3, 0.4), 0.05, np.random.default_rng(1))
    for t in range(3):
        moved = shape.copy()
        moved[:, 0] += 0.02 * t
        frames[t, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = moved
    frames[1, lm.LEFT_HAND_START : lm.LEFT_HAND_START + lm.N_HAND] = shape  # appears once
    out = ft.sequence_features(frames, "hands_pose")
    layout = ft.feature_layout("hands_pose")
    assert out.shape == (3, ft.sequence_feature_dim("hands_pose"))
    assert ft.sequence_feature_dim("hands_pose") == 193 + 106
    vel = out[:, layout.velocity]
    assert np.all(vel[0] == 0.0)
    dom_vx = vel[1:, 0:42:2]  # x velocity of the dominant hand, frames 1 and 2
    np.testing.assert_allclose(dom_vx, 0.02 / 0.2, atol=1e-5)
    # The other hand is only present in frame 1, so it has no velocity anywhere.
    other = vel[:, 42:84]
    assert np.all(other == 0.0)


def test_empty_sequence_gives_empty_features() -> None:
    out = ft.sequence_features(blank_frames(0), "hands_pose")
    assert out.shape == (0, ft.sequence_feature_dim("hands_pose"))


def test_tied_hand_counts_pick_the_hand_that_moves_more() -> None:
    frames = full_body_frames(10, seed=8)  # both hands in every frame; the right moves more
    assert not ft.left_is_dominant(frames)
    assert ft.left_is_dominant(ft.mirror_landmarks(frames))


def test_tie_break_keeps_mirrored_two_handed_clips_identical() -> None:
    frames = full_body_frames(10, seed=9)
    a = ft.sequence_features(frames, "hands_pose")
    b = ft.sequence_features(ft.mirror_landmarks(frames), "hands_pose")
    np.testing.assert_allclose(a, b, atol=1e-5)


def _with_face(frames: np.ndarray, eye_mouth: float, ear_ear: float) -> None:
    """Pose eyes, mouth and ears around (0.5, 0.3) with the given spans."""
    for idx, (x, y) in {
        2: (0.52, 0.3), 5: (0.48, 0.3),
        9: (0.51, 0.3 + eye_mouth), 10: (0.49, 0.3 + eye_mouth),
        7: (0.5 + ear_ear / 2, 0.31), 8: (0.5 - ear_ear / 2, 0.31),
    }.items():  # fmt: skip
        frames[:, lm.pose(idx), :2] = (x, y)
        frames[:, lm.pose(idx), 2] = 0.0


def test_vertical_scale_follows_the_face_proportion() -> None:
    frames = blank_frames(2)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))
    _with_face(frames, eye_mouth=0.086, ear_ear=0.1)  # twice the reference ratio
    anchor = ft.compute_anchor(frames)
    assert anchor.scale == pytest.approx(0.2)
    assert anchor.y_scale == pytest.approx(0.2 * 0.86 / ft.FACE_RATIO_REF)


def test_camera_aspect_ratio_cancels_out() -> None:
    """The same scene in a portrait and a landscape image gives the same features."""
    portrait = full_body_frames(10, seed=4)
    _with_face(portrait, eye_mouth=0.043, ear_ear=0.1)
    landscape = portrait.copy()
    # Same pixels seen through a frame whose height is 0.55x as tall relative to its width:
    # normalised y stretches by 1 / 0.55 about the image centre.
    landscape[..., 1] = 0.5 + (portrait[..., 1] - 0.5) / 0.55
    a = ft.sequence_features(portrait, "hands_pose")
    b = ft.sequence_features(landscape, "hands_pose")
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_no_face_points_means_no_vertical_correction() -> None:
    frames = blank_frames(1)
    put_shoulders(frames, (0.6, 0.5), (0.4, 0.5))
    anchor = ft.compute_anchor(frames)
    assert anchor.y_scale == anchor.scale
