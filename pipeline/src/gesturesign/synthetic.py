"""Synthetic stand-in for Kaggle ``asl-signs``, with the same files and parquet schema.

It exists so the whole pipeline (and the browser's Replay mode) can run before
the real data is available, and so tests never need the 40 GB download.

Each synthetic "sign" is a procedural template: a path for the signing wrist,
a hand shape that changes over time and an optional second hand. Each
participant varies body position and size, speed, jitter and handedness
(left-handed signers are a mirror image), and drops detections the way
MediaPipe does. The output layout matches Kaggle::

    train.csv                                   path, participant_id, sequence_id, sign
    sign_to_prediction_index_map.json
    train_landmark_files/<participant>/<sequence>.parquet
        frame, row_id, type, landmark_index, x, y, z   (543 rows per frame)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from gesturesign import landmarks as lm
from gesturesign.features import mirror_landmarks
from gesturesign.types import Frames

MOTIONS = ("circle", "tap", "line", "arc", "zigzag", "twist")
SHAPES = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0),
    "fist": (0.9, 0.95, 0.95, 0.95, 0.95),
    "point": (0.7, 0.0, 0.95, 0.95, 0.95),
    "vee": (0.7, 0.0, 0.0, 0.95, 0.95),
    "claw": (0.3, 0.55, 0.55, 0.55, 0.55),
    "pinky": (0.8, 0.95, 0.95, 0.95, 0.0),
}
START_REGIONS = {
    "chin": (0.0, -1.35),
    "chest": (0.0, 0.5),
    "side": (-0.9, 0.2),
    "temple": (-0.45, -2.0),
}


@dataclass(frozen=True)
class SignTemplate:
    name: str
    motion: str
    start: tuple[float, float]  # wrist start, in shoulder half-widths from the shoulder centre
    size: float  # motion amplitude, same units
    shape_from: tuple[float, ...]
    shape_to: tuple[float, ...]
    roll: float  # hand rotation over the sign, radians
    two_handed: bool


def sign_templates(n_signs: int, seed: int = 0) -> list[SignTemplate]:
    rng = np.random.default_rng(seed)
    shapes = list(SHAPES)
    regions = list(START_REGIONS)
    out: list[SignTemplate] = []
    for i in range(n_signs):
        motion = MOTIONS[i % len(MOTIONS)]
        shape = shapes[(i * 5 + i // len(MOTIONS)) % len(shapes)]
        end_shape = shapes[(i * 5 + i // len(MOTIONS) + (i % 3)) % len(shapes)]
        region = regions[(i + i // len(MOTIONS)) % len(regions)]
        two = bool(i % 4 == 3)
        name = f"{motion}-{shape}" + ("" if shape == end_shape else f"-{end_shape}")
        if two:
            name += "-both"
        while name in {t.name for t in out}:
            name += "+"
        out.append(
            SignTemplate(
                name=name,
                motion=motion,
                start=START_REGIONS[region],
                size=float(rng.uniform(0.35, 0.7)),
                shape_from=SHAPES[shape],
                shape_to=SHAPES[end_shape],
                roll=float(rng.uniform(-0.6, 0.6)),
                two_handed=two,
            )
        )
    return out


def _path(motion: str, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Unit-scale wrist offset [T, 2] for progress ``s`` in [0, 1]."""
    if motion == "circle":
        a = 2 * math.pi * s
        return np.stack([np.sin(a), 1 - np.cos(a)], axis=1) * 0.5
    if motion == "tap":
        return np.stack([np.zeros_like(s), 0.35 * np.abs(np.sin(3 * math.pi * s))], axis=1)
    if motion == "line":
        return np.stack([s * 0.9, -s * 0.5], axis=1)
    if motion == "arc":
        a = math.pi * s
        return np.stack([-np.cos(a) * 0.5 + 0.5, -np.sin(a) * 0.5], axis=1)
    if motion == "zigzag":
        return np.stack([s * 0.8, 0.25 * np.sin(4 * math.pi * s)], axis=1)
    # twist: the wrist barely moves; the hand rolls and changes shape
    return np.stack([0.08 * np.sin(2 * math.pi * s), np.zeros_like(s)], axis=1)


# Finger bases (across, along) in hand lengths, and their splay angles.
_FINGER_BASES = ((-0.30, 0.25), (-0.22, 0.85), (-0.03, 0.9), (0.15, 0.85), (0.30, 0.75))
_FINGER_ANGLES = (-0.9, -0.12, 0.0, 0.12, 0.28)
_FINGER_SEGMENTS = ((0.25, 0.22, 0.2), (0.4, 0.25, 0.2), (0.45, 0.28, 0.22), (0.4, 0.26, 0.2))


def hand_points(
    wrist: tuple[float, float], length: float, angle: float, curls: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """21 x 3 points of a right hand seen palm-on.

    ``angle`` rotates the hand in the image (0 = fingers up), ``curls`` bends
    each finger (thumb first) from 0 (straight) to 1 (folded onto the palm).
    """
    curl = np.asarray(curls, dtype=np.float64)
    pts = np.zeros((lm.N_HAND, 3))
    ca, sa = math.cos(angle), math.sin(angle)

    def place(across: float, along: float, depth: float) -> tuple[float, float, float]:
        # Fingers point up the image (-y) when angle is 0.
        dx, dy = across * length, -along * length
        return (wrist[0] + ca * dx - sa * dy, wrist[1] + sa * dx + ca * dy, depth * length)

    pts[0] = (wrist[0], wrist[1], 0.0)
    for finger in range(5):
        base_across, base_along = _FINGER_BASES[finger]
        across, along, depth = base_across, base_along, 0.0
        pts[1 + finger * 4] = place(across, along, depth)
        segments = _FINGER_SEGMENTS[0] if finger == 0 else _FINGER_SEGMENTS[finger - 1]
        bend = 0.0
        splay = _FINGER_ANGLES[finger]
        for joint, seg in enumerate(segments):
            bend += curl[finger] * (1.4 if finger else 0.8)
            across += seg * math.sin(splay) * math.cos(bend)
            along += seg * math.cos(splay) * math.cos(bend)
            depth -= seg * math.sin(bend)
            if finger == 0:  # the thumb folds across the palm
                across += seg * curl[0] * 0.6
            pts[2 + finger * 4 + joint] = place(across, along, depth)
    return pts


def _elbow(shoulder: npt.NDArray[np.float64], wrist: npt.NDArray[np.float64], arm: float,
           outward: float) -> npt.NDArray[np.float64]:  # fmt: skip
    mid = (shoulder + wrist) / 2
    d = wrist - shoulder
    dist = float(np.hypot(d[0], d[1])) or 1e-6
    reach = math.sqrt(max(arm**2 - (dist / 2) ** 2, 0.0))
    normal = np.array([d[1], -d[0]]) / dist
    if normal[1] < 0:  # elbows hang down
        normal = -normal
    normal[0] = abs(normal[0]) * outward if abs(normal[0]) > 0 else outward * 0.1
    out: npt.NDArray[np.float64] = mid + normal * reach
    return out


def _face(head: npt.NDArray[np.float64], hw: float, rng: np.random.Generator,
          mouth_open: float) -> npt.NDArray[np.float64]:  # fmt: skip
    pts = np.zeros((lm.N_FACE, 3))
    a = rng.uniform(0, 2 * math.pi, lm.N_FACE)
    r = np.sqrt(rng.uniform(0, 1, lm.N_FACE))
    pts[:, 0] = head[0] + np.cos(a) * r * 0.5 * hw
    pts[:, 1] = head[1] + np.sin(a) * r * 0.68 * hw
    pts[:, 2] = -0.02 * (1 - r)
    mouth = head + np.array([0.0, 0.38 * hw])
    for run_i, run in enumerate(lm.LIP_RUNS):
        outer = run_i < 2
        lower = run_i in (0, 2)
        rx = (0.2 if outer else 0.15) * hw
        ry = (0.07 if outer else 0.03) * hw + (mouth_open * hw * 0.08 if lower else 0.0)
        for k, idx in enumerate(run):
            t = math.pi * k / (len(run) - 1)
            # Runs go from the person's right mouth corner (image left) to the left.
            x = mouth[0] - rx * math.cos(t)
            y = mouth[1] + (ry if lower else -ry) * math.sin(t)
            pts[idx] = (x, y, -0.01)
    return pts


def render_sign(
    template: SignTemplate,
    rng: np.random.Generator,
    *,
    left_handed: bool = False,
    T: int | None = None,
    body_shift: tuple[float, float] = (0.0, 0.0),
    body_scale: float = 1.0,
    jitter: float = 0.004,
    dropout: float = 0.03,
) -> Frames:
    """One clip of ``template`` as [T, 543, 3] landmarks, NaN where undetected."""
    if T is None:
        T = int(rng.integers(26, 52))
    hw = 0.13 * body_scale
    cx, cy = 0.5 + body_shift[0], 0.62 + body_shift[1]
    r_sh = np.array([cx - hw, cy])
    l_sh = np.array([cx + hw, cy])
    head = np.array([cx, cy - 1.75 * hw])
    lead_in, lead_out = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    active = max(T - lead_in - lead_out, 4)
    s = np.clip((np.arange(T) - lead_in) / max(active - 1, 1), 0.0, 1.0)
    start = np.array([cx, cy]) + np.array(template.start) * hw
    path = start + _path(template.motion, s) * template.size * hw * 2
    curls = np.outer(1 - s, template.shape_from) + np.outer(s, template.shape_to)
    hand_len = 0.95 * hw
    arm = 1.25 * hw

    frames = np.full((T, lm.N_POINTS, 3), np.nan)
    mouth_phase = rng.uniform(0, 2 * math.pi)
    face_rng_seed = int(rng.integers(0, 2**31))
    for t in range(T):
        signing = lead_in <= t < T - lead_out
        face = _face(head, hw, np.random.default_rng(face_rng_seed), 0.5 + 0.5 * math.sin(
            mouth_phase + t * 0.3))  # fmt: skip
        frames[t, lm.FACE_START : lm.FACE_START + lm.N_FACE] = face

        pose = np.zeros((lm.N_POSE, 3))
        pose[:, 2] = -0.3
        pose[0, :2] = head
        for i, (dx, dy) in {
            1: (0.12, -0.2), 2: (0.22, -0.2), 3: (0.3, -0.2),
            4: (-0.12, -0.2), 5: (-0.22, -0.2), 6: (-0.3, -0.2),
            7: (0.5, -0.1), 8: (-0.5, -0.1), 9: (0.12, 0.33), 10: (-0.12, 0.33),
        }.items():  # fmt: skip
            pose[i, :2] = head + np.array([dx, dy]) * hw
        pose[11, :2], pose[12, :2] = l_sh, r_sh
        pose[23, :2] = (cx + 0.7 * hw, cy + 2.8 * hw)
        pose[24, :2] = (cx - 0.7 * hw, cy + 2.8 * hw)
        for i in range(25, 33):
            side = 1 if i % 2 else -1
            pose[i, :2] = (cx + side * 0.75 * hw, cy + (3.6 + 0.4 * (i - 25) // 2) * hw)

        r_wrist = path[t] if signing else np.array([cx - 0.8 * hw, cy + 2.6 * hw])
        roll = template.roll * s[t]
        r_hand = hand_points((float(r_wrist[0]), float(r_wrist[1])), hand_len, roll, curls[t])
        if template.two_handed:
            mirror = np.array([2 * cx - r_wrist[0], r_wrist[1]])
            l_hand = hand_points((float(mirror[0]), float(mirror[1])), hand_len, -roll, curls[t])
            l_hand[:, 0] = 2 * mirror[0] - l_hand[:, 0]  # a left hand is a mirrored right hand
            l_wrist = mirror
        else:
            l_hand = None
            l_wrist = np.array([cx + 0.8 * hw, cy + 2.6 * hw])
        pose[16, :2] = r_wrist
        pose[15, :2] = l_wrist
        pose[14, :2] = _elbow(r_sh, r_wrist, arm, -1.0)
        pose[13, :2] = _elbow(l_sh, l_wrist, arm, 1.0)
        for wrist_i, fingers in ((16, (18, 20, 22)), (15, (17, 19, 21))):
            for k, f in enumerate(fingers):
                pose[f, :2] = pose[wrist_i, :2] + np.array([(k - 1) * 0.08, -0.15]) * hw
        frames[t, lm.POSE_START : lm.POSE_START + lm.N_POSE] = pose

        if signing and rng.random() >= dropout:
            frames[t, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = r_hand
        if signing and l_hand is not None and rng.random() >= dropout:
            frames[t, lm.LEFT_HAND_START : lm.LEFT_HAND_START + lm.N_HAND] = l_hand

    frames[..., :2] += rng.normal(0, jitter, size=frames[..., :2].shape)
    out = frames.astype(np.float32)
    return mirror_landmarks(out) if left_handed else out


def frames_to_parquet_frame(frames: Frames, first_frame: int) -> pd.DataFrame:
    """Kaggle's long format: one row per (frame, landmark)."""
    T = frames.shape[0]
    types: list[str] = []
    idx: list[int] = []
    for name in ("face", "left_hand", "pose", "right_hand"):
        types += [name] * lm.TYPE_SIZES[name]
        idx += list(range(lm.TYPE_SIZES[name]))
    frame_no = np.repeat(np.arange(first_frame, first_frame + T, dtype=np.int16), lm.N_POINTS)
    type_col = np.tile(np.array(types, dtype=object), T)
    idx_col = np.tile(np.array(idx, dtype=np.int16), T)
    flat = frames.reshape(-1, 3)
    return pd.DataFrame(
        {
            "frame": frame_no,
            "row_id": [f"{f}-{t}-{i}" for f, t, i in zip(frame_no, type_col, idx_col, strict=True)],
            "type": type_col,
            "landmark_index": idx_col,
            "x": flat[:, 0],
            "y": flat[:, 1],
            "z": flat[:, 2],
        }
    )


def write_dataset(
    out_dir: Path,
    *,
    participants: list[int],
    n_signs: int,
    clips_per_sign: int,
    seed: int = 0,
    left_handed: frozenset[int] = frozenset(),
) -> pd.DataFrame:
    """Write a synthetic dataset in the Kaggle layout and return its ``train.csv`` table."""
    rng = np.random.default_rng(seed)
    templates = sign_templates(n_signs, seed)
    rows: list[dict[str, object]] = []
    sequence_id = 1_000_000 + seed * 100_000
    for pid in participants:
        p_rng = np.random.default_rng([seed, pid])
        shift = (float(p_rng.normal(0, 0.03)), float(p_rng.normal(0, 0.03)))
        scale = float(p_rng.uniform(0.85, 1.15))
        for template in templates:
            for _ in range(clips_per_sign):
                sequence_id += int(rng.integers(1, 50))
                frames = render_sign(
                    template,
                    rng,
                    left_handed=pid in left_handed,
                    body_shift=shift,
                    body_scale=scale,
                )
                rel = Path("train_landmark_files") / str(pid) / f"{sequence_id}.parquet"
                path = out_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                table = frames_to_parquet_frame(frames, int(rng.integers(10, 120)))
                table.to_parquet(path, index=False)
                rows.append(
                    {
                        "path": rel.as_posix(),
                        "participant_id": pid,
                        "sequence_id": sequence_id,
                        "sign": template.name,
                    }
                )
    table = pd.DataFrame(rows, columns=["path", "participant_id", "sequence_id", "sign"])
    table.to_csv(out_dir / "train.csv", index=False)
    names = sorted(t.name for t in templates)
    (out_dir / "sign_to_prediction_index_map.json").write_text(
        json.dumps({n: i for i, n in enumerate(names)}), encoding="utf-8"
    )
    return table
