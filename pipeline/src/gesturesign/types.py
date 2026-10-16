"""Array type aliases used across the pipeline."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Landmarks, [..., 543, 3] float32 with NaN for missing points.
Frames = npt.NDArray[np.float32]
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
