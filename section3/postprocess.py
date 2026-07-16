"""Postprocessing for detector heads."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def boxes_from_network_output(
    boxes_xyxy_letterbox: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Map network-space xyxy boxes back to the original frame.

    pad: (pad_w, pad_h) — half-padding applied on each side during letterbox.
    original_shape: (h, w)
    """
    if boxes_xyxy_letterbox.size == 0:
        return boxes_xyxy_letterbox

    out = boxes_xyxy_letterbox.astype(np.float32).copy()
    pad_w, pad_h = pad
    h, w = original_shape

    # Undo letterbox: remove padding on both axes, then divide by resize gain.
    out[:, [0, 2]] -= pad_w
    out[:, [1, 3]] -= pad_h
    out[:, :4] /= ratio

    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
