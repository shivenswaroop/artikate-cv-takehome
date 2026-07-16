"""Letterbox helpers matching Ultralytics-style resize+pad for ORT inference."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image to new_shape while preserving aspect ratio.

    Returns:
        padded BGR image, scale ratio (new / old), (pad_w, pad_h) as total padding
        applied equally on both sides (left/right or top/bottom).
    """
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image, r, (dw, dh)


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    original_shape: Tuple[int, int],
) -> np.ndarray:
    """Map boxes from letterboxed network space back to original image pixels."""
    if boxes_xyxy.size == 0:
        return boxes_xyxy
    out = boxes_xyxy.copy().astype(np.float32)
    pad_w, pad_h = pad
    out[:, [0, 2]] -= pad_w
    out[:, [1, 3]] -= pad_h
    out[:, :4] /= ratio
    h, w = original_shape
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out
