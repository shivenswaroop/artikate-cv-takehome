"""Coordinate-space tests for letterbox inverse mapping."""

from __future__ import annotations

import numpy as np

from src.letterbox import letterbox, scale_boxes_to_original


def test_scale_boxes_roundtrip_center_box():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lb, ratio, pad = letterbox(img, (640, 640))
    assert lb.shape[0] == 640 and lb.shape[1] == 640

    # Box in letterboxed space that should map near image center
    # After letterbox of 640x480 → 640x640, vertical pad is 80 total → 40 each side.
    boxes = np.array([[320 - 50, 320 - 40, 320 + 50, 320 + 40]], dtype=np.float32)
    out = scale_boxes_to_original(boxes, ratio, pad, (480, 640))
    assert out.shape == (1, 4)
    # Mapped box should sit inside original frame
    assert out[0, 0] >= 0 and out[0, 2] <= 640
    assert out[0, 1] >= 0 and out[0, 3] <= 480


def test_empty_boxes():
    out = scale_boxes_to_original(np.zeros((0, 4), dtype=np.float32), 1.0, (0, 0), (480, 640))
    assert out.shape == (0, 4)
