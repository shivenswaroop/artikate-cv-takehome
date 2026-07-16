"""Regression test that would have caught the Section 3 letterbox inverse bug.

Historical bug (fixed): in section3/postprocess.py, vertical pad used `+= pad_h`
instead of `-= pad_h` when mapping letterbox → original coordinates.
"""

from __future__ import annotations

import numpy as np

from section3.postprocess import boxes_from_network_output
from src.letterbox import letterbox


def test_vertical_pad_is_subtracted_not_added():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, ratio, pad = letterbox(frame, (640, 640))
    pad_w, pad_h = pad
    assert pad_h > 0

    # Content-center in letterbox space.
    content_cy = pad_h + (480 * ratio) / 2
    boxes = np.array(
        [[320 - 20, content_cy - 20, 320 + 20, content_cy + 20]],
        dtype=np.float32,
    )
    out = boxes_from_network_output(boxes, ratio, pad, (480, 640))
    cy = 0.5 * (out[0, 1] + out[0, 3])

    # Must land at vertical center of the original 480-high frame (~240).
    assert abs(cy - 240.0) < 2.0, (
        f"expected cy≈240, got {cy:.2f}. Wrong pad_h sign drifts boxes vertically."
    )

    # Explicitly guard against the old `+= pad_h` mistake.
    wrong = boxes.copy()
    wrong[:, [0, 2]] -= pad_w
    wrong[:, [1, 3]] += pad_h
    wrong /= ratio
    wrong_cy = 0.5 * (wrong[0, 1] + wrong[0, 3])
    assert abs(wrong_cy - 240.0) > 10.0, "sanity: buggy sign should not pass this test"
