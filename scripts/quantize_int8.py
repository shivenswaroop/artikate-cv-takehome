#!/usr/bin/env python3
"""Dynamic INT8 quantization for ONNX (CPU ORT; no calibration dataset required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import WEIGHTS_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, default=str(WEIGHTS_DIR / "best.onnx"))
    p.add_argument("--out", type=str, default=str(WEIGHTS_DIR / "best_int8.onnx"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(args.onnx)
    if not src.exists():
        raise FileNotFoundError(src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic quantization: weights → INT8; activations remain float at runtime.
    # Suitable for CPU ORT demos. Jetson deployments should prefer TensorRT PTQ/QAT.
    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 (dynamic) ONNX → {out}")
    print(
        "Note: dynamic INT8 ≠ TensorRT calibration INT8. "
        "Report both latency and any accuracy delta honestly."
    )


if __name__ == "__main__":
    main()
