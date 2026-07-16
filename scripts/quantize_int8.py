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


def _clamp_ir(path: Path, max_ir: int = 10) -> None:
    import onnx

    model = onnx.load(str(path))
    if model.ir_version > max_ir:
        model.ir_version = max_ir
        onnx.save(model, str(path))


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(args.onnx)
    if not src.exists():
        raise FileNotFoundError(src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Full Conv dynamic INT8 often emits ConvInteger, which many CPU ORT builds
    # cannot execute for YOLO graphs. Prefer MatMul/Gemm weight-only INT8.
    quantize_dynamic(
        model_input=str(src),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    _clamp_ir(out)
    print(f"INT8 (dynamic MatMul/Gemm) ONNX → {out}")
    print(
        "Note: this is weight-only dynamic INT8 on CPU ORT, not TensorRT PTQ. "
        "Jetson numbers will differ; re-benchmark on target hardware."
    )


if __name__ == "__main__":
    main()
