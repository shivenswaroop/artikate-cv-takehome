#!/usr/bin/env python3
"""Export a trained Ultralytics checkpoint to ONNX."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import WEIGHTS_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "best.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--out", type=str, default=str(WEIGHTS_DIR / "best.onnx"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
    )
    exported_path = Path(exported)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, out)
    print(f"ONNX model → {out}")


if __name__ == "__main__":
    main()
