#!/usr/bin/env python3
"""Benchmark FP32 vs INT8 ONNX latency on a sample image or video frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.onnx_infer import YoloOnnxDetector
from src.paths import RESULTS_DIR, WEIGHTS_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", type=str, default=str(WEIGHTS_DIR / "best.onnx"))
    p.add_argument("--int8", type=str, default=str(WEIGHTS_DIR / "best_int8.onnx"))
    p.add_argument("--image", type=str, default="")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def load_bgr(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(args.image)
        return img
    if args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame from {args.video}")
        return frame
    # Synthetic fallback for CI / scaffold
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def bench(model_path: Path, bgr: np.ndarray, warmup: int, runs: int, imgsz: int) -> dict:
    det = YoloOnnxDetector(model_path, imgsz=imgsz)
    for _ in range(warmup):
        det.predict(bgr)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        det.predict(bgr)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "model": str(model_path),
        "runs": runs,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    bgr = load_bgr(args)
    report = {"hardware_note": "CPU ONNX Runtime on local laptop", "image_shape": list(bgr.shape)}

    fp32 = Path(args.fp32)
    if fp32.exists():
        report["fp32"] = bench(fp32, bgr, args.warmup, args.runs, args.imgsz)
    else:
        report["fp32"] = {"error": f"missing {fp32}"}

    int8 = Path(args.int8)
    if int8.exists():
        report["int8_dynamic"] = bench(int8, bgr, args.warmup, args.runs, args.imgsz)
    else:
        report["int8_dynamic"] = {"error": f"missing {int8}"}

    out = RESULTS_DIR / "benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
