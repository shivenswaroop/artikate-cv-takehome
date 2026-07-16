#!/usr/bin/env python3
"""Evaluate precision/recall on the YOLO val split using Ultralytics val() or ORT boxes.

Default path uses Ultralytics for apples-to-apples mAP with the training stack.
Use --onnx to score an exported ONNX model with a simple IoU matcher (P/R at one conf).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.onnx_infer import YoloOnnxDetector
from src.paths import RESULTS_DIR, WEIGHTS_DIR, ensure_output_dirs
from src.resolve_data_yaml import materialize_data_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="configs/data.yaml")
    p.add_argument("--weights", type=str, default=str(WEIGHTS_DIR / "best.pt"))
    p.add_argument("--onnx", type=str, default="", help="If set, evaluate ONNX with IoU matcher")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Return GT boxes xyxy in pixels."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = map(float, parts[:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clip(0).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def eval_onnx(data_yaml: Path, onnx_path: Path, conf: float, iou_thr: float, imgsz: int) -> dict:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = ROOT / root
    val_rel = cfg["val"]
    img_dir = root / val_rel
    # labels parallel to images/
    label_dir = root / "labels" / Path(val_rel).name

    det = YoloOnnxDetector(onnx_path, imgsz=imgsz, conf_thres=conf)
    tp = fp = fn = 0
    images = sorted(
        [
            *img_dir.glob("*.jpg"),
            *img_dir.glob("*.jpeg"),
            *img_dir.glob("*.png"),
            *img_dir.glob("*.bmp"),
        ]
    )
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo_labels(label_dir / f"{img_path.stem}.txt", w, h)
        pred = det.predict(bgr)["boxes"]
        if len(gt) == 0 and len(pred) == 0:
            continue
        if len(gt) == 0:
            fp += len(pred)
            continue
        if len(pred) == 0:
            fn += len(gt)
            continue
        iou = box_iou(pred, gt)
        matched_gt = set()
        for i in range(len(pred)):
            j = int(iou[i].argmax())
            if iou[i, j] >= iou_thr and j not in matched_gt:
                tp += 1
                matched_gt.add(j)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "mode": "onnx_iou_matcher",
        "images": len(images),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "conf": conf,
        "iou": iou_thr,
        "model": str(onnx_path),
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    data = Path(args.data)
    if not data.is_absolute():
        data = ROOT / data

    if args.onnx:
        report = eval_onnx(data, Path(args.onnx), args.conf, args.iou, args.imgsz)
    else:
        from ultralytics import YOLO

        model = YOLO(args.weights)
        data_resolved = materialize_data_yaml(data)
        metrics = model.val(
            data=str(data_resolved), imgsz=args.imgsz, device=args.device, split="val"
        )
        box = metrics.box
        report = {
            "mode": "ultralytics_val",
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "weights": args.weights,
            "data": str(data),
        }

    out = RESULTS_DIR / "val_metrics.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
