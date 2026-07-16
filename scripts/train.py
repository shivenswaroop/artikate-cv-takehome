#!/usr/bin/env python3
"""Fine-tune YOLOv8n on the configured dataset (CPU-friendly defaults)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import RESULTS_DIR, WEIGHTS_DIR, ensure_output_dirs
from src.resolve_data_yaml import materialize_data_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8n for Artikate assignment")
    p.add_argument("--data", type=str, default="configs/data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=2, help="Keep small on ~8GB RAM laptops")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workers", type=int, default=0, help="0 is safest on low-RAM CPUs")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    from ultralytics import YOLO

    data_yaml = materialize_data_yaml(args.data)
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / args.project),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if best.exists():
        dest = WEIGHTS_DIR / "best.pt"
        dest.write_bytes(best.read_bytes())
        print(f"Copied best weights → {dest}")

    metrics_path = RESULTS_DIR / "train_summary.txt"
    with metrics_path.open("w") as f:
        f.write(f"save_dir={results.save_dir}\n")
        f.write(f"best={best}\n")
        f.write(f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} device={args.device}\n")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
