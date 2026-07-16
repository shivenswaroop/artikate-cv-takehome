#!/usr/bin/env python3
"""Create a tiny YOLO-format smoke dataset + short video for pipeline dry-runs.

Uses synthetic shapes so the repo stays self-contained without downloading large
datasets. Replace with Artikate private data for submission metrics.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def draw_object(img: np.ndarray, rng: random.Random) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    bw = rng.randint(40, 120)
    bh = rng.randint(40, 120)
    x1 = rng.randint(10, max(11, w - bw - 10))
    y1 = rng.randint(10, max(11, h - bh - 10))
    x2, y2 = x1 + bw, y1 + bh
    color = (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))
    if rng.random() < 0.5:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), min(bw, bh) // 2, color, -1)
    return x1, y1, x2, y2


def to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def write_split(out_root: Path, split: str, n: int, seed: int) -> None:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60))
        # noise
        noise = rng.randint(0, 25)
        img = np.clip(img + np.random.default_rng(seed + i).integers(0, noise + 1, img.shape), 0, 255).astype(
            np.uint8
        )
        boxes = [draw_object(img, rng) for _ in range(rng.randint(1, 3))]
        name = f"{split}_{i:03d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
        with (lbl_dir / f"{name}.txt").open("w") as f:
            for b in boxes:
                f.write(to_yolo(*b, w, h))


def write_video(out_path: Path, n_frames: int = 45, seed: int = 99) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (w, h))
    cx, cy = w // 2, h // 2
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        cx = int(w * (0.2 + 0.6 * (i / max(1, n_frames - 1))))
        cy = int(h * (0.3 + 0.2 * np.sin(i / 3)))
        cv2.rectangle(img, (cx - 40, cy - 30), (cx + 40, cy + 30), (0, 200, 80), -1)
        if rng.random() < 0.3:
            draw_object(img, rng)
        writer.write(img)
    writer.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(ROOT / "data" / "smoke"))
    p.add_argument("--train", type=int, default=24)
    p.add_argument("--val", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    write_split(out, "train", args.train, seed=1)
    write_split(out, "val", args.val, seed=2)
    write_video(out / "video" / "heldout_smoke.mp4")
    print(f"Smoke dataset ready at {out}")
    print(f"Video: {out / 'video' / 'heldout_smoke.mp4'}")


if __name__ == "__main__":
    main()
