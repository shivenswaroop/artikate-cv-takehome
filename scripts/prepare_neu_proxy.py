#!/usr/bin/env python3
"""Assemble a ~150-image NEU-DET stand-in client dataset + held-out video.

Source: https://github.com/KingRedMan/NEU-DET_Yolo (YOLO labels for NEU-DET
hot-rolled steel surface defects). We take a stratified subset so the pack
matches the assignment's ~150-image scale.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def ensure_source(src: Path) -> Path:
    if (src / "ImageSets" / "train" / "images").exists():
        return src
    src.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.rmtree(src)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/KingRedMan/NEU-DET_Yolo.git",
            str(src),
        ]
    )
    return src


def class_key(stem: str) -> str:
    for name in CLASS_NAMES:
        if stem.startswith(name):
            return name
    return "unknown"


def list_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def stratified_sample(
    pairs: list[tuple[Path, Path]], n: int, rng: random.Random
) -> list[tuple[Path, Path]]:
    by_cls: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for p in pairs:
        by_cls[class_key(p[0].stem)].append(p)
    for v in by_cls.values():
        rng.shuffle(v)
    # round-robin across classes for balance
    picked: list[tuple[Path, Path]] = []
    keys = sorted(by_cls.keys())
    i = 0
    while len(picked) < n and any(by_cls.values()):
        k = keys[i % len(keys)]
        if by_cls[k]:
            picked.append(by_cls[k].pop())
        i += 1
        if i > n * 20:
            break
    return picked


def copy_pairs(pairs: list[tuple[Path, Path]], img_out: Path, lbl_out: Path) -> None:
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)


def write_video(frames: list[Path], out_path: Path, fps: int = 15) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    h, w = first.shape[:2]
    # Upscale slightly so the clip looks more like a camera feed
    tw, th = w * 2, h * 2
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th)
    )
    # ~30s at 15fps = 450 frames; repeat / sample from pool
    target = fps * 30
    for i in range(target):
        img = cv2.imread(str(frames[i % len(frames)]))
        if img is None:
            continue
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()


def write_yaml(out: Path) -> None:
    text = f"""# NEU-DET proxy client dataset (~150 images)
path: data/proxy_neu
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        text += f"  {i}: {name}\n"
    out.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=str,
        default=str(ROOT / "data" / "_cache" / "NEU-DET_Yolo"),
    )
    p.add_argument("--out", type=str, default=str(ROOT / "data" / "proxy_neu"))
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--val", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    src = ensure_source(Path(args.source))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    train_all = list_pairs(src / "ImageSets" / "train")
    val_all = list_pairs(src / "ImageSets" / "val")
    test_all = list_pairs(src / "ImageSets" / "test")

    train_pick = stratified_sample(train_all, args.train, rng)
    val_pick = stratified_sample(val_all, args.val, rng)
    # held-out frames from official test split (never in train/val)
    video_pool = stratified_sample(test_all, min(60, len(test_all)), rng)

    copy_pairs(train_pick, out / "images" / "train", out / "labels" / "train")
    copy_pairs(val_pick, out / "images" / "val", out / "labels" / "val")
    write_video([p[0] for p in video_pool], out / "video" / "heldout.mp4")
    write_yaml(ROOT / "configs" / "data.yaml")

    meta = out / "DATASET.md"
    meta.write_text(
        "\n".join(
            [
                "# Proxy client dataset — NEU-DET subset",
                "",
                f"- Train images: {len(train_pick)}",
                f"- Val images: {len(val_pick)}",
                f"- Held-out video: video/heldout.mp4 (~30s @ 15fps from {len(video_pool)} unused test stills)",
                f"- Classes ({len(CLASS_NAMES)}): {', '.join(CLASS_NAMES)}",
                "- Upstream: KingRedMan/NEU-DET_Yolo (YOLO labels for NEU-DET)",
                "- Original NEU-DET: Northeastern University hot-rolled steel surface defects",
                "",
            ]
        )
    )
    print(f"Wrote proxy dataset → {out}")
    print(meta.read_text())


if __name__ == "__main__":
    main()
