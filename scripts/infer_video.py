#!/usr/bin/env python3
"""Run ONNX detector on a video; write per-frame detections + latency CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.onnx_infer import YoloOnnxDetector
from src.paths import RESULTS_DIR, WEIGHTS_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(WEIGHTS_DIR / "best.onnx"))
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out-csv", type=str, default=str(RESULTS_DIR / "video_detections.csv"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--save-vis", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    detector = YoloOnnxDetector(
        args.model, imgsz=args.imgsz, conf_thres=args.conf, iou_thres=args.iou
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = RESULTS_DIR / "frames"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "det_idx",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "latency_ms",
            ],
        )
        writer.writeheader()
        frame_idx = 0
        pbar = tqdm(desc="infer", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break
            result = detector.predict(frame)
            latencies.append(result["latency_ms"])
            boxes = result["boxes"]
            scores = result["scores"]
            classes = result["classes"]
            if len(scores) == 0:
                writer.writerow(
                    {
                        "frame": frame_idx,
                        "det_idx": -1,
                        "class_id": "",
                        "confidence": "",
                        "x1": "",
                        "y1": "",
                        "x2": "",
                        "y2": "",
                        "latency_ms": f"{result['latency_ms']:.3f}",
                    }
                )
            else:
                for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
                    writer.writerow(
                        {
                            "frame": frame_idx,
                            "det_idx": i,
                            "class_id": int(cls_id),
                            "confidence": f"{float(score):.6f}",
                            "x1": f"{float(box[0]):.2f}",
                            "y1": f"{float(box[1]):.2f}",
                            "x2": f"{float(box[2]):.2f}",
                            "y2": f"{float(box[3]):.2f}",
                            "latency_ms": f"{result['latency_ms']:.3f}",
                        }
                    )
            if args.save_vis and frame_idx % 15 == 0:
                vis = frame.copy()
                for box, score, cls_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{int(cls_id)}:{score:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                cv2.imwrite(str(vis_dir / f"frame_{frame_idx:05d}.jpg"), vis)
            frame_idx += 1
            pbar.update(1)
        pbar.close()
    cap.release()

    if latencies:
        import numpy as np

        arr = np.array(latencies)
        summary = RESULTS_DIR / "video_latency_summary.txt"
        summary.write_text(
            "\n".join(
                [
                    f"frames={len(arr)}",
                    f"latency_ms_mean={arr.mean():.3f}",
                    f"latency_ms_median={np.median(arr):.3f}",
                    f"latency_ms_p95={np.percentile(arr, 95):.3f}",
                    f"model={args.model}",
                    f"video={video_path}",
                ]
            )
            + "\n"
        )
        print(summary.read_text())
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
