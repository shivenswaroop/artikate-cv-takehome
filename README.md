# Artikate Studio — Senior CV / ML Take-Home

End-to-end industrial detection pipeline: fine-tune **YOLOv8n** → export **ONNX** → run **ONNX Runtime (CPU)** → benchmark **FP32 vs dynamic INT8** → per-frame video report.

Private Artikate images/video are **not** in this repo. Smoke data is synthetic and only proves the pipeline. Submission metrics must be regenerated on the private pack.

## Hardware (this machine)

| Item | Value |
|------|--------|
| CPU | AMD Ryzen 5 5500U (12 threads) |
| RAM | ~7.1 GiB |
| GPU | None (no NVIDIA) — **CPU-only** |
| Inference | ONNX Runtime `CPUExecutionProvider` |
| Quantization | ORT **dynamic INT8** (not TensorRT PTQ) |

**Confidence those latency numbers hold elsewhere:** low–medium for absolute ms. Relative FP32 vs INT8 trends may transfer; Jetson Orin + TensorRT will be much faster and may show different INT8 accuracy. Re-measure on the target device.

## Loom / screen recording

Paste unlisted link here after recording the 5–8 min walkthrough on the **private** held-out clip:

```
LOOM_URL: <paste after recording>
```

Suggested narration: env → train/load weights → export ONNX → INT8 → `infer_video.py` → show CSV + latency summary → `benchmark.py` → worst failure cases.

## Quick start

```bash
cd Artikate_assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Smoke dataset (synthetic) — or install private data under data/artikate/
python scripts/prepare_smoke_data.py

# 2) Train (CPU defaults: batch=2)
python scripts/train.py --data configs/data.yaml --epochs 30 --batch 2 --device cpu

# 3) Export + quantize
python scripts/export_onnx.py --weights weights/best.pt
python scripts/quantize_int8.py --onnx weights/best.onnx

# 4) Val metrics
python scripts/eval_val.py --weights weights/best.pt --data configs/data.yaml
# optional ONNX P/R at fixed conf/IoU:
python scripts/eval_val.py --onnx weights/best.onnx --data configs/data.yaml

# 5) Held-out video (smoke path shown; swap for Artikate clip)
python scripts/infer_video.py \
  --model weights/best.onnx \
  --video data/smoke/video/heldout_smoke.mp4 \
  --save-vis

# 6) FP32 vs INT8 latency
python scripts/benchmark.py \
  --fp32 weights/best.onnx \
  --int8 weights/best_int8.onnx \
  --video data/smoke/video/heldout_smoke.mp4
```

### Private dataset drop-in

1. Unpack Artikate images/labels into `data/artikate/` (YOLO layout: `images/train`, `images/val`, `labels/train`, `labels/val`).
2. Copy `configs/data_artikate.yaml.example` → `configs/data.yaml` and set class names / `nc`.
3. Place held-out video at `data/artikate/video/heldout.mp4` (or pass `--video`).
4. Re-run train → export → infer → benchmark → fill numbers below and `results/FAILURE_CASES.md`.

## Measured numbers

_Fill after a real run (smoke or private)._

| Metric | FP32 ONNX | INT8 dynamic ONNX |
|--------|-----------|-------------------|
| Median latency (ms) | _TBD_ | _TBD_ |
| p95 latency (ms) | _TBD_ | _TBD_ |
| Val precision | _TBD_ | _TBD_ |
| Val recall | _TBD_ | _TBD_ |
| mAP50 (Ultralytics, PT) | _TBD_ | n/a |

Artifacts: `results/benchmark.json`, `results/val_metrics.json`, `results/video_detections.csv`, `results/video_latency_summary.txt`.

## Repo map

- `ANSWERS.md` — Sections 1, 3, 4
- `scripts/` — train, export, quantize, infer, benchmark, eval
- `src/` — letterbox + ORT detector
- `section3/` — placeholder until Artikate buggy repo arrives
- `tests/` — unit tests (letterbox + Section 3 when available)

## Section 3

Awaiting the provided buggy repo. Integration notes in `section3/README.md`.
