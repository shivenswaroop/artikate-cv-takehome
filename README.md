# Artikate Studio — Senior CV / ML Take-Home

End-to-end industrial detection pipeline: fine-tune **YOLOv8n** → export **ONNX** → run **ONNX Runtime (CPU)** → benchmark **FP32 vs dynamic INT8** → per-frame video report.

**Private Artikate images/video are not in this repo yet** (awaiting their dataset link). See [`PRIVATE_DATA.md`](PRIVATE_DATA.md) for the drop-in checklist. Smoke metrics below only prove the pipeline runs.

## Hardware (this machine)

| Item | Value |
|------|--------|
| CPU | AMD Ryzen 5 5500U (12 threads) |
| RAM | ~7.1 GiB |
| GPU | None (no NVIDIA) — **CPU-only** |
| Inference | ONNX Runtime `CPUExecutionProvider` |
| Quantization | ORT **dynamic INT8** on MatMul/Gemm (not TensorRT PTQ) |

**Confidence those latency numbers hold elsewhere:** low for absolute ms. Relative FP32 vs INT8 trends may transfer; Jetson Orin + TensorRT will be much faster. Re-measure on the target device.

Set `YOLO_CONFIG_DIR` to a writable path if Ultralytics cannot write `~/.config` (this repo uses `.ultralytics_config/`).

## Loom / screen recording

Paste unlisted link here after recording the 5–8 min walkthrough on the **private** held-out clip:

```
LOOM_URL: <paste after recording>
```

Suggested narration: env → train/load weights → export ONNX → INT8 → `infer_video.py` → show CSV + latency summary → `benchmark.py` → worst failure cases.

## Quick start

```bash
cd Artikate_assignment
# system Python with deps, or:
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

export YOLO_CONFIG_DIR="$PWD/.ultralytics_config"

# 1) Smoke dataset (synthetic) — or follow PRIVATE_DATA.md
python3 scripts/prepare_smoke_data.py

# 2) Train (CPU defaults: batch=2, workers=0)
python3 scripts/train.py --data configs/data.yaml --epochs 30 --batch 2 --device cpu --workers 0

# 3) Export + quantize
python3 scripts/export_onnx.py --weights weights/best.pt
python3 scripts/quantize_int8.py --onnx weights/best.onnx

# 4) Val metrics
python3 scripts/eval_val.py --weights weights/best.pt --data configs/data.yaml

# 5) Held-out video (smoke path; swap for Artikate clip)
python3 scripts/infer_video.py \
  --model weights/best.onnx \
  --video data/smoke/video/heldout_smoke.mp4 \
  --save-vis

# 6) FP32 vs INT8 latency
python3 scripts/benchmark.py \
  --fp32 weights/best.onnx \
  --int8 weights/best_int8.onnx \
  --video data/smoke/video/heldout_smoke.mp4
```

### Private dataset drop-in

Full checklist: [`PRIVATE_DATA.md`](PRIVATE_DATA.md).

1. Unpack into `data/artikate/` (YOLO layout).
2. Fill `configs/data_artikate.yaml.example` (class names / `nc`) and use it as `--data`.
3. Place held-out video at `data/artikate/video/heldout.mp4`.
4. Re-run train → export → infer → benchmark → replace tables below and `results/FAILURE_CASES.md`.

## Measured numbers

### A) Smoke dry-run (synthetic — NOT submission numbers)

| Metric | FP32 ONNX | INT8 dynamic (MatMul/Gemm) |
|--------|-----------|----------------------------|
| Median latency (ms) | 120.4 | 96.1 |
| p95 latency (ms) | 169.6 | 111.8 |
| Val precision (PT) | 0.799 | n/a |
| Val recall (PT) | 0.918 | n/a |
| mAP50 (Ultralytics, PT) | 0.944 | n/a |

Video (45 frames, FP32 ORT): median infer ~105 ms (`results/video_latency_summary.txt`).

### B) Private Artikate set (fill after link arrives)

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
- `PRIVATE_DATA.md` — drop-in steps when the dataset link arrives
- `scripts/` — train, export, quantize, infer, benchmark, eval
- `src/` — letterbox + ORT detector
- `section3/` — silent-bug lab (swap if Artikate ships a different repo)
- `tests/` — letterbox + Section 3 regression

## GitHub

Public repo: https://github.com/shivenswaroop/artikate-cv-takehome

Loom still needs your screen recording after the private clip arrives (`LOOM_URL` above).
