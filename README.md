# Artikate Studio — Senior CV / ML Take-Home

End-to-end industrial detection pipeline: fine-tune **YOLOv8n** → export **ONNX** → run **ONNX Runtime (CPU)** → benchmark **FP32 vs dynamic INT8** → per-frame video report.

Written answers: [`ANSWERS.md`](ANSWERS.md) (Sections 1, 3, 4).

## Why this dataset (proxy for a client hand-off)

Artikate clarified that candidates should source a public annotated stand-in (~100–200 images). I used a **stratified 150-image subset of NEU-DET** (hot-rolled steel surface defects):

| Item | Choice |
|------|--------|
| Source | [NEU-DET](https://github.com/KingRedMan/NEU-DET_Yolo) YOLO labels (Northeastern University steel defects) |
| Scale | 120 train + 30 val (matches the assignment’s ~150-image pack) |
| Classes | `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` |
| Held-out video | ~30s @ 15fps from **unused official test** stills (`data/proxy_neu/video/heldout.mp4`) |

**Why this is a reasonable proxy for an industrial-inspection client set**

1. **Domain match:** Real mill-surface defects under industrial imaging — closer to Artikate’s inspection work than COCO “person/car”.
2. **Annotation quality:** Established academic benchmark with box labels already in YOLO format (no synthetic boxes).
3. **Difficulty:** Multi-class, low-contrast, texture-heavy defects — exposes failure modes (crazing misses, scratch over-fire) rather than toy shapes.
4. **Honest limits:** Stills are small (~200²) and grayscale-ish; the held-out “video” is a slideshow of test stills, not a line-scan camera. I call that out rather than pretending it’s RTSP footage. Rebuild anytime with `python scripts/prepare_neu_proxy.py`.

Details: [`data/proxy_neu/DATASET.md`](data/proxy_neu/DATASET.md).

## Hardware (this machine)

| Item | Value |
|------|--------|
| CPU | AMD Ryzen 5 5500U (12 threads) |
| RAM | ~7.1 GiB |
| GPU | None (no NVIDIA) — **CPU-only** |
| Inference | ONNX Runtime `CPUExecutionProvider` |
| Quantization | ORT **dynamic INT8** on MatMul/Gemm (not TensorRT PTQ) |

**Confidence latency numbers hold elsewhere:** low for absolute ms. Relative FP32 vs INT8 trends may transfer; Jetson Orin + TensorRT will be much faster. Re-measure on the target device.

```bash
export YOLO_CONFIG_DIR="$PWD/.ultralytics_config"
```

## Run on free Colab GPU

Open this notebook (Runtime → GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivenswaroop/artikate-cv-takehome/blob/main/notebooks/artikate_colab.ipynb)

Direct link: https://colab.research.google.com/github/shivenswaroop/artikate-cv-takehome/blob/main/notebooks/artikate_colab.ipynb

It clones this repo, trains on GPU, exports ONNX, runs video infer + FP32/INT8 benchmark, and zips artifacts for download.

## Loom / screen recording

```
LOOM_URL: <paste after recording on held-out clip>
```

Narrate: dataset choice → train → export → INT8 → `infer_video.py` → CSV/latency → FP32 vs INT8 → failure cases.

## Quick start

```bash
cd Artikate_assignment
source .venv/bin/activate   # or: python3 -m venv .venv && pip install -r requirements.txt
export YOLO_CONFIG_DIR="$PWD/.ultralytics_config"

# Rebuild proxy data (if needed)
python scripts/prepare_neu_proxy.py

python scripts/train.py --data configs/data.yaml --epochs 40 --batch 2 --device cpu --workers 0 --name neu_proxy_v1
python scripts/export_onnx.py --weights weights/best.pt
python scripts/quantize_int8.py
python scripts/eval_val.py --weights weights/best.pt --data configs/data.yaml
python scripts/infer_video.py --model weights/best.onnx --video data/proxy_neu/video/heldout.mp4
python scripts/benchmark.py --fp32 weights/best.onnx --int8 weights/best_int8.onnx --video data/proxy_neu/video/heldout.mp4
```

## Measured numbers (NEU-DET proxy — submission)

| Metric | FP32 ONNX | INT8 dynamic (MatMul/Gemm) |
|--------|-----------|----------------------------|
| Median latency (ms) | **93.6** | **89.4** |
| p95 latency (ms) | **100.7** | **103.5** |
| Val precision (PT) | **0.560** | n/a |
| Val recall (PT) | **0.477** | n/a |
| mAP50 (Ultralytics, PT) | **0.551** | n/a |
| mAP50-95 | **0.307** | n/a |

Video (90 frames sampled, FP32 ORT): median infer **~83 ms** (`results/video_latency_summary.txt`).  
Per-frame boxes/conf/latency: `results/video_detections.csv`.  
Worst cases: [`results/FAILURE_CASES.md`](results/FAILURE_CASES.md).

INT8 gain is small here because quantization is **weight-only MatMul/Gemm** on CPU ORT (full Conv INT8 emitted unsupported `ConvInteger` on this stack). On Orin TensorRT PTQ the tradeoff would differ — I’d re-benchmark there before claiming speedups.

## Repo map

- `ANSWERS.md` — Sections 1 (incl. FP16 diagnostic table), 3, 4
- `data/proxy_neu/` — stand-in client images/labels
- `scripts/` — prepare, train, export, quantize, infer, benchmark, eval
- `section3/` — silent-bug lab + regression test
- `tests/` — letterbox + Section 3 tests

## GitHub

https://github.com/shivenswaroop/artikate-cv-takehome
