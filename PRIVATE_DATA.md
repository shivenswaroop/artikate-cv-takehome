# Private Artikate dataset — drop-in checklist

You do **not** have the private link yet. When Artikate sends it, do the following and re-run the pipeline. Do not invent metrics beforehand.

## 1. Unpack

Expected layout (YOLO format):

```
data/artikate/
  images/train/     # ~train split images
  images/val/       # validation images
  labels/train/     # YOLO txt labels, same stems
  labels/val/
  video/heldout.mp4 # ~30s held-out clip
```

If they ship COCO/JSON or a zip with a different layout, convert once (keep a small `scripts/convert_*.py` commit) then keep this layout.

## 2. Point configs at private data

```bash
cp configs/data_artikate.yaml.example configs/data.yaml
# Edit names: / nc: to match their classes
```

Or: `python scripts/train.py --data configs/data_artikate.yaml` after filling the example file.

## 3. Re-run (CPU laptop defaults)

```bash
source .venv_run/bin/activate   # or .venv once deps are installed
python scripts/train.py --data configs/data.yaml --epochs 50 --batch 2 --device cpu
python scripts/export_onnx.py
python scripts/quantize_int8.py
python scripts/eval_val.py --weights weights/best.pt --data configs/data.yaml
python scripts/infer_video.py --model weights/best.onnx --video data/artikate/video/heldout.mp4 --save-vis
python scripts/benchmark.py --fp32 weights/best.onnx --int8 weights/best_int8.onnx --video data/artikate/video/heldout.mp4
```

## 4. Fill submission artifacts

- [ ] `README.md` — measured latency / P/R table (replace smoke TBD)
- [ ] `results/val_metrics.json`, `results/benchmark.json`, `results/video_detections.csv`
- [ ] `results/FAILURE_CASES.md` — 2–3 worst held-out failures + hypotheses
- [ ] Loom link in `README.md`
- [ ] Incremental git commits during the private-data run (not one dump)

## 5. What stays smoke-only

`data/smoke/` is synthetic. Never report smoke mAP/latency as the assignment’s private-dataset numbers.
