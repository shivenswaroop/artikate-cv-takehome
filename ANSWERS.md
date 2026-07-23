# Artikate Studio — Take-Home Assignment Answers

**Candidate hardware (Section 2):** AMD Ryzen 5 5500U, 7.1 GiB RAM, no discrete NVIDIA GPU. Training and inference via PyTorch CPU + ONNX Runtime CPU. TensorRT / Jetson numbers are estimated from public Orin benchmarks and flagged as unverified on Orin hardware.

---

## Section 1 — Diagnose a Failing CV Pipeline

### Scenario A — Accuracy drops after quantization

**Symptom:** YOLOv8 defect detector 0.91 mAP@0.5 (PyTorch FP32) → 0.58 after ONNX + INT8 on Jetson Orin. Same val set. No architecture or training-data change.

#### What I'd check first (order)

1. **FP32 ONNX / TensorRT FP32 engine mAP** on the same val set.  
   Isolates export, opset, NMS-in-graph, and preprocess parity from *any* reduced-precision path.
2. **FP16 checkpoint / TensorRT FP16 engine mAP next** (before INT8).  
   Artikate did not provide an FP16 number — that is intentional. Checking FP16 yourself is the cleanest way to separate “export/preprocess broke” from “INT8 quantization broke.” See outcome table below.
3. **Preprocess parity** between the PyTorch eval path and the ORT/TensorRT path: letterbox vs stretch, color order (RGB/BGR), normalize scale (`/255` vs ImageNet stats), pad color, `imgsz`, dynamic vs static shapes.
4. **INT8 calibration set**: size, class/background coverage, whether it was random ImageNet crops vs real line imagery, and whether calibration ran with the *same* preprocess as inference.
5. **Where NMS / decode lives**: postprocess in Python vs baked into the ONNX graph; compare box decode (xywh→xyxy, stride/anchor assumptions) FP32 vs INT8.
6. **Quantization recipe**: PTQ vs QAT, per-tensor vs per-channel weights, which layers were forced INT8 (esp. detection head / SiLU-heavy blocks), TensorRT vs ORT calibrator.
7. **Eval harness bugs**: confidence threshold, IoU matching, class mapping, or different NMS IoU between the two runs that coincidentally look like “quantization hurt.”

#### FP16 diagnostic — hypotheses for each outcome

Hold the exported graph and preprocess fixed; only change precision to FP16 (TensorRT FP16 engine, or ONNX Runtime with FP16 where supported). Interpret against the known FP32 PyTorch baseline (0.91) and the reported INT8 result (0.58):

| FP16 mAP (hypothesis) | What it implies | Next action |
|-----------------------|-----------------|-------------|
| **≈ FP32 (~0.88–0.91)** | Export + preprocess are healthy. The collapse is **INT8-specific** (bad calibration, sensitive layers, or INT8 kernel/NMS quirks). | Recalibrate INT8 on line imagery; try mixed precision (FP16 head / INT8 backbone); consider QAT. |
| **Already collapsed (~0.55–0.65, similar to INT8)** | Failure is **upstream of INT8** — export, TRT parse/fuse, preprocess mismatch, or NMS/decode baked wrong. INT8 is a red herring. | Diff preprocess tensors; compare FP32 ONNX vs PyTorch box-for-box; rebuild engine without plugins that alter decode. |
| **Mild drop (~0.80–0.87)** | Export mostly OK; some numerical sensitivity. INT8 then amplifies that into the 0.58 cliff. | Fix any mild preprocess drift first, then INT8 with better calibration / partial FP16. |
| **FP16 OK but INT8 fails only on Orin TRT (ORT INT8 OK)** | Device/engine-specific (calibrator, layer fusion, DLA). | Rebuild TRT with explicit layer precision constraints; verify calibrator cache matches preprocess. |

I would not wait for a vendor-provided FP16 number — running that one eval is cheap and partitions the search space before touching calibration.

#### Three independent root causes + distinguishing tests

| # | Root cause | Distinguishing test |
|---|------------|---------------------|
| 1 | **Export / preprocess mismatch** (not quantization): letterbox or normalize differs in the deployed path, so boxes/scores are wrong even before INT8. | FP32 ONNX/TRT mAP + **FP16** mAP. If either already collapses, INT8 is not the root cause. |
| 2 | **Bad or non-representative INT8 calibration** (clean lab images / too few samples / wrong distribution). | FP16 ≈ FP32, but INT8 collapses; recalibrate with **≥200–500 real line frames**. Sharp recovery ⇒ calibration. |
| 3 | **Quantization-sensitive layers** (detection head / late neck activations with heavy outliers). | FP16 ≈ FP32; INT8 hybrid with head left in FP16 recovers most mAP ⇒ sensitivity, not total export failure. |

(Other common causes exist—e.g. TensorRT falling back / differently fusing NMS—but the three above each alone can produce ~0.91→0.58.)

#### Fix and pre-redeploy validation

1. Restore **FP32 ONNX** and confirm **FP16** ≥ baseline−ε on the client val set; freeze that preprocess + export script.
2. Recalibrate INT8 on **production-representative** crops; if still short, use **mixed precision** (INT8 backbone, FP16 head) or short **QAT**.
3. Gate redeploy: same val protocol, report mAP@0.5 / mAP@0.5:0.95, per-class AP, and a fixed confidence operating point (precision/recall on the line’s defect classes).  
4. **Shadow run** on Orin with recorded line video: compare FP16 vs INT8 detections frame-by-frame; only cut over if drop is within the client’s tolerance (and latency still meets budget).

---

### Scenario B — Bounding boxes drift on one camera feed only

**Symptom:** Same model on 12 RTSP feeds; 11 correct. One feed: boxes **consistently** offset; offset **larger near edges than center**.

#### What the offset pattern tells you

Systematic + radially/edge-growing error is almost never “the model randomly failing.” It points to a **geometric / coordinate-mapping** bug between that feed’s pixel space and the space where boxes are drawn:

- Anisotropic scale (stretch instead of letterbox, or wrong aspect assumption)
- Incorrect inverse of letterbox pad/scale (pad applied on one axis only, or pad subtracted wrong)
- Camera/stream delivering a different resolution or pixel aspect than the pipeline assumes
- Undistortion / ROI crop applied in capture but not accounted for when mapping boxes back

Random NMS errors or weight corruption would not be stable and edge-dependent on a single feed.

#### Preprocess / camera-config checks for that feed only

- Decoded frame `H×W` vs the other 11 (and vs config)
- RTSP / camera output: crop, digital zoom, OSD, rotation/mirror, `force_style` style scaling in FFmpeg
- Letterbox vs resize-stretch flag for that camera ID
- Assumed `imgsz` and pad (top/left) used when mapping boxes back to original
- ROI or “active area” metadata unique to that camera
- Pixel aspect ratio / anamorphic flags (rare but matches edge-heavier distortion)
- Whether undistort maps are loaded for that serial and applied only on ingest

#### Root-cause hypothesis and remote confirmation

**Hypothesis:** That feed’s frames are **resized with a different aspect ratio** (or pad/scale inverted wrong) relative to training and the other cameras—so the affine map from network space → display space is wrong, with error growing toward the borders.

**Confirm without physical access:**

1. Pull one keyframe from the bad RTSP URL and one from a good camera (`ffmpeg` / OpenCV).
2. Log preprocess: original shape, resized shape, pad `(dw, dh)`, gain `r`.
3. Run the model; overlay boxes in (a) network input space and (b) after `scale_boxes` to original.  
4. Draw a known grid or use a calibration board if available; if boxes align on the letterboxed tensor but drift on the full frame, the **inverse letterbox** for that resolution is wrong.  
5. Force the bad camera through the **same decode size + letterbox path** as a good camera (software resize to match); if drift disappears, the bug is config/geometry, not the sensor “seeing wrong.”

---

### Scenario C — Model degrades over three months, no redeploys

**Symptom:** Conveyor counting 97% → 84% over ~3 months; no code/model changes; client reports no physical change.

#### Plausible causes + evidence

1. **Lighting / seasonal ambient change** (sun angle, skylights, warehouse lighting schedule).  
   - Confirm: compare mean luminance / histogram of production frames month-0 vs month-3; EXIF or camera AE logs if available.  
   - Rule out: histograms and AE settings unchanged.

2. **Optics / sensor gradual degradation** (dust/oil on lens, focus creep, IR-cut aging, auto-exposure hunting).  
   - Confirm: sharpness (Laplacian variance) trend down; increased blur on edges; more AE flicker in logs.  
   - Rule out: sharpness stable; cleaning lens restores accuracy → optics soiling.

3. **Product / SKU mix shift** (new packaging, sizes, colors) — “setup unchanged” but inventory changed.  
   - Confirm: class-conditional error rises on new SKUs; embedding cluster of frames drifts from baseline set; operations confirms new SKUs.  
   - Rule out: SKU mix reports identical; confusion matrix shape unchanged across SKUs.

(Related: conveyor speed/spacing change affecting double-count/miss without anyone calling it a “setup change.”)

#### Lightweight monitoring that would catch this in ~2 weeks

Track a **daily baseline comparison** on unlabeled production frames (no need for full GT):

- Mean / p50 / p10 of max detection confidence  
- Predicted count per N frames vs a short rolling baseline  
- Population Stability Index (PSI) on confidence histogram and on a cheap embedding (e.g. pooled backbone features or even color histogram) vs week-1 reference  

Alert if PSI > threshold or mean confidence drops >X% for 3 consecutive days. That surfaces distribution shift within days–weeks, before a 13-point accuracy slide goes unnoticed for a quarter.

---

## Section 3 — Find the Silent Bug

Artikate’s external buggy repo was not attached to this workspace. This submission includes a self-contained silent-bug lab under [`section3/`](section3/) that exhibits the same failure *class* the prompt describes (plausible boxes, quietly wrong coordinates). If Artikate later provides a different repo, the same process applies there; replace this section’s file/line with theirs.

### Bug location

- **File:** [`section3/postprocess.py`](section3/postprocess.py)  
- **Buggy line (before fix):** `out[:, [1, 3]] += pad_h` inside `boxes_from_network_output`  
- **What it did wrong:** When undoing letterbox, horizontal pad was subtracted correctly (`-= pad_w`) but **vertical pad was added** instead of subtracted. Boxes were therefore shifted vertically in original-image space by roughly `2 * pad_h / ratio` before clipping.

### Why it looked fine most of the time

Letterbox pad on a 640×480 → 640×640 path is only ~40 px per side. For objects near the vertical mid-band, the drift is a modest fraction of the box height, so a casual overlay still “looks like a detection on the object.” Clipping to image bounds further hides bad coordinates near edges. Roughly when objects sit mid-frame (common in demos), nothing screams broken; when objects sit high/low in the frame—or when a client consumes exact `y1/y2` for measurement—the error shows up (~1 in 5 frames depending on object placement).

### Fix + test

- **Fix:** use `out[:, [1, 3]] -= pad_h` (symmetric with x). See current [`section3/postprocess.py`](section3/postprocess.py).  
- **Test:** [`tests/test_section3_regression.py`](tests/test_section3_regression.py) asserts a content-centered letterbox box maps to `cy ≈ 240` on a 480-high image, and that the old `+= pad_h` path would fail that check.

### Testing-process gap

The original gap is relying on “boxes draw on something” visual smoke tests instead of **coordinate-space property tests** (known synthetic geometry → exact inverse letterbox). Closing it: add unit tests for preprocess/postprocess round-trips on fixed shapes, plus a small fixture image with GT boxes in original pixels compared after full preprocess→fake head→postprocess—without requiring a trained model.

---

## Section 4 — Edge & Air-Gapped Deployment Design

**Client Alpha:** 8× 1080p @ 15 fps → one Jetson AGX Orin 64GB, air-gapped, e2e latency &lt; 200 ms per frame, zero cloud.

### Aggregate throughput (arithmetic)

\[
8 \text{ cameras} \times 15 \text{ fps} = 120 \text{ frames/s aggregate}
\]

Per-camera period: \(1/15 \approx 66.7\) ms between frames. The **200 ms e2e** budget is looser than the inter-frame period, so the binding constraint is usually **sustaining 120 FPS aggregate** (or accepting bounded queueing). If each frame must finish in &lt;200 ms including queue wait, average service rate must stay ≥120 FPS with headroom for jitter.

Decode cost (rough): 8×1080p15 H.264 is modest for Orin NVDEC; the detector dominates.

### Model / precision target

**Primary target:** **YOLOv8n (or YOLOv8s if accuracy requires it) exported to TensorRT INT8**, input ~640, NMS in TRT or a fast CUDA/CPU postprocess.

**Reasoning:**

- 120 FPS aggregate on one Orin is aggressive for larger YOLO variants in FP32.  
- Public Orin AGX reports for YOLOv8n TensorRT are commonly in the **hundreds of FPS** range at 640 in FP16/INT8 for a *single* stream — but **multi-stream + decode + copy + postprocess** eats a large fraction of that.  
- **I have not benchmarked this exact stack on Orin hardware.** On my laptop I only have CPU ORT numbers (see README). Before committing to a client SLA I would measure: TRT YOLOv8n INT8 vs FP16, 1-stream vs 8-stream, with real 1080p letterbox, on the AGX 64GB.

**Working plan:** start YOLOv8n INT8 multi-stream; if mAP drop &gt; tolerance, YOLOv8n FP16 or YOLOv8s INT8; if still short on FPS, reduce `imgsz` to 480 for some cameras, or dual-model (tiny trigger + confirm), or add a second Orin — I’d rather flag the second Orin than invent headroom.

**Latency math check (illustrative, verify on device):**

- Budget per frame e2e: 200 ms  
- If we process streams concurrently with effective throughput \(T\) FPS aggregate, utilization ≈ \(120 / T\).  
- Need \(T \gtrsim 150\)–180 FPS aggregate after decode/post to keep queues small (25–50% headroom).  
- Uncertain: exact TRT FPS for their TRT version / power mode / clocks — **benchmark before commit.**

### Air-gapped retraining loop

1. **Flagging on device:** Operator UI on the Orin (or a local HMI) lets QC mark FP/FN on stills or short clips; store `(image, box, label, verdict, model_version, camera_id, timestamp)` on local disk.  
2. **Export package:** Nightly or on-demand encrypted USB/SSD bundle (images + JSONL labels + `model_version` + hash manifest). No internet.  
3. **Offline retrain station:** Disconnected training box receives the USB, merges into curated dataset, trains, evaluates on a **held-out golden set** that never goes to the floor for training.  
4. **Return package:** New weights + ONNX/TRT engine build scripts + eval report + `model_card.json` (metrics, git SHA, data snapshot hash).  
5. **On-Orin validation before swap:**  
   - Build TRT engine locally on that Orin (engines are device-specific).  
   - Run golden-set eval + **shadow mode**: new model scores live frames in parallel; compare disagreement rate and defect KPI for a soak window (e.g. 1 shift).  
   - Only then atomic activate.

### Rollback and regression detection

- **Deployment:** symlink / `current` → `releases/<version>/`; previous release kept on disk. One command (or UI) retargets `current` and reloads the inference process.  
- **Detection:**  
  - Shadow dual-run disagreement spike  
  - Rolling precision proxy (operator flags per hour)  
  - Count/defect rate vs 7-day baseline outside control limits  
- **Speed:** automated alert within **minutes** on crash/latency; **within one shift** on accuracy proxies; hard rollback &lt; **5 minutes** (reload previous engine).

### Explicit uncertainties

- Exact sustained multi-stream FPS for YOLOv8n/s INT8 on AGX Orin 64GB with their JetPack/TRT — **must measure**.  
- Whether INT8 calibration holds for their defect texture — may need FP16.  
- USB sneakernet latency vs required model update cadence — process design more than ML.  
- Whether 200 ms is measured camera-capture-to-box-available or decode-to-box; that changes queue design.
