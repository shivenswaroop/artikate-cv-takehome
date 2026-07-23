# Worst failure cases (held-out video)

Held-out clip: `data/proxy_neu/video/heldout.mp4` (assembled from unused NEU-DET **test** stills; not in train/val).  
Val reference (same model): mAP50 **0.55**, precision **0.56**, recall **0.48**. Weakest class on val: **crazing**.

## Case 1 — Crazing under-detection

- **Evidence:** Val AP for `crazing` is ~0.14 with recall ~0.08; video frames that are crazing-like often produce no box or a very low-confidence box.
- **What went wrong:** Fine, crack-like texture is easy to confuse with steel grain / sensor noise at 640 after upscaling from ~200×200 NEU stills.
- **Hypothesis:** Class is texture-dominated and under-represented in absolute pixel area; 120-image fine-tune + ImageNet-pretrained YOLO head is not enough for thin crack patterns. More crazing crops, stronger contrast augmentation, or a higher-res `imgsz` would help.

## Case 2 — Rolled-in scale false negatives / loose boxes

- **Evidence:** Val `rolled-in_scale` mAP50 ~0.39; on the held-out slideshow, scale patches are sometimes missed or boxed loosely overlapping neighboring texture.
- **What went wrong:** Diffuse, low-contrast defects without sharp edges — NMS + conf threshold drops them.
- **Hypothesis:** Appearance overlaps with `pitted_surface` / background mill scale; class confusion and soft boundaries. Multi-label soft targets or more diverse lighting would reduce misses.

## Case 3 — High-confidence wrong class on scratch-like linear marks

- **Evidence:** `scratches` recall is high (~0.89) but precision is low (~0.25) on val; video frames show confident scratch boxes on linear mill marks that are labeled as other classes (or background) in the original taxonomy.
- **What went wrong:** Model over-fires the scratch class on any elongated high-contrast streak.
- **Hypothesis:** Scratch features dominate the linear-edge prior of the detector; need harder negatives and class-balanced focal loss / lower conf at the operating point for counting use-cases.
