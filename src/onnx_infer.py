"""ONNX Runtime YOLOv8 detection wrapper (CPU-friendly)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.letterbox import letterbox, scale_boxes_to_original


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS. boxes: (N, 4) xyxy, scores: (N,)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thres]
    return keep


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(
        self, bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        original_shape = bgr.shape[:2]
        lb, ratio, pad = letterbox(bgr, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None]
        return tensor, ratio, pad, original_shape

    def _parse_output(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse Ultralytics YOLOv8 ONNX output to boxes/scores/classes in letterbox space."""
        # Typical shapes: (1, 4+nc, N) or (1, N, 4+nc)
        pred = output[0]
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 512:
            # (4+nc, N) → (N, 4+nc)
            pred = pred.T
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        if cls_scores.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)
        mask = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if len(scores) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        keep = nms_xyxy(xyxy, scores, self.iou_thres)
        return xyxy[keep], scores[keep], class_ids[keep].astype(np.int32)

    def predict(self, bgr: np.ndarray) -> dict[str, Any]:
        tensor, ratio, pad, original_shape = self.preprocess(bgr)
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0
        boxes, scores, classes = self._parse_output(outputs[0])
        boxes = scale_boxes_to_original(boxes, ratio, pad, original_shape)
        return {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "latency_ms": latency_ms,
            "original_shape": original_shape,
        }
