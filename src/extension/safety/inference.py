"""ONNX inference adapter for the internal smoking mock model."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from .config import ModelConfig


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: tuple[float, float, float, float] | None
    observed_at: float


class SafetyModel(Protocol):
    def infer(self, frame: np.ndarray, observed_at: float) -> list[Detection]: ...


def _nms(boxes: list[tuple[float, float, float, float]], scores: list[float]) -> list[int]:
    if not boxes:
        return []
    xywh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=0.0, nms_threshold=0.45)
    return [int(index[0] if isinstance(index, list | tuple | np.ndarray) else index) for index in indices]


class OnnxSafetyModel:
    """Small YOLO export adapter; currently maps the single cigarette class."""

    def __init__(self, session: ort.InferenceSession, input_name: str, threshold: float) -> None:
        self._session = session
        self._input_name = input_name
        self._threshold = threshold
        shape = session.get_inputs()[0].shape
        if list(shape) != [1, 3, 640, 640]:
            raise ValueError(f"unsupported Safety model input shape: {shape}")
        output_shape = session.get_outputs()[0].shape
        if list(output_shape) != [1, 5, 8400]:
            raise ValueError(f"unsupported smoking model output shape: {output_shape}")

    @classmethod
    def build(cls, config: ModelConfig, threshold: float = 0.1) -> OnnxSafetyModel:
        session = ort.InferenceSession(str(config.path), providers=list(config.providers))
        return cls(session, session.get_inputs()[0].name, threshold)

    @property
    def active_provider(self) -> str:
        return self._session.get_providers()[0]

    def infer(self, frame: np.ndarray, observed_at: float | None = None) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        observed_at = time() if observed_at is None else observed_at
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        output = np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        rows = output[0].T
        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        for cx, cy, box_width, box_height, score in rows:
            score = float(score)
            if not np.isfinite(score) or score < self._threshold:
                continue
            values = np.asarray([cx, cy, box_width, box_height], dtype=np.float32)
            if not np.isfinite(values).all() or box_width <= 0 or box_height <= 0:
                continue
            x1 = max(0.0, min(1.0, float((cx - box_width / 2) / 640)))
            y1 = max(0.0, min(1.0, float((cy - box_height / 2) / 640)))
            x2 = max(0.0, min(1.0, float((cx + box_width / 2) / 640)))
            y2 = max(0.0, min(1.0, float((cy + box_height / 2) / 640)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
        return [
            Detection("smoking", scores[index], boxes[index], observed_at)
            for index in _nms(boxes, scores)
        ]
