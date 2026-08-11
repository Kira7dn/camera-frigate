"""Shared Face preprocessing and producer-owned evidence contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from frigate.util.image import area
from frigate.util.passage_trace import passage_evidence

BBox = tuple[int, int, int, int]
MAX_DETECTION_HEIGHT = 1080


@dataclass(frozen=True, slots=True)
class PreparedFaceAttempt:
    bgr: np.ndarray
    detector_box: BBox
    effective_crop_box: BBox
    crop: np.ndarray


def detect_largest_face(
    detector: cv2.FaceDetectorYN | None,
    image: np.ndarray,
    threshold: float,
) -> BBox | None:
    """Run the exact synchronous FaceDetectorYN bbox conversion."""
    if detector is None or image.size == 0:
        return None
    if image.shape[0] > MAX_DETECTION_HEIGHT:
        scale = MAX_DETECTION_HEIGHT / image.shape[0]
        image = cv2.resize(image, (int(scale * image.shape[1]), MAX_DETECTION_HEIGHT))
    else:
        scale = 1.0
    detector.setInputSize((image.shape[1], image.shape[0]))
    detected = detector.detect(image)
    if detected is None or detected[1] is None:
        return None
    candidates: list[BBox] = []
    for value in detected[1]:
        if value[-1] < threshold:
            continue
        raw_bbox = value[0:4].astype(np.uint16)
        x = int(max(raw_bbox[0], 0) / scale)
        y = int(max(raw_bbox[1], 0) / scale)
        width = int(raw_bbox[2] / scale)
        height = int(raw_bbox[3] / scale)
        candidates.append((x, y, x + width, y + height))
    return max(candidates, key=area) if candidates else None


def clamp_box(box: BBox, image: np.ndarray) -> BBox:
    height, width = image.shape[:2]
    return (
        max(0, min(width, box[0])),
        max(0, min(height, box[1])),
        max(0, min(width, box[2])),
        max(0, min(height, box[3])),
    )


def prepare_face_attempt(
    frame: np.ndarray,
    supplied_bgr: np.ndarray | None,
    person_box: BBox,
    attributes: Iterable[dict[str, Any]],
    *,
    requires_face_detection: bool,
    detection_threshold: float,
    min_area: int,
    detect_face: Callable[[np.ndarray, float], BBox | None],
) -> tuple[PreparedFaceAttempt | None, str]:
    """Select and crop the exact image passed to the Face classifier."""
    bgr = (
        supplied_bgr
        if supplied_bgr is not None
        else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    )
    if requires_face_detection:
        left, top, right, bottom = person_box
        detected = detect_face(bgr[top:bottom, left:right], detection_threshold)
        if detected is None:
            return None, "no_face"
        detector_box = (
            detected[0] + left,
            detected[1] + top,
            detected[2] + left,
            detected[3] + top,
        )
    else:
        faces = [
            item
            for item in attributes
            if item.get("label") == "face" and item.get("box")
        ]
        if not faces:
            return None, "no_face"
        best = max(faces, key=lambda item: float(item.get("score", 0.0)))
        values = best["box"]
        if len(values) != 4:
            raise ValueError("face bbox must contain exactly four coordinates")
        detector_box = tuple(int(value) for value in values)
    if area(detector_box) < min_area:
        return None, "too_small"
    effective_box = clamp_box(detector_box, bgr)
    crop = bgr[
        effective_box[1] : effective_box[3],
        effective_box[0] : effective_box[2],
    ]
    if crop.size == 0:
        return None, "empty_crop"
    return PreparedFaceAttempt(bgr, detector_box, effective_box, crop), "accepted"


def render_recognition_boxes(
    image: np.ndarray,
    *,
    object_box: BBox | None = None,
    detail_box: BBox | None = None,
) -> np.ndarray:
    """Render review evidence at the producer from the exact pipeline boxes."""
    rendered = image.copy()
    if object_box is not None:
        cv2.rectangle(
            rendered,
            (object_box[0], object_box[1]),
            (object_box[2], object_box[3]),
            (0, 0, 255),
            3,
        )
    if detail_box is not None:
        cv2.rectangle(
            rendered,
            (detail_box[0], detail_box[1]),
            (detail_box[2], detail_box[3]),
            (0, 255, 0),
            2,
        )
    return rendered


def emit_face_attempt_evidence(
    attempt: PreparedFaceAttempt,
    *,
    evidence_id: str,
    camera: str,
    frame_time: float,
    track_id: str,
    trace_id: str,
    person_box: BBox,
    raw_identity: str,
    raw_score: float,
) -> None:
    """Emit one identical Face evidence bundle for local and external runtimes."""
    common = {
        "evidence_id": evidence_id,
        "camera": camera,
        "frame_time": frame_time,
        "track_id": track_id,
        "trace_id": trace_id,
        "pipeline": "face",
        "object_box": list(person_box),
        "detail_box": list(attempt.detector_box),
        "effective_crop_box": list(attempt.effective_crop_box),
        "bbox_format": "xyxy_pixels",
        "bbox_coordinate_space": "recognition_attempt",
        "raw_identity": raw_identity,
        "raw_score": raw_score,
    }
    passage_evidence("recognition_attempt", image=attempt.bgr, **common)
    passage_evidence(
        "recognition_attempt_bbox",
        image=render_recognition_boxes(
            attempt.bgr,
            object_box=person_box,
            detail_box=attempt.effective_crop_box,
        ),
        **common,
    )
    passage_evidence("face_crop", image=attempt.crop, **common)
