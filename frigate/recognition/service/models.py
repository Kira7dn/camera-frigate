"""Inference-only Face and LPR adapters for the recognition service."""

from __future__ import annotations

import datetime
import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.data_processing.common.face_pipeline import (
    detect_largest_face,
    emit_face_attempt_evidence,
    prepare_face_attempt,
)
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.data_processing.common.license_plate.model import LicensePlateModelRunner
from frigate.util.passage_trace import capture_passage_evidence

from ..contracts import RecognitionArtifact, RecognitionTask, TrackedObservation
from ..ports import ModelRecognition, RawRecognition
from .v1 import recognition_pb2 as pb

MAX_CAPTURE_BYTES = 8 * 1024 * 1024


class _ArtifactCollector:
    def __init__(self, run_id: str | None) -> None:
        self.run_id = run_id
        self.artifacts: list[RecognitionArtifact] = []
        self.total_bytes = 0

    def __call__(self, record: dict[str, Any], image: Any | None) -> None:
        image_jpeg = b""
        image_shape: tuple[int, ...] = ()
        image_sha256 = ""
        if image is not None and record.get("artifact_path"):
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            )
            if not ok:
                raise ValueError("jpeg_encode_failed")
            image_jpeg = encoded.tobytes()
            self.total_bytes += len(image_jpeg)
            if self.total_bytes > MAX_CAPTURE_BYTES:
                raise ValueError("evidence_payload_too_large")
            image_shape = tuple(int(value) for value in np.asarray(image).shape)
            image_sha256 = hashlib.sha256(image_jpeg).hexdigest()

        standard = {
            "sequence",
            "stage",
            "pipeline",
            "trace_id",
            "evidence_id",
            "camera",
            "frame_time",
            "track_id",
            "image_index",
        }
        metadata = {key: value for key, value in record.items() if key not in standard}
        metadata["run_id"] = self.run_id
        self.artifacts.append(
            RecognitionArtifact(
                sequence=int(record["sequence"]),
                stage=str(record["stage"]),
                pipeline=str(record["pipeline"]),
                trace_id=str(record["trace_id"]),
                evidence_id=str(record["evidence_id"]),
                camera=str(record["camera"]),
                frame_time=(
                    float(record["frame_time"])
                    if record.get("frame_time") is not None
                    else None
                ),
                track_id=(
                    str(record["track_id"])
                    if record.get("track_id") is not None
                    else None
                ),
                image_index=(
                    int(record["image_index"])
                    if record.get("image_index") is not None
                    else None
                ),
                metadata=metadata,
                image_jpeg=image_jpeg,
                image_shape=image_shape,
                image_sha256=image_sha256,
            )
        )


class _Value:
    def __init__(self) -> None:
        self.value = 0.0


class ServiceMetrics:
    """Minimal metric sink required by the existing model implementations."""

    def __getattr__(self, name: str) -> _Value:
        value = _Value()
        setattr(self, name, value)
        return value


class _Requestor:
    def send_data(self, topic: str, payload: Any = None) -> None:
        return None


class FaceRecognitionModel:
    def __init__(self, config: FrigateConfig) -> None:
        self.config = config
        self.face_config = config.face_recognition
        self.face_detector: cv2.FaceDetectorYN | None = None
        self.requires_face_detection = "face" not in config.objects.all_objects
        self.face_counters: Counter[str] = Counter()
        self._build_detector()
        self.recognizer: FaceRecognizer
        if self.face_config.model_size == "small":
            self.recognizer = FaceNetRecognizer(config)
        else:
            self.recognizer = ArcFaceRecognizer(config)
        self.recognizer.build()

    def recognize(
        self, observation: TrackedObservation, frame: np.ndarray
    ) -> RawRecognition | ModelRecognition | None:
        camera = observation.key.camera_id
        person_box = observation.object_bbox
        attempt, reason = prepare_face_attempt(
            frame,
            None,
            person_box,
            observation.attributes.get("current_attributes", ()),
            requires_face_detection=self.requires_face_detection,
            detection_threshold=self.face_config.detection_threshold,
            min_area=self.config.cameras[camera].face_recognition.min_area,
            detect_face=self._detect_face,
        )
        if attempt is None:
            self.face_counters[reason] += 1
            return None
        result = self.recognizer.classify(attempt.crop)
        if result is None:
            return None
        name, score = result
        if score <= self.face_config.unknown_score:
            name = "unknown"
        raw = RawRecognition(
            name,
            float(score),
            detail_bbox=attempt.detector_box,
            area=int(attempt.crop.shape[0] * attempt.crop.shape[1]),
        )
        capture = observation.evidence_capture
        if capture is None:
            return raw
        collector = _ArtifactCollector(capture.run_id)
        with capture_passage_evidence(collector):
            emit_face_attempt_evidence(
                attempt,
                evidence_id=capture.evidence_id,
                camera=camera,
                frame_time=observation.frame_time,
                track_id=observation.key.track_id,
                trace_id=capture.trace_id,
                person_box=person_box,
                raw_identity=raw.value or "unknown",
                raw_score=raw.score,
            )
        return ModelRecognition(raw, tuple(collector.artifacts))

    async def manage(self, request: pb.FaceLibraryRequest) -> pb.FaceLibraryResponse:
        if request.operation == pb.FACE_LIBRARY_OPERATION_CLEAR:
            self.recognizer.clear()
            return pb.FaceLibraryResponse(
                success=True, message="Face classifier cleared"
            )
        image = cv2.imdecode(
            np.frombuffer(request.image, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            return pb.FaceLibraryResponse(success=False, message="Invalid face image")
        if request.operation == pb.FACE_LIBRARY_OPERATION_REGISTER:
            if not request.label:
                return pb.FaceLibraryResponse(
                    success=False, message="No face was detected"
                )
            if request.logical_name == "__cropped__":
                encoded_bytes = bytes(request.image)
            else:
                face_box = self._detect_face(image, 0.5)
                if face_box is None:
                    return pb.FaceLibraryResponse(
                        success=False, message="No face was detected"
                    )
                face = image[face_box[1] : face_box[3], face_box[0] : face_box[2]]
                ok, encoded = cv2.imencode(
                    ".webp", face, [int(cv2.IMWRITE_WEBP_QUALITY), 100]
                )
                if not ok:
                    return pb.FaceLibraryResponse(
                        success=False, message="Face encode failed"
                    )
                encoded_bytes = encoded.tobytes()
            if not encoded_bytes:
                return pb.FaceLibraryResponse(
                    success=False, message="Invalid face image"
                )
            target = Path(FACE_DIR) / request.label
            target.mkdir(parents=True, exist_ok=True)
            filename = f"{request.label}_{datetime.datetime.now().timestamp()}.webp"
            (target / filename).write_bytes(encoded_bytes)
            self.recognizer.clear()
            return pb.FaceLibraryResponse(success=True, message="Face registered")
        if request.operation in {
            pb.FACE_LIBRARY_OPERATION_RECOGNIZE,
            pb.FACE_LIBRARY_OPERATION_REPROCESS,
        }:
            if request.operation == pb.FACE_LIBRARY_OPERATION_REPROCESS:
                crop = image
            else:
                face_box = self._detect_face(image, 0.5)
                if face_box is None:
                    return pb.FaceLibraryResponse(
                        success=False, message="No face was detected"
                    )
                crop = image[face_box[1] : face_box[3], face_box[0] : face_box[2]]
            result = self.recognizer.classify(crop)
            if result is None:
                return pb.FaceLibraryResponse(
                    success=False, message="No face was recognized"
                )
            name, score = result
            if score <= self.face_config.unknown_score:
                name = "unknown"
            return pb.FaceLibraryResponse(
                success=True,
                message="Face recognized",
                face_name=name,
                score=float(score),
            )
        return pb.FaceLibraryResponse(success=False, message="Operation is unspecified")

    def _build_detector(self) -> None:
        path = os.path.join(MODEL_CACHE_DIR, "facedet/facedet.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Face detector model is missing: {path}")
        self.face_detector = cv2.FaceDetectorYN.create(
            path,
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
        )

    def _detect_face(
        self, image: np.ndarray, threshold: float
    ) -> tuple[int, int, int, int] | None:
        return detect_largest_face(self.face_detector, image, threshold)


class _LprBase:
    def __init__(self, config: FrigateConfig, metrics: ServiceMetrics) -> None:
        self.config = config
        self.metrics = metrics


class LprRecognitionModel(LicensePlateProcessingMixin, _LprBase):
    def __init__(self, config: FrigateConfig, metrics: ServiceMetrics) -> None:
        self._emit_runtime_side_effects = False
        self.requestor = _Requestor()
        self.model_runner = LicensePlateModelRunner(
            self.requestor,
            device=config.lpr.device or "CPU",
            model_size=config.lpr.model_size,
        )
        self.lpr_config = config.lpr
        super().__init__(config, metrics)

    def recognize(
        self, observation: TrackedObservation, frame: np.ndarray
    ) -> ModelRecognition:
        object_data = dict(observation.attributes.get("object_data", {}))
        if not object_data:
            raise ValueError("LPR observation requires object_data")
        capture = observation.evidence_capture
        if capture is None:
            result = self.lpr_process(object_data, frame, False)
            return ModelRecognition(
                result if isinstance(result, RawRecognition) else None
            )
        object_data["_recognition_trace_id"] = capture.trace_id
        object_data["_recognition_evidence_id"] = capture.evidence_id
        collector = _ArtifactCollector(capture.run_id)
        with capture_passage_evidence(collector):
            result = self.lpr_process(object_data, frame, False)
        return ModelRecognition(
            result if isinstance(result, RawRecognition) else None,
            tuple(collector.artifacts),
        )


class FrigateRecognitionModel:
    def __init__(self, config: FrigateConfig) -> None:
        self.face = (
            FaceRecognitionModel(config) if config.face_recognition.enabled else None
        )
        metrics = ServiceMetrics()
        self.lpr = LprRecognitionModel(config, metrics) if config.lpr.enabled else None

    def recognize(
        self,
        task: RecognitionTask,
        observation: TrackedObservation,
        evidence: object,
    ) -> RawRecognition | ModelRecognition | None:
        frame = np.asarray(evidence)
        if task is RecognitionTask.FACE:
            if self.face is None:
                return None
            return self.face.recognize(observation, frame)
        if self.lpr is None:
            return None
        return self.lpr.recognize(observation, frame)

    async def manage_face_library(
        self, request: pb.FaceLibraryRequest
    ) -> pb.FaceLibraryResponse:
        if self.face is None:
            return pb.FaceLibraryResponse(
                success=False, message="Face recognition is disabled"
            )
        return await self.face.manage(request)
