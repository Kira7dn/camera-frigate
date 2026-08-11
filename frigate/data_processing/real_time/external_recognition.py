"""Fail-closed Frigate adapter for the dedicated recognition runtime."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.const import FACE_DIR
from frigate.data_processing.common.license_plate.mixin import lpr_camera_eligible
from frigate.recognition.adapters.frigate import FrigateEventAdapter
from frigate.recognition.contracts import (
    EvidenceCaptureRequest,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcomeStatus,
    RecognitionTask,
    TrackedObservation,
    TrackKey,
)
from frigate.recognition.service.config_fingerprint import canonical_config_json
from frigate.recognition.service.evidence import RawI420Evidence
from frigate.recognition.service.grpc_client import TlsClientConfig
from frigate.recognition.service.threaded_client import ThreadedRecognitionClient
from frigate.recognition.service.v1 import recognition_pb2 as pb
from frigate.util.face_snapshot import (
    FaceAttemptJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    LatestPerObjectWorker,
    write_face_attempt,
    write_face_snapshot_artifact,
)
from frigate.util.passage_trace import (
    canonical_trace_id,
    passage_evidence_enabled,
    passage_evidence_id,
    passage_evidence_should_capture,
    passage_trace,
    persist_passage_evidence_bundle,
)
from rapidfuzz.distance import Levenshtein

from frigate.config import FrigateConfig

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


class ExternalRecognitionProcessor(RealTimeProcessorApi):
    """Submit copied frame evidence and publish only validated service outcomes."""

    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        stream_epoch: str,
    ) -> None:
        super().__init__(config, metrics)
        self._stream_epoch = stream_epoch
        self._requestor = requestor
        self._publisher = sub_label_publisher
        self._client_id = f"frigate-{uuid.uuid4().hex}"
        self._sequence: dict[TrackKey, int] = {}
        self._last_applied: dict[TrackKey, int] = {}
        self._ending: set[TrackKey] = set()
        self._ended: set[TrackKey] = set()
        self._evidence_by_job: dict[str, np.ndarray] = {}
        self._task_by_job: dict[str, RecognitionTask] = {}
        self._capture_jobs: dict[
            str, tuple[RecognitionTask, float, tuple[int, int, int, int]]
        ] = {}
        self._snapshot_published: set[tuple[str, str]] = set()
        self._rejected = 0
        self._snapshot_worker = LatestPerObjectWorker(
            write_face_snapshot_artifact,
            max_objects=max(4, min(32, 4 * len(config.cameras))),
        )
        self._attempt_worker = LatestPerObjectWorker(
            write_face_attempt,
            max_objects=4,
            name="external_face_attempt_worker",
        )
        runtime = config.recognition
        tls = TlsClientConfig(
            root_ca=_read_bytes(runtime.tls.ca),
            certificate=_read_bytes(runtime.tls.certificate),
            private_key=_read_bytes(runtime.tls.key),
            server_name=runtime.tls.server_name,
        )
        config_json = canonical_config_json(config)
        self._client = ThreadedRecognitionClient(
            runtime.endpoint,
            self._client_id,
            config_json,
            tls=tls,
            deadline=runtime.deadline,
            observation_capacity=runtime.observation_capacity,
            control_capacity=runtime.control_capacity,
            outcome_capacity=runtime.outcome_capacity,
        )
        self._events = FrigateEventAdapter(
            lambda payload: self._requestor.send_data(
                "tracked_object_update", json.dumps(payload)
            ),
            lambda kind, payload: self._publisher.publish(payload, kind),
            known_plate_label=self._known_plate_label,
        )

    @property
    def recognition_stats(self) -> dict[str, int]:
        stats = self._client.stats
        return {
            "sessions": len(self._sequence),
            "in_flight": len(self._evidence_by_job),
            "evidence_pinned": len(self._evidence_by_job),
            "queue_depth": stats["queue_depth"],
            "outcome_depth": stats["outcome_depth"],
            "rejected": self._rejected,
            "service_healthy": stats["healthy"],
        }

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        if not obj_data.get("box"):
            return
        camera = str(obj_data["camera"])
        tasks = []
        if (
            self.config.face_recognition.enabled
            and self.config.cameras[camera].face_recognition.enabled
            and obj_data.get("label") == "person"
        ):
            tasks.append(RecognitionTask.FACE)
        if (
            self.config.lpr.enabled
            and self.config.cameras[camera].lpr.enabled
            and lpr_camera_eligible(camera)
        ):
            tasks.append(RecognitionTask.LPR)
        for task in tasks:
            self._submit_observation(task, obj_data, frame)

    def _submit_observation(
        self, task: RecognitionTask, obj_data: dict[str, Any], frame: np.ndarray
    ) -> None:
        key = TrackKey(str(obj_data["camera"]), self._stream_epoch, str(obj_data["id"]))
        if key in self._ending or key in self._ended:
            return
        sequence = self._next_sequence(key)
        frame_time = float(obj_data["frame_time"])
        passage_trace(
            "track_seen",
            camera=key.camera_id,
            frame_time=frame_time,
            track_id=key.track_id,
            trace_id=canonical_trace_id(task.value, key.camera_id, key.track_id),
            task=task.value,
            object_box=list(obj_data["box"]),
        )
        evidence_id = (
            f"{task.value}:{key.camera_id}:{key.track_id}:{frame_time:.6f}:{sequence}"
        )
        copied = np.ascontiguousarray(frame).copy()
        evidence = RawI420Evidence(
            evidence_id,
            copied.tobytes(),
            tuple(copied.shape),
            "uint8",
            "I420",
            copied.nbytes,
            int((time.time() + self.config.recognition.job_deadline + 1) * 1000),
        )
        attributes = {
            "label": obj_data.get("label"),
            "sub_label": obj_data.get("sub_label"),
            "current_attributes": _json_value(obj_data.get("current_attributes", ())),
            "detect_fps": self.config.cameras[key.camera_id].detect.fps,
        }
        if task is RecognitionTask.LPR:
            attributes["object_data"] = _json_value(obj_data)
        capture = None
        capture_selected = passage_evidence_enabled() and (
            task is RecognitionTask.FACE
            or passage_evidence_should_capture(key.camera_id, key.track_id, frame_time)
        )
        if capture_selected:
            capture = EvidenceCaptureRequest(
                canonical_trace_id(task.value, key.camera_id, key.track_id),
                passage_evidence_id(key.camera_id, key.track_id, frame_time, sequence),
                os.environ.get("PASSAGE_RUN_ID") or None,
            )
        observation = TrackedObservation(
            task,
            key,
            frame_time,
            tuple(int(value) for value in obj_data["box"]),
            observed_in_frame=obj_data.get("observed_in_frame"),
            evidence_ref=evidence,
            attributes=attributes,
            evidence_capture=capture,
        )
        job_id = uuid.uuid4().hex
        job = RecognitionJob(
            job_id,
            self._client_id,
            self._client.service_epoch,
            key,
            sequence,
            RecognitionOperation.OBSERVE,
            observation,
            time.monotonic() + self.config.recognition.job_deadline,
        )
        receipt = self._client.submit_nowait(job)
        if not receipt.accepted:
            self._rejected += 1
            logger.warning(
                "Recognition observation rejected job=%s reason=%s retryable=%s",
                job_id,
                receipt.reason,
                receipt.retryable,
            )
            return
        self._evidence_by_job[job_id] = copied
        self._task_by_job[job_id] = task
        if capture is not None:
            self._capture_jobs[job_id] = (
                task,
                frame_time,
                tuple(int(value) for value in obj_data["box"]),
            )

    def drain_results(self) -> list[Any]:
        for result in self._client.drain_results():
            if result.receipt is not None and not result.receipt.accepted:
                self._rejected += 1
                self._evidence_by_job.pop(result.receipt.job_id, None)
                self._task_by_job.pop(result.receipt.job_id, None)
                self._capture_jobs.pop(result.receipt.job_id, None)
                logger.error(
                    "Recognition service rejected job=%s reason=%s",
                    result.receipt.job_id,
                    result.receipt.reason,
                )
                continue
            outcome = result.outcome
            if outcome is None:
                continue
            frame = self._evidence_by_job.pop(outcome.job_id, None)
            task = self._task_by_job.pop(outcome.job_id, None)
            capture_info = self._capture_jobs.pop(outcome.job_id, None)
            if outcome.service_epoch != self._client.service_epoch:
                continue
            if outcome.status is RecognitionOutcomeStatus.ENDED:
                self._ending.discard(outcome.key)
                self._ended.add(outcome.key)
                self._sequence.pop(outcome.key, None)
                self._last_applied.pop(outcome.key, None)
                self._snapshot_published.discard(
                    (outcome.key.camera_id, outcome.key.track_id)
                )
                continue
            if outcome.status is not RecognitionOutcomeStatus.SUCCEEDED:
                failed_task = task or (
                    capture_info[0] if capture_info is not None else None
                )
                if failed_task is None:
                    continue
                passage_trace(
                    "recognition_failed",
                    camera=outcome.key.camera_id,
                    track_id=outcome.key.track_id,
                    trace_id=canonical_trace_id(
                        failed_task.value,
                        outcome.key.camera_id,
                        outcome.key.track_id,
                    ),
                    task=failed_task.value,
                    reason=outcome.reason or outcome.status.value,
                )
                if outcome.status is RecognitionOutcomeStatus.FAILED:
                    logger.error(
                        "Recognition job failed job=%s reason=%s",
                        outcome.job_id,
                        outcome.reason,
                    )
                continue
            if capture_info is not None:
                if not outcome.artifacts:
                    task, frame_time, object_box = capture_info
                    passage_trace(
                        "recognition_skipped",
                        camera=outcome.key.camera_id,
                        frame_time=frame_time,
                        track_id=outcome.key.track_id,
                        trace_id=canonical_trace_id(
                            task.value,
                            outcome.key.camera_id,
                            outcome.key.track_id,
                        ),
                        task=task.value,
                        object_box=list(object_box),
                        reason=outcome.reason or "no_recognition_result",
                    )
                elif not persist_passage_evidence_bundle(outcome.artifacts):
                    logger.error(
                        "Recognition evidence queue rejected job=%s", outcome.job_id
                    )
            if outcome.key in self._ended:
                continue
            if outcome.sequence <= self._last_applied.get(outcome.key, -1):
                continue
            self._last_applied[outcome.key] = outcome.sequence
            for update in outcome.updates:
                self._events.on_update(update)
                self._trace_update(update)
                if update.task is RecognitionTask.FACE and frame is not None:
                    face_crop = next(
                        (
                            cv2.imdecode(
                                np.frombuffer(artifact.image_jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            for artifact in outcome.artifacts
                            if artifact.stage == "face_crop" and artifact.image_jpeg
                        ),
                        None,
                    )
                    self._handle_face_media(update, frame, face_crop)
        payloads = []
        for value in self._snapshot_worker.drain_results():
            if isinstance(value, FaceRecognitionResult):
                payloads.append({"type": "face_snapshot", **value.as_payload()})
        return payloads

    def expire_object(self, object_id: str, camera: str) -> None:
        key = TrackKey(camera, self._stream_epoch, str(object_id))
        if key not in self._sequence:
            return
        job = RecognitionJob(
            uuid.uuid4().hex,
            self._client_id,
            self._client.service_epoch,
            key,
            self._next_sequence(key),
            RecognitionOperation.END_TRACK,
            reason="event_end",
        )
        receipt = self._client.submit_nowait(job)
        if not receipt.accepted:
            self._rejected += 1
            logger.error("Recognition end rejected reason=%s", receipt.reason)
            return
        self._ending.add(key)

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        operations = {
            EmbeddingsRequestEnum.clear_face_classifier.value: (
                pb.FACE_LIBRARY_OPERATION_CLEAR
            ),
            EmbeddingsRequestEnum.recognize_face.value: (
                pb.FACE_LIBRARY_OPERATION_RECOGNIZE
            ),
            EmbeddingsRequestEnum.register_face.value: (
                pb.FACE_LIBRARY_OPERATION_REGISTER
            ),
            EmbeddingsRequestEnum.reprocess_face.value: (
                pb.FACE_LIBRARY_OPERATION_REPROCESS
            ),
        }
        operation = operations.get(topic)
        if operation is None:
            return None
        image = b""
        logical_name = ""
        if topic == EmbeddingsRequestEnum.reprocess_face.value:
            logical_name = str(request_data.get("image_file", ""))
            try:
                image = Path(logical_name).read_bytes()
            except OSError:
                return {"success": False, "message": "Invalid image file."}
        elif "image" in request_data:
            value = request_data["image"]
            if isinstance(value, str):
                try:
                    image = base64.b64decode(value, validate=True)
                except ValueError:
                    return {"success": False, "message": "Invalid face image."}
            elif isinstance(value, bytes | bytearray):
                image = bytes(value)
            elif isinstance(value, np.ndarray):
                if request_data.get("cropped"):
                    image = value.tobytes()
                else:
                    ok, encoded = cv2.imencode(".webp", value)
                    if not ok:
                        return {"success": False, "message": "Invalid face image."}
                    image = encoded.tobytes()
            else:
                return {"success": False, "message": "Invalid face image."}
        response = self._client.manage_face_library(
            pb.FaceLibraryRequest(
                operation=operation,
                label=str(request_data.get("face_name", "")),
                image=image,
                logical_name=(
                    "__cropped__"
                    if topic == EmbeddingsRequestEnum.register_face.value
                    and bool(request_data.get("cropped"))
                    else os.path.basename(logical_name)
                ),
                deadline_budget_ms=int(self.config.recognition.deadline * 1000),
            ),
            self.config.recognition.deadline,
        )
        result: dict[str, Any] = {
            "success": response.success,
            "message": response.message,
        }
        if response.HasField("face_name"):
            result["face_name"] = response.face_name
        if response.HasField("score"):
            result["score"] = response.score
        if (
            response.success
            and topic == EmbeddingsRequestEnum.reprocess_face.value
            and self.config.face_recognition.save_attempts
        ):
            name = response.face_name.replace("-", "_")
            target = Path(FACE_DIR) / "train"
            target.mkdir(parents=True, exist_ok=True)
            parts = Path(logical_name).name.split("-")
            if len(parts) < 5:
                return {"success": False, "message": "Invalid image file."}
            id_time, id_rand, timestamp = parts[:3]
            shutil.move(
                logical_name,
                target
                / f"{id_time}-{id_rand}-{timestamp}-{name}-{response.score}.webp",
            )
        return result

    def shutdown(self) -> None:
        if not self._client.close(self.config.recognition.shutdown_drain):
            logger.error("Recognition client did not drain before shutdown deadline")
        self._snapshot_worker.stop()
        self._attempt_worker.stop()
        self._sequence.clear()
        self._last_applied.clear()
        self._ending.clear()
        self._ended.clear()
        self._evidence_by_job.clear()
        self._task_by_job.clear()
        self._capture_jobs.clear()

    def _handle_face_media(
        self, update, frame: np.ndarray, source_face_crop: np.ndarray | None
    ) -> None:
        face_box = update.detail_bbox
        if face_box is None:
            return
        if (
            source_face_crop is not None
            and source_face_crop.size
            and self.config.face_recognition.save_attempts
            and update.raw_value == "unknown"
        ):
            self._attempt_worker.submit(
                (update.key.camera_id, update.key.track_id),
                FaceAttemptJob(
                    frame=source_face_crop.copy(),
                    event_id=update.key.track_id,
                    timestamp=update.frame_time,
                    sub_label=update.raw_value or "unknown",
                    score=update.raw_score,
                    face_dir=FACE_DIR,
                    max_files=self.config.face_recognition.save_attempts,
                ),
            )
        snapshot_key = (update.key.camera_id, update.key.track_id)
        if not update.publish or snapshot_key in self._snapshot_published:
            return
        accepted = self._snapshot_worker.submit(
            snapshot_key,
            FaceSnapshotJob(
                camera=update.key.camera_id,
                event_id=update.key.track_id,
                frame_time=update.frame_time,
                person_box=update.object_bbox,
                face_box=face_box,
                sub_label=update.aggregate_value or "unknown",
                face_score=update.aggregate_score,
                frame=frame.copy(),
            ),
        )
        if accepted:
            self._snapshot_published.add(snapshot_key)

    @staticmethod
    def _trace_update(update: Any) -> None:
        task = update.task.value
        fields = {
            "camera": update.key.camera_id,
            "frame_time": update.frame_time,
            "track_id": update.key.track_id,
            "trace_id": canonical_trace_id(
                task, update.key.camera_id, update.key.track_id
            ),
            "object_box": list(update.object_bbox),
        }
        if update.task is RecognitionTask.FACE:
            detail_box = (
                list(update.detail_bbox) if update.detail_bbox is not None else None
            )
            for stage in ("first_qualified_face", "candidate_submitted"):
                passage_trace(stage, **fields, face_box=detail_box)
            passage_trace(
                "first_attempt",
                **fields,
                identity=update.raw_value,
                score=update.raw_score,
                face_box=detail_box,
            )
            if update.publish:
                passage_trace(
                    "confirmed_result",
                    **fields,
                    identity=update.aggregate_value,
                    score=update.aggregate_score,
                    face_box=detail_box,
                )
            return
        passage_trace(
            "ocr_result",
            **fields,
            plate=update.raw_value,
            score=update.raw_score,
        )
        if update.publish:
            passage_trace(
                "event_published",
                **fields,
                plate=update.aggregate_value,
                score=update.aggregate_score,
            )

    def _next_sequence(self, key: TrackKey) -> int:
        value = self._sequence.get(key, 0)
        self._sequence[key] = value + 1
        return value

    def _known_plate_label(self, plate: str) -> str | None:
        try:
            return next(
                (
                    label
                    for label, patterns in self.config.lpr.known_plates.items()
                    if any(
                        re.match(f"^{pattern}$", plate)
                        or Levenshtein.distance(pattern, plate)
                        <= self.config.lpr.match_distance
                        for pattern in patterns
                    )
                ),
                None,
            )
        except re.error:
            logger.error("Invalid known-plate regular expression")
            return None
