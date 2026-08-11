"""Conversion between transport-neutral contracts and protobuf messages."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from ..contracts import (
    BBox,
    EvidenceCaptureRequest,
    JobReceipt,
    RecognitionArtifact,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
)
from .evidence import RawI420Evidence
from .v1 import recognition_pb2 as pb

_TASK_TO_PROTO = {
    RecognitionTask.FACE: pb.RECOGNITION_TASK_FACE,
    RecognitionTask.LPR: pb.RECOGNITION_TASK_LPR,
}
_TASK_FROM_PROTO = {value: key for key, value in _TASK_TO_PROTO.items()}
_STATUS_TO_PROTO = {
    RecognitionOutcomeStatus.SUCCEEDED: pb.RECOGNITION_OUTCOME_STATUS_SUCCEEDED,
    RecognitionOutcomeStatus.ENDED: pb.RECOGNITION_OUTCOME_STATUS_ENDED,
    RecognitionOutcomeStatus.CANCELLED: pb.RECOGNITION_OUTCOME_STATUS_CANCELLED,
    RecognitionOutcomeStatus.DEADLINE_EXCEEDED: pb.RECOGNITION_OUTCOME_STATUS_DEADLINE_EXCEEDED,
    RecognitionOutcomeStatus.FAILED: pb.RECOGNITION_OUTCOME_STATUS_FAILED,
}
_STATUS_FROM_PROTO = {value: key for key, value in _STATUS_TO_PROTO.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(_plain(item) for item in value)
    return value


def _bbox_from_proto(value: pb.BBox) -> BBox:
    return (value.left, value.top, value.right, value.bottom)


def _bbox_to_proto(value: BBox) -> pb.BBox:
    return pb.BBox(left=value[0], top=value[1], right=value[2], bottom=value[3])


def _key_from_proto(value: pb.TrackKey) -> TrackKey:
    return TrackKey(value.camera_id, value.stream_epoch, value.track_id)


def _key_to_proto(value: TrackKey) -> pb.TrackKey:
    return pb.TrackKey(
        camera_id=value.camera_id,
        stream_epoch=value.stream_epoch,
        track_id=value.track_id,
    )


def evidence_from_proto(value: pb.RawI420Evidence) -> RawI420Evidence:
    evidence = RawI420Evidence(
        value.evidence_id,
        bytes(value.data),
        tuple(value.shape),
        value.dtype,
        value.layout,
        value.byte_length,
        value.expiry_unix_ms,
    )
    evidence.validate()
    return evidence


def job_from_envelope(envelope: pb.ClientEnvelope) -> RecognitionJob:
    request = envelope.WhichOneof("request")
    if request == "observe":
        value = envelope.observe
        observation = value.observation
        key = _key_from_proto(observation.key)
        task = _TASK_FROM_PROTO.get(observation.task)
        if task is None:
            raise ValueError("recognition task is unspecified")
        tracked = TrackedObservation(
            task,
            key,
            observation.frame_time,
            _bbox_from_proto(observation.object_bbox),
            _bbox_from_proto(observation.detail_bbox)
            if observation.HasField("detail_bbox")
            else None,
            observation.observed_in_frame
            if observation.HasField("observed_in_frame")
            else None,
            evidence_from_proto(observation.evidence),
            json.loads(observation.attributes_json or "{}"),
            EvidenceCaptureRequest(
                observation.evidence_capture.trace_id,
                observation.evidence_capture.evidence_id,
                observation.evidence_capture.run_id
                if observation.evidence_capture.HasField("run_id")
                else None,
            )
            if observation.HasField("evidence_capture")
            else None,
        )
        return RecognitionJob(
            value.job_id,
            value.client_id,
            value.expected_service_epoch,
            key,
            value.sequence,
            RecognitionOperation.OBSERVE,
            tracked,
            time.monotonic() + value.deadline_budget_ms / 1000,
        )
    if request == "end_track":
        value = envelope.end_track
        return RecognitionJob(
            value.job_id,
            value.client_id,
            value.expected_service_epoch,
            _key_from_proto(value.key),
            value.sequence,
            RecognitionOperation.END_TRACK,
            reason=value.reason,
        )
    if request == "cancel":
        value = envelope.cancel
        return RecognitionJob(
            value.job_id,
            value.client_id,
            value.expected_service_epoch,
            _key_from_proto(value.key),
            value.sequence,
            RecognitionOperation.CANCEL,
            target_job_id=value.target_job_id,
        )
    raise ValueError("request is unspecified")


def receipt_to_proto(value: JobReceipt) -> pb.JobReceipt:
    return pb.JobReceipt(
        job_id=value.job_id,
        service_epoch=value.service_epoch,
        accepted=value.accepted,
        reason=value.reason,
        retryable=value.retryable,
    )


def _update_to_proto(value: RecognitionUpdate) -> pb.RecognitionUpdate:
    evidence = value.evidence_ref
    evidence_id = (
        evidence.evidence_id
        if isinstance(evidence, RawI420Evidence)
        else str(evidence or "")
    )
    result = pb.RecognitionUpdate(
        task=_TASK_TO_PROTO[value.task],
        key=_key_to_proto(value.key),
        frame_time=value.frame_time,
        evidence_id=evidence_id,
        raw_score=value.raw_score,
        aggregate_score=value.aggregate_score,
        object_bbox=_bbox_to_proto(value.object_bbox),
        publish=value.publish,
        reason=value.reason,
        metadata_json=json.dumps(_plain(value.metadata), separators=(",", ":")),
    )
    if value.raw_value is not None:
        result.raw_value = value.raw_value
    if value.aggregate_value is not None:
        result.aggregate_value = value.aggregate_value
    if value.detail_bbox is not None:
        result.detail_bbox.CopyFrom(_bbox_to_proto(value.detail_bbox))
    return result


def outcome_to_proto(value: RecognitionOutcome) -> pb.RecognitionOutcome:
    return pb.RecognitionOutcome(
        job_id=value.job_id,
        client_id=value.client_id,
        service_epoch=value.service_epoch,
        key=_key_to_proto(value.key),
        sequence=value.sequence,
        status=_STATUS_TO_PROTO[value.status],
        updates=[_update_to_proto(update) for update in value.updates],
        reason=value.reason,
        retryable=value.retryable,
        artifacts=[_artifact_to_proto(artifact) for artifact in value.artifacts],
    )


def _artifact_to_proto(value: RecognitionArtifact) -> pb.RecognitionArtifact:
    result = pb.RecognitionArtifact(
        sequence=value.sequence,
        stage=value.stage,
        pipeline=value.pipeline,
        trace_id=value.trace_id,
        evidence_id=value.evidence_id,
        camera=value.camera,
        metadata_json=json.dumps(_plain(value.metadata), separators=(",", ":")),
        image_jpeg=value.image_jpeg,
        image_shape=value.image_shape,
        image_sha256=value.image_sha256,
    )
    if value.frame_time is not None:
        result.frame_time = value.frame_time
    if value.track_id is not None:
        result.track_id = value.track_id
    if value.image_index is not None:
        result.image_index = value.image_index
    return result


def envelope_from_job(value: RecognitionJob) -> pb.ClientEnvelope:
    if value.operation is RecognitionOperation.OBSERVE:
        observation = value.observation
        if observation is None or not isinstance(
            observation.evidence_ref, RawI420Evidence
        ):
            raise ValueError("observe job requires raw I420 evidence")
        evidence = observation.evidence_ref
        evidence.validate()
        message = pb.Observation(
            task=_TASK_TO_PROTO[observation.task],
            key=_key_to_proto(observation.key),
            frame_time=observation.frame_time,
            object_bbox=_bbox_to_proto(observation.object_bbox),
            evidence=pb.RawI420Evidence(
                evidence_id=evidence.evidence_id,
                data=evidence.data,
                shape=evidence.shape,
                dtype=evidence.dtype,
                layout=evidence.layout,
                byte_length=evidence.byte_length,
                expiry_unix_ms=evidence.expiry_unix_ms,
            ),
            attributes_json=json.dumps(
                _plain(observation.attributes), separators=(",", ":")
            ),
        )
        if observation.detail_bbox is not None:
            message.detail_bbox.CopyFrom(_bbox_to_proto(observation.detail_bbox))
        if observation.observed_in_frame is not None:
            message.observed_in_frame = observation.observed_in_frame
        if observation.evidence_capture is not None:
            capture = observation.evidence_capture
            message.evidence_capture.CopyFrom(
                pb.EvidenceCaptureRequest(
                    trace_id=capture.trace_id,
                    evidence_id=capture.evidence_id,
                )
            )
            if capture.run_id is not None:
                message.evidence_capture.run_id = capture.run_id
        budget_ms = 5000
        if value.deadline_monotonic is not None:
            budget_ms = max(
                1, int((value.deadline_monotonic - time.monotonic()) * 1000)
            )
        return pb.ClientEnvelope(
            observe=pb.Observe(
                job_id=value.job_id,
                client_id=value.client_id,
                expected_service_epoch=value.service_epoch,
                sequence=value.sequence,
                deadline_budget_ms=budget_ms,
                observation=message,
            )
        )
    if value.operation is RecognitionOperation.END_TRACK:
        return pb.ClientEnvelope(
            end_track=pb.EndTrack(
                job_id=value.job_id,
                client_id=value.client_id,
                expected_service_epoch=value.service_epoch,
                key=_key_to_proto(value.key),
                sequence=value.sequence,
                reason=value.reason,
            )
        )
    return pb.ClientEnvelope(
        cancel=pb.Cancel(
            job_id=value.job_id,
            client_id=value.client_id,
            expected_service_epoch=value.service_epoch,
            key=_key_to_proto(value.key),
            sequence=value.sequence,
            target_job_id=value.target_job_id or "",
        )
    )


def receipt_from_proto(value: pb.JobReceipt) -> JobReceipt:
    return JobReceipt(
        value.job_id,
        value.service_epoch,
        value.accepted,
        value.reason,
        value.retryable,
    )


def _update_from_proto(value: pb.RecognitionUpdate) -> RecognitionUpdate:
    task = _TASK_FROM_PROTO.get(value.task)
    if task is None:
        raise ValueError("recognition update task is unspecified")
    return RecognitionUpdate(
        task,
        _key_from_proto(value.key),
        value.frame_time,
        value.evidence_id,
        value.raw_value if value.HasField("raw_value") else None,
        value.raw_score,
        value.aggregate_value if value.HasField("aggregate_value") else None,
        value.aggregate_score,
        _bbox_from_proto(value.object_bbox),
        _bbox_from_proto(value.detail_bbox) if value.HasField("detail_bbox") else None,
        value.publish,
        value.reason,
        json.loads(value.metadata_json or "{}"),
    )


def outcome_from_proto(value: pb.RecognitionOutcome) -> RecognitionOutcome:
    status = _STATUS_FROM_PROTO.get(value.status)
    if status is None:
        raise ValueError("recognition outcome status is unspecified")
    return RecognitionOutcome(
        job_id=value.job_id,
        client_id=value.client_id,
        service_epoch=value.service_epoch,
        key=_key_from_proto(value.key),
        sequence=value.sequence,
        status=status,
        updates=tuple(_update_from_proto(update) for update in value.updates),
        reason=value.reason,
        retryable=value.retryable,
        artifacts=tuple(_artifact_from_proto(item) for item in value.artifacts),
    )


def _artifact_from_proto(value: pb.RecognitionArtifact) -> RecognitionArtifact:
    return RecognitionArtifact(
        sequence=value.sequence,
        stage=value.stage,
        pipeline=value.pipeline,
        trace_id=value.trace_id,
        evidence_id=value.evidence_id,
        camera=value.camera,
        frame_time=value.frame_time if value.HasField("frame_time") else None,
        track_id=value.track_id if value.HasField("track_id") else None,
        image_index=value.image_index if value.HasField("image_index") else None,
        metadata=json.loads(value.metadata_json or "{}"),
        image_jpeg=bytes(value.image_jpeg),
        image_shape=tuple(value.image_shape),
        image_sha256=value.image_sha256,
    )
