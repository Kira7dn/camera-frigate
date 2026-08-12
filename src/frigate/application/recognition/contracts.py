"""Transport-neutral contracts for synchronous recognition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias

BBox: TypeAlias = tuple[int, int, int, int]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(item) for item in value)
    return value


class RecognitionTask(StrEnum):
    FACE = "face"
    LPR = "lpr"


class RecognitionOperation(StrEnum):
    OBSERVE = "observe"
    END_TRACK = "end_track"
    CANCEL = "cancel"


class RecognitionOutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ENDED = "ended"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TrackKey:
    camera_id: str
    stream_epoch: str
    track_id: str


@dataclass(frozen=True, slots=True)
class EvidenceCaptureRequest:
    trace_id: str
    evidence_id: str
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.trace_id or not self.evidence_id:
            raise ValueError("trace_id and evidence_id are required")


@dataclass(frozen=True, slots=True)
class RecognitionArtifact:
    sequence: int
    stage: str
    pipeline: str
    trace_id: str
    evidence_id: str
    camera: str
    frame_time: float | None
    track_id: str | None
    image_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    image_jpeg: bytes = b""
    image_shape: tuple[int, ...] = ()
    image_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.sequence < 0 or not self.stage or not self.evidence_id:
            raise ValueError("invalid recognition artifact identity")
        if self.image_jpeg:
            digest = hashlib.sha256(self.image_jpeg).hexdigest()
            if self.image_sha256 != digest:
                raise ValueError("recognition artifact checksum mismatch")
        elif self.image_sha256 or self.image_shape:
            raise ValueError("metadata-only artifact must not describe an image")


@dataclass(frozen=True, slots=True)
class TrackedObservation:
    task: RecognitionTask
    key: TrackKey
    frame_time: float
    object_bbox: BBox
    detail_bbox: BBox | None = None
    observed_in_frame: bool | None = None
    evidence_ref: object | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    evidence_capture: EvidenceCaptureRequest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze(self.attributes))


@dataclass(frozen=True, slots=True)
class RecognitionUpdate:
    task: RecognitionTask
    key: TrackKey
    frame_time: float
    evidence_ref: object | None
    raw_value: str | None
    raw_score: float
    aggregate_value: str | None
    aggregate_score: float
    object_bbox: BBox
    detail_bbox: BBox | None
    publish: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class RecognitionJob:
    job_id: str
    client_id: str
    service_epoch: str
    key: TrackKey
    sequence: int
    operation: RecognitionOperation
    observation: TrackedObservation | None = None
    deadline_monotonic: float | None = None
    reason: str = ""
    target_job_id: str | None = None

    def __post_init__(self) -> None:
        if not self.job_id or not self.client_id:
            raise ValueError("job_id and client_id are required")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.operation is RecognitionOperation.OBSERVE:
            if self.observation is None or self.observation.key != self.key:
                raise ValueError("observe job requires an observation for the same key")
        elif self.observation is not None:
            raise ValueError("control job must not include an observation")
        if self.operation is RecognitionOperation.CANCEL and not self.target_job_id:
            raise ValueError("cancel job requires target_job_id")


@dataclass(frozen=True, slots=True)
class JobReceipt:
    job_id: str
    service_epoch: str
    accepted: bool
    reason: str = "accepted"
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RecognitionOutcome:
    job_id: str
    client_id: str
    service_epoch: str
    key: TrackKey
    sequence: int
    status: RecognitionOutcomeStatus
    updates: tuple[RecognitionUpdate, ...] = ()
    reason: str = ""
    retryable: bool = False
    artifacts: tuple[RecognitionArtifact, ...] = ()
