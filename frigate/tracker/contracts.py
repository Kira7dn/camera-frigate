"""Transport-neutral contracts shared by embedded and edge tracker producers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TrackerOperation(StrEnum):
    START = "START"
    UPDATE = "UPDATE"
    END = "END"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError("bbox origin must be non-negative")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bbox must have positive area")


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    node_id: str
    camera_id: str
    stream_epoch: str
    track_id: str

    def __post_init__(self) -> None:
        if not all((self.node_id, self.camera_id, self.stream_epoch, self.track_id)):
            raise ValueError("track identity fields are required")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    byte_length: int
    sha256: str
    expiry_unix_ms: int
    durable: bool = False

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        camera_id: str,
        stream_epoch: str,
        frame_seq: int,
        data: bytes,
        expiry_unix_ms: int,
        durable: bool = False,
    ) -> EvidenceReference:
        digest = hashlib.sha256(data).hexdigest()
        evidence_id = (
            f"ev1:{node_id}:{camera_id}:{stream_epoch}:{frame_seq}:{digest[:16]}"
        )
        return cls(evidence_id, len(data), digest, expiry_unix_ms, durable)

    def validate_binding(
        self,
        identity: TrackIdentity,
        frame_seq: int,
    ) -> None:
        expected = (
            f"ev1:{identity.node_id}:{identity.camera_id}:"
            f"{identity.stream_epoch}:{frame_seq}:{self.sha256[:16]}"
        )
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not bind update lineage and checksum")
        if len(self.sha256) != 64:
            raise ValueError("evidence sha256 must be a full hex digest")


@dataclass(frozen=True, slots=True)
class LifecycleFailure:
    code: str
    detail: str = ""
    retryable: bool = False
    gap: bool = False

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("typed failure code is required")


@dataclass(frozen=True, slots=True)
class MediaManifest:
    media_id: str
    event_id: str
    camera_id: str
    start_time: float
    end_time: float
    codec: str
    byte_size: int
    sha256: str
    expiry_unix_ms: int
    media_type: str

    def __post_init__(self) -> None:
        if not all((self.media_id, self.event_id, self.camera_id, self.codec)):
            raise ValueError("media identity and codec are required")
        if self.end_time < self.start_time or self.byte_size < 0:
            raise ValueError("invalid media time or size")
        if len(self.sha256) != 64:
            raise ValueError("media sha256 must be a full hex digest")


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    node_id: str
    node_epoch: str
    camera_id: str
    stream_epoch: str
    journal_sequence: int
    frame_seq: int
    source_pts: int
    frame_time: float
    event_id: str
    track_id: str
    operation: TrackerOperation
    label: str
    score_history: tuple[float, ...]
    score: float
    bbox: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)
    current_zones: tuple[str, ...] = ()
    entered_zones: tuple[str, ...] = ()
    path: tuple[tuple[float, float], ...] = ()
    speed: float | None = None
    motion: dict[str, Any] = field(default_factory=dict)
    region: dict[str, Any] = field(default_factory=dict)
    evidence: EvidenceReference | None = None
    failure: LifecycleFailure | None = None
    media: tuple[MediaManifest, ...] = ()

    @property
    def identity(self) -> TrackIdentity:
        return TrackIdentity(
            self.node_id, self.camera_id, self.stream_epoch, self.track_id
        )

    def __post_init__(self) -> None:
        if not all(
            (
                self.node_id,
                self.node_epoch,
                self.camera_id,
                self.stream_epoch,
                self.event_id,
                self.track_id,
                self.label,
            )
        ):
            raise ValueError("tracker update identity fields are required")
        if self.journal_sequence < 0 or self.frame_seq < 0:
            raise ValueError("sequences must be non-negative")
        if not 0 <= self.score <= 1 or any(
            score < 0 or score > 1 for score in self.score_history
        ):
            raise ValueError("scores must be in [0, 1]")
        if self.evidence is not None:
            self.evidence.validate_binding(self.identity, self.frame_seq)
        for manifest in self.media:
            if manifest.event_id != self.event_id:
                raise ValueError("media manifest must use producer event_id")
            if manifest.camera_id != self.camera_id:
                raise ValueError("media manifest camera mismatch")

    def to_json(self) -> str:
        value = asdict(self)
        value["operation"] = self.operation.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> TrackerUpdate:
        data = json.loads(value)
        data["operation"] = TrackerOperation(data["operation"])
        data["bbox"] = BoundingBox(**data["bbox"])
        if data.get("evidence") is not None:
            data["evidence"] = EvidenceReference(**data["evidence"])
        if data.get("failure") is not None:
            data["failure"] = LifecycleFailure(**data["failure"])
        data["media"] = tuple(MediaManifest(**item) for item in data.get("media", ()))
        for key in ("score_history", "current_zones", "entered_zones"):
            data[key] = tuple(data.get(key, ()))
        data["path"] = tuple(tuple(point) for point in data.get("path", ()))
        return cls(**data)
