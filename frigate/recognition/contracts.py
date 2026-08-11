"""Transport-neutral contracts for synchronous recognition."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class TrackKey:
    camera_id: str
    stream_epoch: str
    track_id: str


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attributes", _freeze(self.attributes)
        )


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
