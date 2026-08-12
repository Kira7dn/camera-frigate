"""Injected ports used by the standalone core."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import (
    BBox,
    RecognitionArtifact,
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
    _freeze,
)


@dataclass(frozen=True, slots=True)
class RawRecognition:
    value: str | None
    score: float
    detail_bbox: BBox | None = None
    area: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelRecognition:
    result: RawRecognition | None
    artifacts: tuple[RecognitionArtifact, ...] = ()


class EvidenceResolver(Protocol):
    def resolve(
        self, observation: TrackedObservation
    ) -> AbstractContextManager[object]: ...


class RecognitionModel(Protocol):
    def recognize(
        self,
        task: RecognitionTask,
        observation: TrackedObservation,
        evidence: object,
    ) -> RawRecognition | ModelRecognition | None: ...


class RecognitionObserver(Protocol):
    def on_update(self, update: RecognitionUpdate) -> None: ...

    def on_track_end(
        self, task: RecognitionTask, key: TrackKey, reason: str
    ) -> None: ...
