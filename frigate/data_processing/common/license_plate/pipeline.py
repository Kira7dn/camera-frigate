"""Bounded, latest-only execution primitives for realtime LPR."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Any

from frigate.data_processing.common.evidence import EvidenceCandidate, FrameRef
from frigate.data_processing.common.recognition import (
    RecognitionAttemptLease,
    RecognitionOutcome,
)


@dataclass(frozen=True, slots=True)
class LprTrackKey:
    camera: str
    passage_id: str
    generation: int

    @property
    def track_id(self) -> str:
        return self.passage_id


@dataclass(slots=True)
class LprFrameTask:
    key: LprTrackKey
    obj_data: dict[str, Any] | str
    frame_ref: FrameRef
    dedicated_lpr: bool
    frame_time: float
    enqueued_at: float = field(default_factory=time.monotonic)
    collection_deadline: float = 0.0
    priority: float = 0.0


@dataclass(frozen=True, slots=True)
class LprExpireTask:
    camera: str
    track_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class PlateObservation:
    key: LprTrackKey
    frame_time: float
    plate: str
    char_confidences: tuple[float, ...]
    text_area: int
    plate_box: tuple[int, int, int, int]
    object_box: tuple[int, int, int, int] | None
    obj_data: dict[str, Any] | None
    dedicated_lpr: bool
    evidence: EvidenceCandidate
    attempt: RecognitionAttemptLease | None = None
    ocr_path: str = "paddle_detected_text"
    ocr_variant: str | None = None
    vehicle_track_id: str | None = None
    plate_track_ids: tuple[str, ...] = ()

    @property
    def confidence(self) -> float:
        if not self.char_confidences:
            return 0.0
        return sum(self.char_confidences) / len(self.char_confidences)


@dataclass(frozen=True, slots=True)
class PreparedPlateCandidate:
    key: LprTrackKey
    frame_time: float
    plate_box: tuple[int, int, int, int]
    object_box: tuple[int, int, int, int] | None
    obj_data: dict[str, Any] | None
    dedicated_lpr: bool
    evidence: EvidenceCandidate
    plate_frame: Any
    detector_score: float | None = None
    vehicle_track_id: str | None = None
    plate_track_ids: tuple[str, ...] = ()
    prepared_monotonic: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class PlateTrackState:
    key: LprTrackKey
    outcomes: list[RecognitionOutcome] = field(default_factory=list)
    seen_frames: set[float] = field(default_factory=set)
    seen_frame_order: deque[float] = field(default_factory=deque)
    committed_plates: set[str] = field(default_factory=set)
    committed_plate: str | None = None
    committed_strength: tuple[int, float, int] | None = None
    representative_plate: str | None = None
    object_box: tuple[int, int, int, int] | None = None
    switch_candidate: str | None = None
    switch_count: int = 0
    event_id: str | None = None
    last_seen: float | None = None


@dataclass(frozen=True, slots=True)
class PlateCommit:
    commit_id: str
    key: LprTrackKey
    event_id: str
    camera: str
    plate: str
    score: float
    sub_label: str | None
    timestamp: float
    frame_time: float
    plate_box: tuple[int, int, int, int]
    object_box: tuple[int, int, int, int] | None
    evidence_id: str
    frame_ref: str
    frame_width: int
    frame_height: int
    obj_data: dict[str, Any] | None
    dedicated_lpr: bool
    snapshot: str | None = None
    candidate_id: str = ""
    quality_score: float = 0.0
    quality_components: dict[str, float] = field(default_factory=dict)
    source_role: str = "detect"


@dataclass(frozen=True, slots=True)
class PlateActivity:
    """Dedicated-LPR heartbeat used only to keep its manual event alive."""

    event_id: str
    camera: str
    frame_time: float
    key: LprTrackKey


class LatestLprTaskQueue:
    """At most one pending frame per track, with non-droppable controls."""

    def __init__(self, max_tracks: int = 8) -> None:
        self.max_tracks = max_tracks
        self._pending: OrderedDict[LprTrackKey, LprFrameTask] = OrderedDict()
        self._controls: OrderedDict[tuple[str, str], LprExpireTask] = OrderedDict()
        self._generations: dict[tuple[str, str], int] = {}
        self._condition = threading.Condition()
        self.replaced = 0
        self.full_drops = 0

    def generation(self, camera: str, track_id: str) -> int:
        with self._condition:
            return self._generations.get((camera, track_id), 0)

    def is_current(self, key: LprTrackKey) -> bool:
        with self._condition:
            return (
                self._generations.get((key.camera, key.track_id), 0) == key.generation
            )

    def submit(self, task: LprFrameTask) -> bool:
        with self._condition:
            current = self._generations.get((task.key.camera, task.key.track_id), 0)
            if task.key.generation != current:
                return False
            if task.key in self._pending:
                previous = self._pending[task.key]
                if previous.collection_deadline > 0:
                    task.collection_deadline = previous.collection_deadline
                self._pending[task.key] = task
                self._pending.move_to_end(task.key)
                self.replaced += 1
                self._condition.notify()
                return True
            if len(self._pending) >= self.max_tracks:
                worst_key = min(
                    self._pending,
                    key=lambda pending_key: (
                        self._pending[pending_key].priority,
                        -self._pending[pending_key].enqueued_at,
                    ),
                )
                if self._pending[worst_key].priority >= task.priority:
                    self.full_drops += 1
                    return False
                self._pending.pop(worst_key)
                self.replaced += 1
            self._pending[task.key] = task
            self._condition.notify()
            return True

    def advance_generation(self, camera: str, track_id: str) -> int:
        """Invalidate queued/in-flight work and enqueue a priority reset."""
        with self._condition:
            identity = (camera, track_id)
            generation = self._generations.get(identity, 0) + 1
            self._generations[identity] = generation
            for key in list(self._pending):
                if key.camera == camera and key.track_id == track_id:
                    del self._pending[key]
            if ("", "") not in self._controls:
                identity = (camera, track_id)
                self._controls[identity] = LprExpireTask(camera, track_id, generation)
                self._controls.move_to_end(identity)
                if len(self._controls) > self.max_tracks:
                    # Generations are authoritative, so one bounded sweep control
                    # represents every invalidation without losing an expire.
                    self._controls.clear()
                    self._controls[("", "")] = LprExpireTask("", "", -1)
            self._condition.notify()
            return generation

    def rekey(self, observation: PlateObservation) -> PlateObservation:
        generation = self.advance_generation(
            observation.key.camera, observation.key.track_id
        )
        return replace(
            observation,
            key=LprTrackKey(
                observation.key.camera, observation.key.track_id, generation
            ),
        )

    def cancel(self, key: LprTrackKey) -> bool:
        """Cancel queued preparation without consuming an inference attempt."""
        with self._condition:
            removed = self._pending.pop(key, None) is not None
            if removed:
                self._condition.notify_all()
            return removed

    def contains(self, key: LprTrackKey) -> bool:
        with self._condition:
            return key in self._pending

    def get(self, timeout: float = 0.5) -> LprFrameTask | LprExpireTask | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._controls and not self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._controls:
                _, control = self._controls.popitem(last=False)
                return control
            while self._pending:
                now = time.monotonic()
                ready_keys = [
                    key
                    for key, task in self._pending.items()
                    if task.collection_deadline <= 0
                    or now >= task.collection_deadline
                ]
                ready_key = (
                    max(
                        ready_keys,
                        key=lambda key: (
                            self._pending[key].priority,
                            -self._pending[key].enqueued_at,
                        ),
                    )
                    if ready_keys
                    else None
                )
                if ready_key is not None:
                    return self._pending.pop(ready_key)
                nearest = min(
                    task.collection_deadline for task in self._pending.values()
                )
                remaining = min(deadline, nearest) - now
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
                if self._controls:
                    _, control = self._controls.popitem(last=False)
                    return control
            return None

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def control_depth(self) -> int:
        with self._condition:
            return len(self._controls)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()
