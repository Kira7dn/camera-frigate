"""Single producer core used by embedded and managed-edge adapters.

This module deliberately consumes the output of the existing camera pipeline. It does
not implement detection, Norfair association, zone, speed, recorder, or PTZ a second
time; those owners feed their canonical observation into this envelope builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import (
    BoundingBox,
    EvidenceReference,
    LifecycleFailure,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
)
from .journal import EdgeJournal


@dataclass(frozen=True, slots=True)
class ProducerContext:
    node_id: str
    node_epoch: str
    camera_id: str
    stream_epoch: str


class TrackerProducerCore:
    """Normalize one existing-pipeline observation into the frozen wire contract."""

    def __init__(
        self, context: ProducerContext, journal: EdgeJournal | None = None
    ) -> None:
        self.context = context
        self.journal = journal

    def emit(
        self,
        observation: dict[str, Any],
        *,
        operation: TrackerOperation,
        event_id: str,
        evidence: EvidenceReference | None = None,
        media: tuple[MediaManifest, ...] = (),
        failure: LifecycleFailure | None = None,
    ) -> TrackerUpdate:
        box = observation["box"]
        update = TrackerUpdate(
            node_id=self.context.node_id,
            node_epoch=self.context.node_epoch,
            camera_id=self.context.camera_id,
            stream_epoch=self.context.stream_epoch,
            journal_sequence=0,
            frame_seq=int(observation["frame_seq"]),
            source_pts=int(observation.get("source_pts", 0)),
            frame_time=float(observation["frame_time"]),
            event_id=event_id,
            track_id=str(observation["track_id"]),
            operation=operation,
            label=str(observation["label"]),
            score_history=tuple(float(v) for v in observation.get("score_history", ())),
            score=float(observation["score"]),
            bbox=BoundingBox(*(int(v) for v in box)),
            attributes=dict(observation.get("attributes", {})),
            current_zones=tuple(observation.get("current_zones", ())),
            entered_zones=tuple(observation.get("entered_zones", ())),
            path=tuple(tuple(point) for point in observation.get("path", ())),
            speed=observation.get("speed"),
            motion=dict(observation.get("motion", {})),
            region=dict(observation.get("region", {})),
            evidence=evidence,
            failure=failure,
            media=media,
        )
        return self.journal.append(update) if self.journal is not None else update
