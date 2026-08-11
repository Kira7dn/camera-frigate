"""Thin mapping between Frigate tracked objects and recognition contracts.

The adapter intentionally accepts callbacks instead of importing Frigate comms or
Event classes. This keeps decision ownership in the core and makes the boundary
testable without starting the Frigate runtime.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from ..contracts import (
    BBox,
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
)
from ..core import RecognitionCore

logger = logging.getLogger(__name__)


class BorrowedEvidenceResolver:
    """Borrow evidence only for the duration of one synchronous ``observe`` call."""

    def __init__(self) -> None:
        self._borrowed: dict[object, object] = {}

    @contextmanager
    def borrow(self, evidence_ref: object, evidence: object):
        if evidence_ref in self._borrowed:
            raise RuntimeError("evidence reference is already borrowed")
        self._borrowed[evidence_ref] = evidence
        try:
            yield
        finally:
            self._borrowed.pop(evidence_ref, None)

    def resolve(self, observation: TrackedObservation):
        @contextmanager
        def resolved():
            if observation.evidence_ref not in self._borrowed:
                raise RuntimeError("evidence reference is not borrowed")
            yield self._borrowed[observation.evidence_ref]

        return resolved()

    def stats(self) -> dict[str, int]:
        return {"pinned": len(self._borrowed)}


class FrigateRecognitionAdapter:
    def __init__(
        self,
        core: RecognitionCore,
        stream_epoch: str,
        evidence_resolver: BorrowedEvidenceResolver,
    ) -> None:
        self._core = core
        self._stream_epoch = stream_epoch
        self._evidence_resolver = evidence_resolver

    def observe(
        self,
        task: RecognitionTask,
        obj_data: dict[str, Any],
        frame_time: float,
        evidence_ref: object,
        *,
        evidence: object,
        detail_bbox: BBox | None = None,
        observed_in_frame: bool | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> tuple[RecognitionUpdate, ...]:
        key = self.key_for(obj_data)
        observation = TrackedObservation(
            task=task,
            key=key,
            frame_time=frame_time,
            object_bbox=tuple(obj_data["box"]),
            detail_bbox=detail_bbox,
            observed_in_frame=observed_in_frame,
            evidence_ref=evidence_ref,
            attributes={
                "label": obj_data.get("label"),
                "sub_label": obj_data.get("sub_label"),
                **(attributes or {}),
            },
        )
        with self._evidence_resolver.borrow(evidence_ref, evidence):
            return self._core.observe(observation)

    def key_for(self, obj_data: dict[str, Any]) -> TrackKey:
        return TrackKey(
            camera_id=str(obj_data["camera"]),
            stream_epoch=self._stream_epoch,
            track_id=str(obj_data["id"]),
        )

    def end_track(self, camera_id: str, track_id: str, reason: str) -> None:
        self._core.end_track(
            TrackKey(camera_id, self._stream_epoch, str(track_id)), reason
        )

    @property
    def stats(self) -> dict[str, int]:
        return self._core.stats

    def shutdown(self) -> None:
        self._core.shutdown()


class FrigateEventAdapter:
    """Publish core updates immediately using Frigate-shaped callback payloads."""

    def __init__(
        self,
        tracked_object_update: Callable[[dict[str, Any]], None],
        metadata_update: Callable[[str, tuple[Any, ...]], None],
        *,
        known_plate_label: Callable[[str], str | None] | None = None,
        snapshot_after_decision: Callable[[RecognitionUpdate], None] | None = None,
        track_end: Callable[[RecognitionTask, TrackKey, str], None] | None = None,
    ) -> None:
        self._tracked_object_update = tracked_object_update
        self._metadata_update = metadata_update
        self._known_plate_label = known_plate_label or (lambda _plate: None)
        self._snapshot_after_decision = snapshot_after_decision
        self._track_end = track_end

    def on_update(self, update: RecognitionUpdate) -> None:
        if update.task is RecognitionTask.FACE:
            self._publish_face(update)
        else:
            self._publish_lpr(update)
        if update.publish and self._snapshot_after_decision is not None:
            try:
                self._snapshot_after_decision(update)
            except Exception:
                logger.exception("Recognition media enqueue failed after publication")

    def _publish_face(self, update: RecognitionUpdate) -> None:
        self._tracked_object_update(
            {
                "type": "face",
                "name": update.aggregate_value,
                "score": update.aggregate_score,
                "id": update.key.track_id,
                "camera": update.key.camera_id,
                "timestamp": update.frame_time,
                "bbox": update.object_bbox,
                "detail_bbox": update.detail_bbox,
                "evidence_ref": update.evidence_ref,
            }
        )
        if update.publish:
            self._metadata_update(
                "sub_label",
                (
                    update.key.track_id,
                    update.aggregate_value,
                    update.aggregate_score,
                ),
            )

    def _publish_lpr(self, update: RecognitionUpdate) -> None:
        plate = update.aggregate_value
        if plate is None or not update.publish:
            return
        label = self._known_plate_label(plate)
        if label is not None:
            self._metadata_update(
                "sub_label",
                (update.key.track_id, label, update.aggregate_score),
            )
        self._tracked_object_update(
            {
                "type": "lpr",
                "name": label,
                "plate": plate,
                "score": update.aggregate_score,
                "id": update.key.track_id,
                "camera": update.key.camera_id,
                "timestamp": update.frame_time,
                "bbox": update.object_bbox,
                "plate_box": update.detail_bbox,
                "evidence_ref": update.evidence_ref,
            }
        )
        self._metadata_update(
            "attribute",
            (
                update.key.track_id,
                "recognized_license_plate",
                plate,
                update.aggregate_score,
            ),
        )

    def on_track_end(self, task: RecognitionTask, key: TrackKey, reason: str) -> None:
        if self._track_end is not None:
            self._track_end(task, key, reason)
