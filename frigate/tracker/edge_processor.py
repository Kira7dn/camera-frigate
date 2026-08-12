"""Edge adapter around Frigate's existing CameraState behavior.

Detection, Norfair association, score history, filtering, zones, speed/path and
autotracking remain implemented by the original Frigate classes. This adapter only
converts their callbacks into durable producer-owned envelopes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from frigate.camera.state import CameraState
from frigate.config import FrigateConfig
from frigate.ptz.autotrack import PtzAutoTrackerThread
from frigate.track.tracked_object import TrackedObject
from frigate.util.image import SharedMemoryFrameManager

from .contracts import LifecycleFailure, TrackerOperation, TrackerUpdate
from .evidence import EvidenceCapacityError, EvidenceRing
from .journal import SpoolFullError
from .producer import ProducerContext, TrackerProducerCore


class EdgeTrackedObjectProcessor:
    """Feed original Frigate camera-state callbacks to one ordered edge producer."""

    def __init__(
        self,
        config: FrigateConfig,
        context: ProducerContext,
        producer: TrackerProducerCore,
        evidence: EvidenceRing,
        ptz_autotracker_thread: PtzAutoTrackerThread,
        publish: Callable[[TrackerUpdate], None],
    ) -> None:
        self.config = config
        self.context = context
        self.producer = producer
        self.evidence = evidence
        self.publish = publish
        self.frame_manager = SharedMemoryFrameManager()
        self.frame_seq = 0
        self.motion_boxes: list[tuple[int, int, int, int]] = []
        self.regions: list[tuple[int, int, int, int]] = []
        self._event_ids: dict[str, str] = {}
        self._last_objects: dict[str, TrackedObject] = {}
        self.camera_state = CameraState(
            context.camera_id,
            config,
            self.frame_manager,
            ptz_autotracker_thread,
        )
        self._ptz = ptz_autotracker_thread.ptz_autotracker
        self.camera_state.on("start", self._start)
        self.camera_state.on("update", self._update)
        self.camera_state.on("end", self._end)
        self.camera_state.on("autotrack", self._autotrack)

    def process(
        self,
        frame_name: str,
        frame_time: float,
        current_tracked_objects: dict[str, dict[str, Any]],
        motion_boxes: list[tuple[int, int, int, int]],
        regions: list[tuple[int, int, int, int]],
    ) -> None:
        self.frame_seq += 1
        self.motion_boxes = motion_boxes
        self.regions = regions
        self.camera_state.update(
            frame_name,
            frame_time,
            current_tracked_objects,
            motion_boxes,
            regions,
        )

    def _start(
        self,
        camera: str,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
    ) -> None:
        raw_track_id = str(obj.obj_data["id"])
        self._event_ids[raw_track_id] = uuid.uuid4().hex[:30]
        self._last_objects[raw_track_id] = obj
        self._emit(TrackerOperation.START, obj, frame_name, observed_in_frame)

    def _update(
        self,
        camera: str,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
    ) -> None:
        self._last_objects[str(obj.obj_data["id"])] = obj
        self._emit(TrackerOperation.UPDATE, obj, frame_name, observed_in_frame)

    def _end(
        self,
        camera: str,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
    ) -> None:
        self._emit(TrackerOperation.END, obj, frame_name, observed_in_frame)
        raw_track_id = str(obj.obj_data["id"])
        if not obj.false_positive:
            self._ptz.end_object(camera, obj)
        self._event_ids.pop(raw_track_id, None)
        self._last_objects.pop(raw_track_id, None)

    def _autotrack(
        self,
        camera: str,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
    ) -> None:
        self._ptz.autotrack_object(camera, obj)

    def _emit(
        self,
        operation: TrackerOperation,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
        failure: LifecycleFailure | None = None,
    ) -> None:
        raw_track_id = str(obj.obj_data["id"])
        try:
            evidence = (
                self._capture_evidence(frame_name) if observed_in_frame else None
            )
        except EvidenceCapacityError:
            evidence = None
            failure = failure or LifecycleFailure(
                "evidence_capacity", "bounded evidence ring is full", True
            )
        durable_reference = evidence
        if evidence is not None and self.producer.journal is not None:
            _, data, shape = self.evidence.get(evidence.evidence_id)
            self.producer.journal.pin_evidence(evidence, data, shape)
            durable_reference = replace(evidence, durable=True)
        observation = obj.to_dict()
        observation.update(
            {
                "frame_seq": self.frame_seq,
                "source_pts": int(float(observation["frame_time"]) * 1_000_000),
                "track_id": raw_track_id,
                "score_history": obj.score_history,
                "path": [point for point, _ in obj.path_data],
                "speed": obj.current_estimated_speed,
                "motion": {"boxes": self.motion_boxes},
                "region": {"boxes": self.regions},
            }
        )
        try:
            update = self.producer.emit(
                observation,
                operation=operation,
                event_id=self._event_ids[raw_track_id],
                evidence=durable_reference,
                failure=failure,
            )
        except (SpoolFullError, ValueError):
            if evidence is not None and self.producer.journal is not None:
                self.producer.journal.release_evidence(evidence.evidence_id)
            raise
        self.publish(update)

    def _capture_evidence(self, frame_name: str):
        camera_config = self.config.cameras[self.context.camera_id]
        frame = self.frame_manager.get(frame_name, camera_config.frame_shape_yuv)
        if frame is None:
            return None
        array = np.asarray(frame, dtype=np.uint8)
        return self.evidence.put(
            stream_epoch=self.context.stream_epoch,
            frame_seq=self.frame_seq,
            data=array.tobytes(),
            shape=(array.shape[0] * 2 // 3, array.shape[1]),
        )

    def fail_active(self, code: str, detail: str, *, retryable: bool) -> None:
        """End only active producer state; never invent tracks after a transport fault."""
        failure = LifecycleFailure(code, detail, retryable)
        for obj in tuple(self._last_objects.values()):
            self._emit(TrackerOperation.END, obj, "", False, failure)
        self._event_ids.clear()
        self._last_objects.clear()

    def terminal_state(self) -> dict[str, int]:
        return {
            "active": len(self._last_objects),
            "pinned_evidence": self.evidence.pinned_count,
        }
