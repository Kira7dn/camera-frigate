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

from frigate.domain.camera.state import CameraState
from frigate.infrastructure.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.infrastructure.config import FrigateConfig
from frigate.domain.ptz.autotrack import PtzAutoTrackerThread
from frigate.domain.track.tracked_object import TrackedObject
from frigate.util.image import SharedMemoryFrameManager

from camera_platform.tracker.domain.lifecycle import (
    apply_media_policy,
    project_tracker_observation,
    publish_video_detection,
)

from ..domain.contracts import LifecycleFailure, MediaManifest, TrackerOperation, TrackerUpdate
from .evidence import EvidenceCapacityError, EvidenceRing
from .journal import SpoolFullError
from .media import MediaAuthority
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
        media: MediaAuthority | None = None,
        detection_publisher: Any | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.producer = producer
        self.evidence = evidence
        self.publish = publish
        self.media = media
        self.detection_publisher = detection_publisher or DetectionPublisher(
            DetectionTypeEnum.all.value
        )
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
        publish_video_detection(
            self.detection_publisher,
            self.context.camera_id,
            frame_name,
            frame_time,
            [obj.to_dict() for obj in self.camera_state.tracked_objects.values()],
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
        apply_media_policy(self.config, camera, obj)
        self._last_objects[str(obj.obj_data["id"])] = obj
        self._emit(TrackerOperation.UPDATE, obj, frame_name, observed_in_frame)

    def _end(
        self,
        camera: str,
        obj: TrackedObject,
        frame_name: str,
        observed_in_frame: bool,
    ) -> None:
        apply_media_policy(self.config, camera, obj)
        raw_track_id = str(obj.obj_data["id"])
        snapshot_jpg = None
        if obj.has_snapshot and self.media is not None:
            snapshots = self.config.cameras[camera].snapshots
            snapshot_jpg, _ = obj.get_img_bytes(
                ext="jpg",
                timestamp=snapshots.timestamp,
                bounding_box=snapshots.bounding_box,
                crop=snapshots.crop,
                height=snapshots.height,
                quality=snapshots.quality,
            )
        if obj.has_snapshot or obj.has_clip:
            obj.write_thumbnail_to_disk()
        if obj.has_snapshot:
            obj.write_snapshot_to_disk()
        manifests: tuple[MediaManifest, ...] = ()
        if self.media is not None:
            output = list(self.media.register_event_files(
                camera_id=camera,
                raw_track_id=raw_track_id,
                event_id=self._event_ids[raw_track_id],
                frame_time=float(obj.obj_data["frame_time"]),
                has_snapshot=obj.has_snapshot,
            ))
            if snapshot_jpg is not None:
                output.append(
                    self.media.register(
                        media_id=f"snapshot-jpg-{self._event_ids[raw_track_id]}",
                        event_id=self._event_ids[raw_track_id],
                        camera_id=camera,
                        data=snapshot_jpg,
                        start_time=float(obj.obj_data["frame_time"]),
                        end_time=float(obj.obj_data["frame_time"]),
                        codec="jpeg",
                        media_type="snapshot_jpg",
                        ttl_seconds=3600,
                    )
                )
            if obj.has_clip:
                clip = self.media.register_event_clip(
                    config=self.config,
                    camera_id=camera,
                    event_id=self._event_ids[raw_track_id],
                    start_time=float(obj.obj_data["start_time"]),
                    end_time=float(obj.obj_data["frame_time"]),
                )
                if clip is not None:
                    output.append(clip)
            manifests = tuple(output)
        self._emit(
            TrackerOperation.END,
            obj,
            frame_name,
            observed_in_frame,
            media=manifests,
        )
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
        media: tuple[MediaManifest, ...] = (),
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
        observation = project_tracker_observation(
            obj,
            frame_seq=self.frame_seq,
            motion_boxes=self.motion_boxes,
            regions=self.regions,
        )
        try:
            update = self.producer.emit(
                observation,
                operation=operation,
                event_id=self._event_ids[raw_track_id],
                evidence=durable_reference,
                failure=failure,
                media=media,
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

    def close(self) -> None:
        self.detection_publisher.stop()
