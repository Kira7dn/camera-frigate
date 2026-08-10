"""Handle processing images for face detection and recognition."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.data_processing.common.license_plate.model import (
    LicensePlateModelRunner,
)

from frigate.config import FrigateConfig

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)

PENDING_ELIGIBILITY_MAX_FRAMES = 12
PENDING_ELIGIBILITY_MAX_SECONDS = 3.0


@dataclass
class PendingLprEligibility:
    """Canonical Event ownership waiting for an upstream-eligible frame."""

    camera: str
    object_id: str
    scheduled_frame_time: float
    last_attempt_frame_time: float
    attempts: int = 0


class LicensePlateRealTimeProcessor(LicensePlateProcessingMixin, RealTimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        model_runner: LicensePlateModelRunner,
        detected_license_plates: dict[str, dict[str, Any]],
    ):
        self.requestor = requestor
        self.detected_license_plates = detected_license_plates
        self.model_runner = model_runner
        self.lpr_config = config.lpr
        self.config = config
        self.sub_label_publisher = sub_label_publisher
        self.camera_current_cars: dict[str, list[str]] = {}
        self._pending_eligibility: dict[
            tuple[str, str], PendingLprEligibility
        ] = {}
        self._closed_retry_tracks: set[tuple[str, str]] = set()
        self._active_eligibility_retry: dict[str, Any] | None = None
        super().__init__(config, metrics)

    def _pending(self) -> dict[tuple[str, str], PendingLprEligibility]:
        """Return lazily initialized state for isolated tests using object.__new__."""
        pending = getattr(self, "_pending_eligibility", None)
        if pending is None:
            pending = {}
            self._pending_eligibility = pending
        return pending

    def _closed_retries(self) -> set[tuple[str, str]]:
        closed = getattr(self, "_closed_retry_tracks", None)
        if closed is None:
            closed = set()
            self._closed_retry_tracks = closed
        return closed

    @staticmethod
    def _blocked_by_initial_position_gate(obj_data: dict[str, Any]) -> bool:
        return (
            obj_data.get("position_changes", 0) == 0
            and not bool(obj_data.get("stationary", False))
        )

    def _supports_pending_retry(self, obj_data: dict[str, Any]) -> bool:
        camera = str(obj_data.get("camera") or "")
        object_id = str(obj_data.get("id") or "")
        label = obj_data.get("label")
        camera_config = getattr(self, "config", None)
        cameras = getattr(camera_config, "cameras", {})
        configured = cameras.get(camera) if camera else None
        return bool(
            camera
            and object_id
            and configured is not None
            and getattr(getattr(configured, "lpr", None), "enabled", False)
            and (
                label in getattr(self, "lp_objects", ())
                or label == "license_plate"
            )
        )

    def _schedule_pending_retry(self, obj_data: dict[str, Any]) -> None:
        camera = str(obj_data["camera"])
        object_id = str(obj_data["id"])
        frame_time = float(obj_data.get("frame_time") or 0.0)
        key = (camera, object_id)
        if key in self._pending() or key in self._closed_retries():
            return
        self._pending()[key] = PendingLprEligibility(
            camera=camera,
            object_id=object_id,
            scheduled_frame_time=frame_time,
            last_attempt_frame_time=frame_time,
        )
        from frigate.util.passage_trace import passage_trace

        passage_trace(
            "eligibility_retry_scheduled",
            camera=camera,
            frame_time=frame_time,
            track_id=object_id,
            object_box=obj_data.get("box"),
            reason="no_position_changes",
        )

    def _cancel_pending(
        self, camera: str, object_id: str, reason: str, frame_time: float | None = None
    ) -> bool:
        pending = self._pending().pop((camera, object_id), None)
        if pending is None:
            return False
        from frigate.util.passage_trace import passage_trace

        passage_trace(
            "eligibility_retry_cancelled",
            camera=camera,
            frame_time=frame_time,
            track_id=object_id,
            reason=reason,
            retry_attempts=pending.attempts,
            scheduled_frame_time=pending.scheduled_frame_time,
        )
        return True

    def has_pending_retry(self, camera: str) -> bool:
        """Return whether a canonical track on this camera is awaiting eligibility."""
        return any(key_camera == camera for key_camera, _ in self._pending())

    def reset_camera(self, camera: str) -> None:
        """Cancel retry state when a detection stream starts a new epoch."""
        for key_camera, object_id in list(self._pending()):
            if key_camera == camera:
                self._cancel_pending(camera, object_id, "stream_epoch_reset")
        self._closed_retry_tracks = {
            key for key in self._closed_retries() if key[0] != camera
        }

    def retry_pending_frame(
        self,
        camera: str,
        tracked_objects: list[dict[str, Any]],
        frame: np.ndarray,
        frame_time: float,
    ) -> int:
        """Retry only canonical-owned tracks using bbox and pixels from one frame."""
        current_time = float(frame_time)
        current_by_id = {
            str(obj["id"]): obj
            for obj in tracked_objects
            if obj.get("id") and obj.get("box")
        }
        retried = 0
        for key, pending in list(self._pending().items()):
            if pending.camera != camera:
                continue
            if current_time < pending.scheduled_frame_time - 1.0:
                self.reset_camera(camera)
                break
            if (
                current_time - pending.scheduled_frame_time
                > PENDING_ELIGIBILITY_MAX_SECONDS
                or pending.attempts >= PENDING_ELIGIBILITY_MAX_FRAMES
            ):
                self._cancel_pending(
                    camera, pending.object_id, "retry_budget_exhausted", current_time
                )
                self._closed_retries().add(key)
                continue
            if current_time <= pending.last_attempt_frame_time:
                continue
            obj = current_by_id.get(pending.object_id)
            if obj is None:
                continue

            retry_obj = dict(obj)
            retry_obj["camera"] = camera
            retry_obj["frame_time"] = current_time
            pending.last_attempt_frame_time = current_time
            pending.attempts += 1
            retry_context = {
                "scheduled_frame_time": pending.scheduled_frame_time,
                "retry_frame_time": current_time,
                "retry_index": pending.attempts,
                "retry_object_box": retry_obj.get("box"),
            }
            from frigate.util.passage_trace import passage_trace

            passage_trace(
                "eligibility_retry_attempted",
                camera=camera,
                frame_time=current_time,
                track_id=pending.object_id,
                object_box=retry_obj.get("box"),
                **retry_context,
            )
            became_eligible = not self._blocked_by_initial_position_gate(retry_obj)
            if became_eligible:
                self._pending().pop(key, None)
                self._closed_retries().add(key)

            self._active_eligibility_retry = retry_context
            try:
                self.lpr_process(retry_obj, frame, False)
            finally:
                self._active_eligibility_retry = None
            retried += 1

            if became_eligible:
                passage_trace(
                    "eligibility_retry_resolved",
                    camera=camera,
                    frame_time=current_time,
                    track_id=pending.object_id,
                    object_box=retry_obj.get("box"),
                    retry_index=pending.attempts,
                    scheduled_frame_time=pending.scheduled_frame_time,
                )
        return retried

    CONFIG_UPDATE_TOPIC = "config/lpr"

    def update_config(self, topic: str, payload: Any) -> None:
        """Update LPR config at runtime."""
        if topic != self.CONFIG_UPDATE_TOPIC:
            return

        previous_min_area = self.config.lpr.min_area
        self.config.lpr = payload
        self.lpr_config = payload

        for camera_config in self.config.cameras.values():
            if camera_config.lpr.min_area == previous_min_area:
                camera_config.lpr.min_area = payload.min_area

        logger.debug("LPR config updated dynamically")

    def process_frame(
        self,
        obj_data: dict[str, Any],
        frame: np.ndarray,
        dedicated_lpr: bool = False,
    ) -> None:
        """Look for license plates in image."""
        self.lpr_process(obj_data, frame, dedicated_lpr)
        if dedicated_lpr or not isinstance(obj_data, dict):
            return
        if not self._supports_pending_retry(obj_data):
            return
        camera = str(obj_data["camera"])
        object_id = str(obj_data["id"])
        if self._blocked_by_initial_position_gate(obj_data):
            self._schedule_pending_retry(obj_data)
        else:
            self._closed_retries().add((camera, object_id))
            self._cancel_pending(
                camera,
                object_id,
                "canonical_event_became_eligible",
                float(obj_data.get("frame_time") or 0.0),
            )

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        """Expire lpr objects."""
        self._cancel_pending(camera, object_id, "event_end")
        self._closed_retries().discard((camera, object_id))
        self.lpr_expire(object_id, camera)

    def shutdown(self) -> None:
        """Cancel retry-only ownership without running recognition."""
        for camera, object_id in list(self._pending()):
            self._cancel_pending(camera, object_id, "shutdown")
        self._closed_retries().clear()
