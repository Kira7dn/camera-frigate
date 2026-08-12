"""Shared tracked-object lifecycle projection for embedded and edge adapters."""

from __future__ import annotations

from typing import Any

from frigate.infrastructure.config import FrigateConfig
from frigate.domain.track.tracked_object import TrackedObject

from .policy import should_retain_recording, should_save_snapshot


def apply_media_policy(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> None:
    """Apply the original Frigate snapshot and recording decisions in place."""
    obj.has_snapshot = should_save_snapshot(config, camera, obj) or (
        obj.face_snapshot is not None
    )
    obj.has_clip = should_retain_recording(config, camera, obj)


def project_tracker_observation(
    obj: TrackedObject,
    *,
    frame_seq: int,
    motion_boxes: list[tuple[int, int, int, int]],
    regions: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    """Project existing Frigate state without recomputing tracker behavior."""
    observation = obj.to_dict()
    observation.update(
        {
            "frame_seq": frame_seq,
            "source_pts": int(float(observation["frame_time"]) * 1_000_000),
            "track_id": str(obj.obj_data["id"]),
            "score_history": tuple(obj.score_history),
            "path": tuple(point for point, _ in obj.path_data),
            "speed": obj.current_estimated_speed,
            "motion": {"boxes": tuple(motion_boxes)},
            "region": {"boxes": tuple(regions)},
        }
    )
    return observation


def publish_video_detection(
    publisher: Any,
    camera: str,
    frame_name: str,
    frame_time: float,
    tracked_objects: list[dict[str, Any]],
    motion_boxes: list[tuple[int, int, int, int]],
    regions: list[tuple[int, int, int, int]],
) -> None:
    """Publish the native payload consumed by recorder and live output."""
    from frigate.infrastructure.comms.detections_updater import DetectionTypeEnum

    publisher.publish(
        (
            camera,
            frame_name,
            frame_time,
            tracked_objects,
            motion_boxes,
            regions,
        ),
        DetectionTypeEnum.video.value,
    )
