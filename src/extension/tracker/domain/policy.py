"""Shared object media-retention policy from the original Frigate event path."""

from __future__ import annotations

import logging

from frigate.infrastructure.config import FrigateConfig
from frigate.domain.track.tracked_object import TrackedObject

logger = logging.getLogger(__name__)


def should_save_snapshot(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> bool:
    """Preserve the existing Frigate snapshot decision exactly."""
    if obj.false_positive:
        return False
    snapshot_config = config.cameras[camera].snapshots
    if not snapshot_config.enabled:
        return False
    if obj.obj_data["position_changes"] == 0:
        return False
    required_zones = snapshot_config.required_zones
    if required_zones and not set(obj.entered_zones) & set(required_zones):
        logger.debug(
            "Not creating snapshot for %s because it did not enter required zones",
            obj.obj_data["id"],
        )
        return False
    return True


def should_retain_recording(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> bool:
    """Preserve the existing Frigate recording decision exactly."""
    if obj.false_positive:
        return False
    record_config = config.cameras[camera].record
    if not record_config.enabled:
        return False
    if obj.obj_data["position_changes"] == 0:
        return False
    return obj.max_severity is not None
