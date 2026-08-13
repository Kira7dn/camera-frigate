"""Shared bootstrap for embedded and edge-owned camera lanes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent

from extension.topology.compiler import PlatformTopologyPlan

from frigate.const import (
    CACHE_DIR,
    CLIPS_DIR,
    CONFIG_DIR,
    FACE_DIR,
    MODEL_CACHE_DIR,
    RECORD_DIR,
    THUMB_DIR,
    TRIGGER_DIR,
)
from frigate.domain.object_detection.base import ObjectDetectProcess
from frigate.infrastructure.config import FrigateConfig
from frigate.models import Event, Recordings, Regions, ReviewSegment, Timeline
from frigate.util.image import UntrackedSharedMemory

logger = logging.getLogger(__name__)

# Every process hosting CameraMaintainer/RecordProcess needs these model bindings.
CAMERA_RUNTIME_MODELS = (Recordings, ReviewSegment, Regions)
# CameraMaintainer reuses Frigate's historical-region projection, which reads
# these local history models before processing the first frame.
CAMERA_HISTORY_MODELS = (Event, Timeline)


@dataclass(slots=True)
class CameraDetectorRuntime:
    processes: dict[str, ObjectDetectProcess]
    shared_memory: list[UntrackedSharedMemory]


def camera_runtime_config(
    config: FrigateConfig,
    *,
    edge_node_id: str | None = None,
    topology: PlatformTopologyPlan,
) -> FrigateConfig:
    """Return the camera ownership view from the already compiled topology."""
    return topology.camera_config(config, node_id=edge_node_id)


def ensure_runtime_dirs(config: FrigateConfig) -> None:
    """Create the same runtime roots before embedded or edge workers start."""
    directories = [
        CONFIG_DIR,
        THUMB_DIR,
        f"{CLIPS_DIR}/cache",
        CACHE_DIR,
        MODEL_CACHE_DIR,
    ]
    if any(camera.record.enabled for camera in config.cameras.values()):
        directories.append(RECORD_DIR)
    if config.face_recognition.enabled:
        directories.append(FACE_DIR)
    if config.semantic_search.enabled:
        directories.append(TRIGGER_DIR)

    for directory in directories:
        if not os.path.exists(directory) and not os.path.islink(directory):
            logger.info("Creating directory: %s", directory)
            os.makedirs(directory, exist_ok=True)
        else:
            logger.debug("Skipping directory: %s", directory)


def start_detector_runtime(
    config: FrigateConfig,
    detection_queue: Queue,
    stop_event: MpEvent,
    *,
    readiness_timeout: float = 60,
) -> CameraDetectorRuntime:
    """Start Frigate's detector processes for one already-owned camera view."""
    if not config.cameras:
        return CameraDetectorRuntime({}, [])

    largest_frame = max(
        (
            detector.model.height * detector.model.width * 3
            if detector.model is not None
            else 320
            for detector in config.detectors.values()
        ),
        default=320,
    )
    shared_memory: list[UntrackedSharedMemory] = []
    for name in config.cameras:
        for shm_name, size in (
            (name, largest_frame),
            (f"out-{name}", 20 * 6 * 4),
        ):
            try:
                shm = UntrackedSharedMemory(name=shm_name, create=True, size=size)
            except FileExistsError:
                shm = UntrackedSharedMemory(name=shm_name)
            shared_memory.append(shm)

    processes: dict[str, ObjectDetectProcess] = {}
    for name, detector_config in config.detectors.items():
        process = ObjectDetectProcess(
            name,
            detection_queue,
            list(config.cameras),
            config,
            detector_config,
            stop_event,
        )
        processes[name] = process

    for name, process in processes.items():
        if not process.ready_event.wait(timeout=readiness_timeout):
            raise RuntimeError(f"Detector {name} did not become ready before cameras")

    return CameraDetectorRuntime(processes, shared_memory)
