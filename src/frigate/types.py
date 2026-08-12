from enum import Enum
from typing import Any, TypedDict

from frigate.infrastructure.data_processing.types import DataProcessorMetrics
from frigate.domain.object_detection.base import ObjectDetectProcess


class StatsTrackingTypes(TypedDict):
    # multiprocessing.Manager returns a DictProxy, not a builtin dict. Keep the
    # IPC proxy intact so stats always observe live camera metric updates.
    camera_metrics: Any
    embeddings_metrics: DataProcessorMetrics
    detectors: dict[str, ObjectDetectProcess]
    started: int
    latest_frigate_version: str
    last_updated: int
    processes: dict[str, int]


class ModelStatusTypesEnum(str, Enum):
    not_downloaded = "not_downloaded"
    downloading = "downloading"
    downloaded = "downloaded"
    error = "error"
    training = "training"
    complete = "complete"
    failed = "failed"


class JobStatusTypesEnum(str, Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"


class TrackedObjectUpdateTypesEnum(str, Enum):
    description = "description"
    face = "face"
    lpr = "lpr"
    classification = "classification"
