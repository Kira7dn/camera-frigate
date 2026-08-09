"""Embeddings types."""

from __future__ import annotations

from enum import Enum
from multiprocessing.managers import DictProxy, SyncManager, ValueProxy
from typing import Any

import sherpa_onnx

from frigate.data_processing.real_time.whisper_online import FasterWhisperASR


class DataProcessorMetrics:
    image_embeddings_speed: ValueProxy[float]
    image_embeddings_eps: ValueProxy[float]
    text_embeddings_speed: ValueProxy[float]
    text_embeddings_eps: ValueProxy[float]
    face_rec_speed: ValueProxy[float]
    face_rec_fps: ValueProxy[float]
    alpr_speed: ValueProxy[float]
    alpr_pps: ValueProxy[float]
    yolov9_lpr_speed: ValueProxy[float]
    yolov9_lpr_pps: ValueProxy[float]
    lpr_queue_depth: ValueProxy[float]
    lpr_queue_replaced: ValueProxy[float]
    lpr_queue_full_drops: ValueProxy[float]
    lpr_queue_ttl_drops: ValueProxy[float]
    lpr_stale_generation_drops: ValueProxy[float]
    lpr_task_age: ValueProxy[float]
    lpr_worker_latency: ValueProxy[float]
    evidence_frames: ValueProxy[float]
    evidence_bytes: ValueProxy[float]
    evidence_pinned: ValueProxy[float]
    evidence_time_evictions: ValueProxy[float]
    evidence_capacity_evictions: ValueProxy[float]
    evidence_pinned_capacity_drops: ValueProxy[float]
    evidence_misses: ValueProxy[float]
    evidence_camera_stats: Any
    quality_accepted: ValueProxy[float]
    quality_rejected: ValueProxy[float]
    quality_deduped: ValueProxy[float]
    quality_replaced: ValueProxy[float]
    quality_top_k_depth: ValueProxy[float]
    quality_reject_counts: dict[str, ValueProxy[float]]
    review_desc_speed: ValueProxy[float]
    review_desc_dps: ValueProxy[float]
    object_desc_speed: ValueProxy[float]
    object_desc_dps: ValueProxy[float]
    classification_speeds: DictProxy[str, ValueProxy[float]]
    classification_cps: DictProxy[str, ValueProxy[float]]

    def __init__(self, manager: SyncManager, custom_classification_models: list[str]):
        self.image_embeddings_speed = manager.Value("d", 0.0)
        self.image_embeddings_eps = manager.Value("d", 0.0)
        self.text_embeddings_speed = manager.Value("d", 0.0)
        self.text_embeddings_eps = manager.Value("d", 0.0)
        self.face_rec_speed = manager.Value("d", 0.0)
        self.face_rec_fps = manager.Value("d", 0.0)
        self.alpr_speed = manager.Value("d", 0.0)
        self.alpr_pps = manager.Value("d", 0.0)
        self.yolov9_lpr_speed = manager.Value("d", 0.0)
        self.yolov9_lpr_pps = manager.Value("d", 0.0)
        self.lpr_queue_depth = manager.Value("d", 0.0)
        self.lpr_queue_replaced = manager.Value("d", 0.0)
        self.lpr_queue_full_drops = manager.Value("d", 0.0)
        self.lpr_queue_ttl_drops = manager.Value("d", 0.0)
        self.lpr_stale_generation_drops = manager.Value("d", 0.0)
        self.lpr_task_age = manager.Value("d", 0.0)
        self.lpr_worker_latency = manager.Value("d", 0.0)
        self.evidence_frames = manager.Value("d", 0.0)
        self.evidence_bytes = manager.Value("d", 0.0)
        self.evidence_pinned = manager.Value("d", 0.0)
        self.evidence_time_evictions = manager.Value("d", 0.0)
        self.evidence_capacity_evictions = manager.Value("d", 0.0)
        self.evidence_pinned_capacity_drops = manager.Value("d", 0.0)
        self.evidence_misses = manager.Value("d", 0.0)
        self.evidence_camera_stats = manager.dict()
        self.quality_accepted = manager.Value("d", 0.0)
        self.quality_rejected = manager.Value("d", 0.0)
        self.quality_deduped = manager.Value("d", 0.0)
        self.quality_replaced = manager.Value("d", 0.0)
        self.quality_top_k_depth = manager.Value("d", 0.0)
        quality_reasons = (
            "detail_width_below_minimum",
            "detail_height_below_minimum",
            "blur_below_minimum",
            "underexposed",
            "overexposed",
            "frame_expired",
            "buffer_capacity",
            "top_k_not_selected",
        )
        self.quality_reject_counts = {
            f"{task}:{reason}": manager.Value("d", 0.0)
            for task in ("face", "lpr")
            for reason in quality_reasons
        }
        self.review_desc_speed = manager.Value("d", 0.0)
        self.review_desc_dps = manager.Value("d", 0.0)
        self.object_desc_speed = manager.Value("d", 0.0)
        self.object_desc_dps = manager.Value("d", 0.0)
        self.classification_speeds = manager.dict()
        self.classification_cps = manager.dict()

        if custom_classification_models:
            for key in custom_classification_models:
                self.classification_speeds[key] = manager.Value("d", 0.0)
                self.classification_cps[key] = manager.Value("d", 0.0)


class DataProcessorModelRunner:
    def __init__(self, requestor: Any, device: str = "CPU", model_size: str = "large"):
        self.requestor = requestor
        self.device = device
        self.model_size = model_size


class PostProcessDataEnum(str, Enum):
    recording = "recording"
    review = "review"
    tracked_object = "tracked_object"


AudioTranscriptionModel = FasterWhisperASR | sherpa_onnx.OnlineRecognizer | None
