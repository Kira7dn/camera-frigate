"""Embeddings types."""

from __future__ import annotations

from enum import Enum
from multiprocessing.managers import DictProxy, SyncManager, ValueProxy
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
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
    recognition_sessions: ValueProxy[float]
    recognition_in_flight: ValueProxy[float]
    recognition_evidence_pinned: ValueProxy[float]
    recognition_writer_depth: ValueProxy[float]
    recognition_writer_drops: ValueProxy[float]
    recognition_writer_errors: ValueProxy[float]
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
        self.recognition_sessions = manager.Value("d", 0.0)
        self.recognition_in_flight = manager.Value("d", 0.0)
        self.recognition_evidence_pinned = manager.Value("d", 0.0)
        self.recognition_writer_depth = manager.Value("d", 0.0)
        self.recognition_writer_drops = manager.Value("d", 0.0)
        self.recognition_writer_errors = manager.Value("d", 0.0)
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


if TYPE_CHECKING:
    AudioTranscriptionModel: TypeAlias = (
        FasterWhisperASR | sherpa_onnx.OnlineRecognizer | None
    )
else:
    AudioTranscriptionModel = Any
