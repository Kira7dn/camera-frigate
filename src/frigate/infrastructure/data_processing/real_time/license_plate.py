"""Canonical tracked-object LPR adapter for the standalone recognition core."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

import numpy as np
from frigate.infrastructure.comms.event_metadata_updater import EventMetadataPublisher
from frigate.infrastructure.comms.inter_process import InterProcessRequestor
from frigate.infrastructure.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.infrastructure.data_processing.common.license_plate.model import LicensePlateModelRunner
from frigate.application.recognition.adapters.frigate import (
    BorrowedEvidenceResolver,
    FrigateEventAdapter,
    FrigateRecognitionAdapter,
)
from frigate.application.recognition.contracts import RecognitionTask
from frigate.application.recognition.core import RecognitionCore
from frigate.application.recognition.face import FacePolicy
from frigate.application.recognition.lpr import LprPolicy
from frigate.application.recognition.ports import RawRecognition
from frigate.util.passage_trace import passage_trace
from rapidfuzz.distance import Levenshtein

from frigate.infrastructure.config import FrigateConfig

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


class LicensePlateRealTimeProcessor(LicensePlateProcessingMixin, RealTimeProcessorApi):
    CONFIG_UPDATE_TOPIC = "config/lpr"

    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        model_runner: LicensePlateModelRunner,
        stream_epoch: str = "process",
    ) -> None:
        self.requestor = requestor
        self.model_runner = model_runner
        self.lpr_config = config.lpr
        self.config = config
        self.sub_label_publisher = sub_label_publisher
        self.stream_epoch = stream_epoch
        self._recognition_adapters: dict[str, FrigateRecognitionAdapter] = {}
        super().__init__(config, metrics)

    def _known_plate_label(self, plate: str) -> str | None:
        try:
            return next(
                (
                    label
                    for label, patterns in (self.lpr_config.known_plates or {}).items()
                    if any(
                        re.match(f"^{pattern}$", plate)
                        or Levenshtein.distance(pattern, plate)
                        <= self.lpr_config.match_distance
                        for pattern in patterns
                    )
                ),
                None,
            )
        except re.error:
            logger.error(
                "Invalid regex in known plates configuration: %s",
                self.lpr_config.known_plates,
            )
            return None

    def _recognition_adapter(self, camera: str) -> FrigateRecognitionAdapter:
        adapter = self._recognition_adapters.get(camera)
        if adapter is not None:
            return adapter

        evidence = BorrowedEvidenceResolver()
        event_adapter = FrigateEventAdapter(
            lambda payload: self.requestor.send_data(
                "tracked_object_update", json.dumps(payload)
            ),
            lambda kind, payload: self.sub_label_publisher.publish(payload, kind),
            known_plate_label=self._known_plate_label,
        )
        core = RecognitionCore(
            self,
            evidence,
            LprPolicy(
                detect_fps=self.config.cameras[camera].detect.fps,
                recognition_threshold=self.lpr_config.recognition_threshold,
                min_plate_length=self.lpr_config.min_plate_length,
                plate_format=self.lpr_config.format,
            ),
            FacePolicy(
                unknown_score=self.config.face_recognition.unknown_score,
                recognition_threshold=self.config.face_recognition.recognition_threshold,
                min_faces=self.config.face_recognition.min_faces,
            ),
            event_adapter,
        )
        adapter = FrigateRecognitionAdapter(core, self.stream_epoch, evidence)
        self._recognition_adapters[camera] = adapter
        return adapter

    @property
    def recognition_stats(self) -> dict[str, int]:
        totals = {"sessions": 0, "in_flight": 0, "evidence_pinned": 0}
        for adapter in self._recognition_adapters.values():
            for name in totals:
                totals[name] += adapter.stats[name]
        return totals

    def recognize(
        self, task: RecognitionTask, observation, evidence: object
    ) -> RawRecognition | None:
        """Run preprocessing, plate detection and OCR for one core observation."""
        if task is not RecognitionTask.LPR:
            return None
        if not isinstance(evidence, tuple) or len(evidence) != 2:
            return None
        obj_data, frame = evidence
        result = self.lpr_process(cast(Any, obj_data), cast(Any, frame), False)
        return result if isinstance(result, RawRecognition) else None

    def update_config(self, topic: str, payload: Any) -> None:
        if topic != self.CONFIG_UPDATE_TOPIC:
            return
        previous_min_area = self.config.lpr.min_area
        self.config.lpr = payload
        self.lpr_config = payload
        for camera_config in self.config.cameras.values():
            if camera_config.lpr.min_area == previous_min_area:
                camera_config.lpr.min_area = payload.min_area
        for adapter in self._recognition_adapters.values():
            adapter.shutdown()
        self._recognition_adapters.clear()
        logger.debug("LPR config updated and sessions reset")

    def process_frame(
        self,
        obj_data: Any,
        frame: Any,
        dedicated_lpr: bool = False,
        **kwargs: Any,
    ) -> None:
        """Recognize LPR only from canonical caller-owned track IDs."""
        if dedicated_lpr:
            logger.error(
                "Dedicated untracked LPR is not supported by the track contract"
            )
            return
        if not isinstance(obj_data, dict) or not obj_data.get("box"):
            return
        camera = str(obj_data["camera"])
        frame_time = float(obj_data["frame_time"])
        track_id = str(obj_data["id"])
        evidence_ref = f"lpr:{camera}:{track_id}:{frame_time:.6f}"
        updates = self._recognition_adapter(camera).observe(
            RecognitionTask.LPR,
            obj_data,
            frame_time,
            evidence_ref,
            evidence=(obj_data, frame),
            observed_in_frame=obj_data.get("observed_in_frame"),
            attributes={"current_attributes": obj_data.get("current_attributes", ())},
        )
        for update in updates:
            if not update.publish:
                continue
            passage_trace(
                "event_published",
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                trace_id=f"lpr:{camera}:{track_id}",
                plate=update.aggregate_value,
                score=update.aggregate_score,
                plate_box=update.detail_bbox,
                object_box=update.object_bbox,
                evidence_id=str(update.evidence_ref),
                frame_ref=str(update.evidence_ref),
                source_role="detect",
            )

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        adapter = self._recognition_adapters.get(camera)
        if adapter is not None:
            adapter.end_track(camera, object_id, "event_end")

    def shutdown(self) -> None:
        for adapter in self._recognition_adapters.values():
            adapter.shutdown()
        self._recognition_adapters.clear()
