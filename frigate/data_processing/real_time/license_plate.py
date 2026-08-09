"""Run realtime license plate recognition outside the maintainer call path."""

from __future__ import annotations

import base64
import datetime
import hashlib
import logging
import os
import random
import re
import string
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from rapidfuzz.distance import Levenshtein

from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import CLIPS_DIR
from frigate.data_processing.common.evidence import EvidenceRingBuffer, FrameRef
from frigate.data_processing.common.license_plate.association import (
    LprPassageAdmission,
    LprPassageRegistry,
    LprPassageRejection,
    associate_lpr_passages,
)
from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.data_processing.common.license_plate.pipeline import (
    LatestLprTaskQueue,
    LprExpireTask,
    LprFrameTask,
    LprTrackKey,
    PlateActivity,
    PlateCommit,
    PlateObservation,
    PlateTrackState,
    PreparedPlateCandidate,
)
from frigate.data_processing.common.quality import QualitySelector
from frigate.data_processing.common.recognition import (
    RecognitionKey,
    RecognitionLifecycle,
    RecognitionPolicy,
    RecognitionStatus,
)
from frigate.util.passage_trace import passage_trace

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

if TYPE_CHECKING:
    from frigate.data_processing.common.license_plate.model import (
        LicensePlateModelRunner,
    )

logger = logging.getLogger(__name__)


class LicensePlateRealTimeProcessor(LicensePlateProcessingMixin, RealTimeProcessorApi):
    MAX_TRACKS = 8
    MAX_STATES = 64
    MAX_RESULTS = 64
    TASK_TTL = 1.0

    def _get_recognition_lifecycle(self) -> RecognitionLifecycle:
        lifecycle = getattr(self, "recognition_lifecycle", None)
        if lifecycle is None:
            lifecycle = RecognitionLifecycle()
            self.recognition_lifecycle = lifecycle
        return lifecycle

    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        model_runner: LicensePlateModelRunner,
        detected_license_plates: dict[str, dict[str, Any]],
        evidence_ring: EvidenceRingBuffer | None = None,
        quality_selector: QualitySelector | None = None,
        recognition_lifecycle: RecognitionLifecycle | None = None,
    ):
        self.requestor = requestor
        # This compatibility view is updated by EmbeddingMaintainer after commit.
        self.detected_license_plates = detected_license_plates
        self.model_runner = model_runner
        self.lpr_config = config.lpr
        self.config = config
        self.sub_label_publisher = sub_label_publisher
        self.camera_current_cars: dict[str, list[str]] = {}
        self.evidence_ring = evidence_ring
        self.quality_selector = quality_selector
        self.recognition_lifecycle = recognition_lifecycle or RecognitionLifecycle()
        self._passage_registry = LprPassageRegistry()
        super().__init__(config, metrics)

        self._tasks = LatestLprTaskQueue(self.MAX_TRACKS)
        self._states: OrderedDict[LprTrackKey, PlateTrackState] = OrderedDict()
        self._prepared: dict[LprTrackKey, list[PreparedPlateCandidate]] = {}
        self._collection_started: dict[LprTrackKey, float] = {}
        self._last_prepared_monotonic: dict[LprTrackKey, float] = {}
        self._last_seen_monotonic: dict[LprTrackKey, float] = {}
        self._terminal_keys: set[LprTrackKey] = set()
        self._active_detection_ids: dict[str, set[str]] = {}
        self._results: deque[PlateCommit | PlateActivity] = deque()
        self._results_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="lpr_realtime_worker",
        )
        self._worker.start()

    CONFIG_UPDATE_TOPIC = "config/lpr"

    def _set_metric(self, name: str, value: float) -> None:
        metric = getattr(self.metrics, name, None)
        if metric is not None:
            metric.value = value

    def _update_queue_metrics(self) -> None:
        self._set_metric("lpr_queue_depth", float(self._tasks.depth))
        self._set_metric("lpr_queue_replaced", float(self._tasks.replaced))
        self._set_metric("lpr_queue_full_drops", float(self._tasks.full_drops))

    def _increment_metric(self, name: str) -> None:
        metric = getattr(self.metrics, name, None)
        if metric is not None:
            metric.value += 1

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

    def associate_frame_objects(
        self, camera: str, objects: list[dict[str, Any]]
    ) -> tuple[list[LprPassageAdmission], list[LprPassageRejection]]:
        frame_time = max(
            (float(obj.get("frame_time") or 0.0) for obj in objects), default=0.0
        )
        admissions, rejections = associate_lpr_passages(
            objects,
            registry=self._passage_registry,
            camera=camera,
            frame_time=frame_time,
        )
        for rejection in rejections:
            self._get_recognition_lifecycle().record_skip("lpr", rejection.reason)
            passage_trace(
                "recognition_candidate",
                task="lpr",
                camera=camera,
                passage_id=None,
                plate_track_id=rejection.plate_track_id,
                raw_track_lineage=[rejection.plate_track_id],
                decision_reason=rejection.reason,
                candidate_vehicle_track_ids=list(
                    rejection.candidate_vehicle_track_ids
                ),
                admitted=False,
            )
        return admissions, rejections

    def _is_eligible(
        self, obj_data: dict[str, Any] | str, dedicated_lpr: bool
    ) -> tuple[str, str, float] | None:
        if dedicated_lpr:
            camera = str(obj_data)
            if (
                camera not in self.config.cameras
                or not self.config.cameras[camera].lpr.enabled
            ):
                return None
            return camera, "dedicated-lpr", datetime.datetime.now().timestamp()
        if not isinstance(obj_data, dict):
            return None
        camera = str(obj_data.get("camera"))
        if (
            camera not in self.config.cameras
            or not self.config.cameras[camera].lpr.enabled
        ):
            return None

        label = obj_data.get("label")
        if label not in self.lp_objects and label != "license_plate":
            return None
        if obj_data.get("stationary") is True:
            camera_config = self.config.cameras[camera]
            threshold = camera_config.detect.stationary.threshold
            motionless = obj_data.get("motionless_count", 0)
            if motionless >= threshold:
                elapsed = (motionless - threshold) / camera_config.detect.fps
                if elapsed > self.stationary_scan_duration:
                    return None
        if "license_plate" not in self.config.cameras[camera].objects.track:
            if not obj_data.get("box"):
                return None
        elif label != "license_plate" and not obj_data.get("current_attributes"):
            return None
        track_id = str(obj_data.get("_recognition_passage_id") or obj_data.get("id"))
        frame_time = float(
            obj_data.get("frame_time") or datetime.datetime.now().timestamp()
        )
        return camera, track_id, frame_time

    def process_frame(
        self,
        obj_data: dict[str, Any] | str,
        frame_ref: FrameRef,
        dedicated_lpr: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Gate, take one owning frame copy, and enqueue latest work per track."""
        eligible = self._is_eligible(obj_data, dedicated_lpr)
        if eligible is None:
            return
        camera, track_id, frame_time = eligible
        object_data: dict[str, Any] = (
            {} if dedicated_lpr else obj_data if isinstance(obj_data, dict) else {}
        )
        passage_trace(
            "lpr_eligible",
            camera=camera,
            frame_time=frame_time,
            passage_id=track_id,
            recognition_passage_id=track_id,
            track_id=str(object_data.get("id")) if not dedicated_lpr else track_id,
            raw_track_lineage=[]
            if dedicated_lpr
            else [
                value
                for value in (
                    object_data.get("_recognition_vehicle_track_id"),
                    *object_data.get("_recognition_plate_track_ids", ()),
                )
                if value
            ],
            object_box=None if dedicated_lpr else object_data.get("box"),
        )
        generation = self._tasks.generation(camera, track_id)
        key = LprTrackKey(camera, track_id, generation)
        lifecycle_key = RecognitionKey("lpr", camera, track_id, generation)
        if self._get_recognition_lifecycle().is_terminal(lifecycle_key):
            self._increment_metric("lpr_terminal_skips")
            return
        if not hasattr(self, "_last_seen_monotonic"):
            self._last_seen_monotonic = {}
        if not hasattr(self, "_last_prepared_monotonic"):
            self._last_prepared_monotonic = {}
        now = time.monotonic()
        self._last_seen_monotonic[key] = now
        prepare_interval = getattr(
            self.config.cameras[camera].recognition_lifecycle,
            "min_candidate_interval_seconds",
            0.4,
        )
        deadline = max(
            now,
            self._last_prepared_monotonic.get(key, now - prepare_interval)
            + prepare_interval,
        )
        object_box = None if dedicated_lpr else object_data.get("box")
        priority = (
            float(max(0, object_box[2] - object_box[0]))
            * float(max(0, object_box[3] - object_box[1]))
            * max(0.1, float(object_data.get("score", 1.0)))
            if object_box
            else 0.0
        )
        task = LprFrameTask(
            key=key,
            obj_data=str(obj_data) if dedicated_lpr else dict(object_data),
            frame_ref=frame_ref,
            dedicated_lpr=dedicated_lpr,
            frame_time=frame_time,
            # Plate quality is only known after plate detection in lpr_process.
            # Keep this raw-frame queue conflated but ready immediately; the
            # shared selector/lifecycle performs admission before OCR.
            collection_deadline=deadline,
            priority=priority,
        )
        self._tasks.submit(task)
        self._update_queue_metrics()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._tasks.get(timeout=0.1)
            if task is None:
                self._run_ready_candidate()
                self._expire_idle_passages()
                continue
            self._update_queue_metrics()
            if isinstance(task, LprExpireTask):
                self._expire_state(task)
                continue

            age = time.monotonic() - task.enqueued_at
            self._set_metric("lpr_task_age", age)
            if age > self.TASK_TTL:
                self._increment_metric("lpr_queue_ttl_drops")
                continue
            if not self._tasks.is_current(task.key):
                self._increment_metric("lpr_stale_generation_drops")
                continue
            if self._get_recognition_lifecycle().is_terminal(
                RecognitionKey(
                    "lpr", task.key.camera, task.key.track_id, task.key.generation
                )
            ):
                self._increment_metric("lpr_terminal_skips")
                self._get_recognition_lifecycle().record_skip("lpr", "terminal")
                continue

            started = time.monotonic()
            frame_lease = None
            try:
                if self.evidence_ring is None or self.quality_selector is None:
                    continue
                self._last_prepared_monotonic[task.key] = time.monotonic()
                frame_lease = self.evidence_ring.acquire(task.frame_ref)
                if frame_lease is None:
                    self.quality_selector.record_reject("lpr", "frame_expired")
                    continue
                prepared = self.lpr_process(
                    task.obj_data,
                    frame_lease.frame,
                    task.dedicated_lpr,
                    task.key,
                    task.frame_ref,
                )
                if not isinstance(prepared, PreparedPlateCandidate):
                    continue
                if not self._tasks.is_current(prepared.key):
                    self._increment_metric("lpr_stale_generation_drops")
                    prepared.evidence.release()
                    continue
                self._store_prepared(prepared)
                self._run_ready_candidate()
                self._expire_idle_passages()
            except Exception:
                logger.exception("Error processing realtime LPR task")
            finally:
                if frame_lease is not None:
                    frame_lease.release()
                self._set_metric("lpr_worker_latency", time.monotonic() - started)

    def _store_prepared(self, candidate: PreparedPlateCandidate) -> None:
        key = candidate.key
        selector = self.quality_selector
        if selector is None:
            candidate.evidence.release()
            return
        active_ids = selector.active_candidate_ids(
            "lpr", key.camera, key.passage_id, key.generation
        )
        retained: list[PreparedPlateCandidate] = []
        for previous in self._prepared.get(key, []):
            if previous.evidence.candidate_id in active_ids:
                retained.append(previous)
            else:
                previous.evidence.release()
        if candidate.evidence.candidate_id not in {
            item.evidence.candidate_id for item in retained
        }:
            retained.append(candidate)
        else:
            candidate.evidence.release()
        retained.sort(
            key=lambda item: (
                item.evidence.quality_score,
                item.detector_score if item.detector_score is not None else -1.0,
                item.evidence.candidate_id,
            ),
            reverse=True,
        )
        self._prepared[key] = retained
        self._collection_started.setdefault(key, time.monotonic())
        passage_trace(
            "recognition_candidate",
            task="lpr",
            camera=key.camera,
            passage_id=key.passage_id,
            recognition_passage_id=key.passage_id,
            track_id=key.passage_id,
            generation=key.generation,
            vehicle_track_id=candidate.vehicle_track_id,
            plate_track_ids=list(candidate.plate_track_ids),
            raw_track_lineage=[
                value
                for value in (candidate.vehicle_track_id, *candidate.plate_track_ids)
                if value
            ],
            candidate_id=candidate.evidence.candidate_id,
            evidence_id=candidate.evidence.frame_ref.identity,
            frame_id=candidate.evidence.frame_ref.frame_id,
            frame_time=candidate.frame_time,
            bbox=list(candidate.plate_box),
            object_box=list(candidate.object_box) if candidate.object_box else None,
            quality_score=candidate.evidence.quality_score,
            quality_components=candidate.evidence.quality_components,
            admitted=True,
        )

    def _run_ready_candidate(self) -> None:
        now = time.monotonic()
        eligible: list[PreparedPlateCandidate] = []
        for key, candidates in self._prepared.items():
            if not candidates:
                continue
            lifecycle_config = self.config.cameras[key.camera].recognition_lifecycle
            if (
                now - self._collection_started.get(key, now) + 1e-9
                < lifecycle_config.candidate_collection_seconds
            ):
                continue
            eligible.append(
                max(
                    candidates,
                    key=lambda item: (
                        item.evidence.quality_score,
                        item.evidence.candidate_id,
                    ),
                )
            )
        if not eligible:
            return
        candidate = max(
            eligible,
            key=lambda item: (
                item.evidence.quality_score,
                -item.prepared_monotonic,
                item.evidence.candidate_id,
            ),
        )
        self._prepared[candidate.key].remove(candidate)
        self._recognize_prepared(candidate)

    def _recognize_prepared(self, candidate: PreparedPlateCandidate) -> None:
        key = candidate.key
        lifecycle_config = self.config.cameras[key.camera].recognition_lifecycle
        lifecycle_key = RecognitionKey("lpr", key.camera, key.passage_id, key.generation)
        attempt, skip_reason = self._get_recognition_lifecycle().begin_attempt(
            lifecycle_key,
            candidate_id=candidate.evidence.candidate_id,
            frame_time=candidate.frame_time,
            detail_bbox=candidate.plate_box,
            quality_score=candidate.evidence.quality_score,
            policy=RecognitionPolicy(
                max_attempts=lifecycle_config.max_attempts,
                min_candidate_interval_seconds=(
                    lifecycle_config.min_candidate_interval_seconds
                ),
                max_candidate_bbox_iou=lifecycle_config.max_candidate_bbox_iou,
            ),
        )
        if attempt is None:
            candidate.evidence.release()
            passage_trace(
                "recognition_attempt",
                task="lpr",
                camera=key.camera,
                passage_id=key.passage_id,
                recognition_passage_id=key.passage_id,
                track_id=key.passage_id,
                generation=key.generation,
                candidate_id=candidate.evidence.candidate_id,
                evidence_id=candidate.evidence.frame_ref.identity,
                frame_id=candidate.evidence.frame_ref.frame_id,
                frame_time=candidate.frame_time,
                bbox=list(candidate.plate_box),
                object_box=list(candidate.object_box) if candidate.object_box else None,
                vehicle_track_id=candidate.vehicle_track_id,
                plate_track_ids=list(candidate.plate_track_ids),
                quality_score=candidate.evidence.quality_score,
                decision_reason=skip_reason,
                inference_started=False,
            )
            if skip_reason == "attempt_budget_exhausted":
                self._finish_passage(key, "insufficient_quality", boundary=False)
            return

        started = time.monotonic()
        plates, confidences, areas = self._process_license_plate(
            key.camera,
            key.passage_id,
            candidate.plate_frame,
            int(candidate.frame_time * 1000),
        )
        self.plates_rec_second.update()
        self.plate_rec_speed.update(time.monotonic() - started)
        plate = plates[0] if plates else None
        char_confidences = confidences[0] if confidences else []
        confidence = (
            sum(char_confidences) / len(char_confidences)
            if char_confidences
            else 0.0
        )
        if plate:
            passage_trace(
                "ocr_result",
                task="lpr",
                camera=key.camera,
                passage_id=key.passage_id,
                recognition_passage_id=key.passage_id,
                track_id=key.passage_id,
                generation=key.generation,
                vehicle_track_id=candidate.vehicle_track_id,
                plate_track_ids=list(candidate.plate_track_ids),
                candidate_id=candidate.evidence.candidate_id,
                evidence_id=candidate.evidence.frame_ref.identity,
                frame_id=candidate.evidence.frame_ref.frame_id,
                frame_time=candidate.frame_time,
                plate=plate,
                score=confidence,
                score_type="raw_mean_character_score",
                plate_box=list(candidate.plate_box),
                object_box=list(candidate.object_box)
                if candidate.object_box
                else None,
            )
        completed = self._get_recognition_lifecycle().complete_attempt(
            attempt,
            result=plate,
            confidence=confidence,
            confidence_type="raw_mean_character_score",
            reason="inference_completed" if plate else "no_ocr_result",
        )
        passage_trace(
            "recognition_attempt",
            task="lpr",
            camera=key.camera,
            passage_id=key.passage_id,
            recognition_passage_id=key.passage_id,
            track_id=key.passage_id,
            generation=key.generation,
            vehicle_track_id=candidate.vehicle_track_id,
            plate_track_ids=list(candidate.plate_track_ids),
            raw_track_lineage=[
                value
                for value in (candidate.vehicle_track_id, *candidate.plate_track_ids)
                if value
            ],
            attempt_index=attempt.attempt_index,
            candidate_id=candidate.evidence.candidate_id,
            evidence_id=candidate.evidence.frame_ref.identity,
            frame_id=candidate.evidence.frame_ref.frame_id,
            frame_time=candidate.frame_time,
            bbox=list(candidate.plate_box),
            object_box=list(candidate.object_box) if candidate.object_box else None,
            quality_score=candidate.evidence.quality_score,
            ocr=plate,
            confidence=confidence,
            latency_ms=(time.monotonic() - attempt.started_monotonic) * 1000,
            decision_reason="consensus_pending" if plate else "no_ocr_result",
            inference_started=True,
        )
        if not completed:
            candidate.evidence.release()
            self._increment_metric("lpr_stale_generation_drops")
            return
        if not plate:
            candidate.evidence.release()
            if attempt.attempt_index >= lifecycle_config.max_attempts:
                self._finish_passage(key, "insufficient_quality", boundary=False)
            return
        observation = PlateObservation(
            key=key,
            frame_time=candidate.frame_time,
            plate=plate,
            char_confidences=tuple(float(value) for value in char_confidences),
            text_area=int(areas[0]) if areas else 0,
            plate_box=candidate.plate_box,
            object_box=candidate.object_box,
            obj_data=candidate.obj_data,
            dedicated_lpr=candidate.dedicated_lpr,
            evidence=candidate.evidence,
            attempt=attempt,
            vehicle_track_id=candidate.vehicle_track_id,
            plate_track_ids=candidate.plate_track_ids,
        )
        commit = self._reduce(observation)
        if commit is not None and self._tasks.is_current(commit.key):
            self._emit_commit(commit)
        elif commit is not None:
            self._increment_metric("lpr_stale_generation_drops")

    def _release_prepared(self, key: LprTrackKey) -> None:
        for candidate in self._prepared.pop(key, []):
            candidate.evidence.release()
        self._collection_started.pop(key, None)

    def _expire_idle_passages(self) -> None:
        now = time.monotonic()
        for key, last_seen in list(
            getattr(self, "_last_seen_monotonic", {}).items()
        ):
            idle_seconds = getattr(
                self.config.cameras[key.camera].recognition_lifecycle,
                "passage_idle_seconds",
                1.0,
            )
            if now - last_seen >= idle_seconds:
                if self._tasks.contains(key) or self._prepared.get(key):
                    continue
                self._finish_passage(key, "insufficient_quality", boundary=False)

    def _finish_passage(
        self, key: LprTrackKey, reason: str, *, boundary: bool
    ) -> None:
        lifecycle_key = RecognitionKey("lpr", key.camera, key.passage_id, key.generation)
        lifecycle = self._get_recognition_lifecycle()
        searching = lifecycle.status(lifecycle_key) == RecognitionStatus.SEARCHING
        if searching:
            lifecycle.terminal(lifecycle_key, RecognitionStatus.EXHAUSTED, reason)
            passage_trace(
                "recognition_terminal",
                task="lpr",
                camera=key.camera,
                passage_id=key.passage_id,
                recognition_passage_id=key.passage_id,
                track_id=key.passage_id,
                generation=key.generation,
                status=RecognitionStatus.EXHAUSTED.value,
                reason=reason,
                winner=None,
                best_effort=False,
            )
        state = self._states.pop(key, None)
        if state is not None:
            self._release_state(state)
        self._release_prepared(key)
        if hasattr(self._tasks, "cancel"):
            self._tasks.cancel(key)
        getattr(self, "_last_prepared_monotonic", {}).pop(key, None)
        getattr(self, "_last_seen_monotonic", {}).pop(key, None)
        if self.quality_selector is not None:
            self.quality_selector.expire(
                "lpr", key.camera, key.passage_id, key.generation
            )
        if boundary:
            self._terminal_keys.discard(key)
            lifecycle.expire(lifecycle_key, reason)
        else:
            self._terminal_keys.add(key)

    def _expire_state(self, task: LprExpireTask) -> None:
        keys = set(self._states) | set(self._prepared) | self._terminal_keys
        for key in list(keys):
            sweep = task.generation < 0 and not self._tasks.is_current(key)
            targeted = (
                key.camera == task.camera
                and key.passage_id == task.track_id
                and key.generation < task.generation
            )
            if sweep or targeted:
                self._finish_passage(key, "insufficient_quality", boundary=True)

    @staticmethod
    def _normalized_plate(plate: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def _new_state(self, key: LprTrackKey) -> PlateTrackState:
        while len(self._states) >= self.MAX_STATES:
            expired_key = next(iter(self._states))
            self._finish_passage(
                expired_key, "insufficient_quality", boundary=False
            )
        state = PlateTrackState(key=key)
        self._states[key] = state
        return state

    def _reduce(self, observation: PlateObservation) -> PlateCommit | None:
        state = self._states.get(observation.key)
        if state is None:
            state = self._new_state(observation.key)
        else:
            self._states.move_to_end(observation.key)
        if observation.frame_time in state.seen_frames:
            observation.evidence.release()
            return None
        state.seen_frames.add(observation.frame_time)
        state.seen_frame_order.append(observation.frame_time)
        attempt_index = (
            observation.attempt.attempt_index
            if observation.attempt
            else len(state.seen_frames)
        )
        lifecycle_config = self.config.cameras[
            observation.key.camera
        ].recognition_lifecycle
        normalized = self._normalized_plate(observation.plate)
        valid = (
            observation.confidence >= self.lpr_config.recognition_threshold
            and len(normalized) >= self.lpr_config.min_plate_length
        )
        if valid and self.lpr_config.format:
            try:
                valid = re.fullmatch(self.lpr_config.format, normalized) is not None
            except re.error:
                logger.error("Invalid regex in LPR format configuration")
                valid = False
        if not valid:
            observation.evidence.release()
            if attempt_index >= lifecycle_config.max_attempts:
                self._finish_passage(
                    observation.key, "insufficient_quality", boundary=False
                )
            return None

        state.variants.append(
            {
                "plate": normalized,
                "conf": observation.confidence,
                "char_confidences": list(observation.char_confidences),
                "area": observation.text_area,
                "timestamp": observation.frame_time,
                "candidate_id": observation.evidence.candidate_id,
                "quality": observation.evidence.quality_score,
                "observation": observation,
            }
        )
        winner = max(
            state.variants,
            key=lambda variant: (
                sum(
                    peer["plate"] == variant["plate"]
                    for peer in state.variants
                ),
                float(variant["conf"]),
                int(variant["area"]),
                float(variant["quality"]),
                str(variant["candidate_id"]),
            ),
        )
        cluster = [
            variant
            for variant in state.variants
            if variant["plate"] == winner["plate"]
        ]
        support = len(cluster)
        representative_variant = max(
            cluster,
            key=lambda variant: (
                float(variant["conf"]),
                int(variant["area"]),
                float(variant["quality"]),
                str(variant["candidate_id"]),
            ),
        )
        representative = representative_variant["observation"]
        rep_plate = str(representative_variant["plate"])
        rep_conf = float(representative_variant["conf"])
        rep_area = int(representative_variant["area"])
        state.representative_plate = rep_plate
        state.object_box = observation.object_box
        state.last_seen = observation.frame_time
        if support < lifecycle_config.lpr_min_consensus_votes:
            if attempt_index >= lifecycle_config.max_attempts:
                self._finish_passage(
                    observation.key, "insufficient_quality", boundary=False
                )
            return None
        strength = (support, float(rep_conf), int(rep_area))
        if state.event_id is None:
            if observation.dedicated_lpr:
                suffix = "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=6)
                )
                state.event_id = f"{datetime.datetime.now().timestamp()}-{suffix}"
            else:
                # Lifecycle ownership is canonical, while the external Event
                # update targets the raw vehicle Event that owns the winning
                # representative evidence.
                state.event_id = (
                    representative.vehicle_track_id
                    or observation.key.passage_id
                )

        sub_label = self._known_plate_label(rep_plate)
        commit = self._materialize_commit(
            representative, state.event_id, rep_plate, rep_plate, rep_conf, sub_label
        )
        if commit is None:
            if attempt_index >= lifecycle_config.max_attempts:
                self._finish_passage(
                    observation.key, "insufficient_quality", boundary=False
                )
            return None
        state.committed_plates.add(rep_plate)
        state.committed_plate = rep_plate
        state.committed_strength = strength
        self._get_recognition_lifecycle().terminal(
            RecognitionKey(
                "lpr",
                observation.key.camera,
                observation.key.passage_id,
                observation.key.generation,
            ),
            RecognitionStatus.ACCEPTED,
            "consensus_accepted",
        )
        passage_trace(
            "recognition_terminal",
            camera=observation.key.camera,
            frame_time=representative.frame_time,
            passage_id=observation.key.passage_id,
            recognition_passage_id=observation.key.passage_id,
            track_id=observation.key.passage_id,
            generation=observation.key.generation,
            task="lpr",
            status=RecognitionStatus.ACCEPTED.value,
            reason="consensus_accepted",
            winner=rep_plate,
            candidate_id=representative.evidence.candidate_id,
            evidence_id=representative.evidence.frame_ref.identity,
            frame_id=representative.evidence.frame_ref.frame_id,
            bbox=list(representative.plate_box),
            vehicle_track_id=representative.vehicle_track_id,
            plate_track_ids=list(representative.plate_track_ids),
            best_effort=False,
            consensus_tier="independent_candidates",
            consensus_support=support,
        )
        if self.quality_selector is not None:
            self.quality_selector.expire(
                "lpr",
                observation.key.camera,
                observation.key.passage_id,
                observation.key.generation,
            )
        self._states.pop(observation.key, None)
        self._release_state(state)
        self._release_prepared(observation.key)
        if hasattr(self._tasks, "cancel"):
            self._tasks.cancel(observation.key)
        getattr(self, "_last_prepared_monotonic", {}).pop(observation.key, None)
        getattr(self, "_last_seen_monotonic", {}).pop(observation.key, None)
        self._terminal_keys.add(observation.key)
        return commit

    def _known_plate_label(self, plate: str) -> str | None:
        known_plates = self.lpr_config.known_plates or {}
        try:
            return next(
                (
                    label
                    for label, plates in known_plates.items()
                    if any(
                        re.match(f"^{candidate}$", plate)
                        or Levenshtein.distance(candidate, plate)
                        <= self.lpr_config.match_distance
                        for candidate in plates
                    )
                ),
                None,
            )
        except re.error:
            logger.error("Invalid regex in known plates configuration")
            return None

    def _materialize_commit(
        self,
        observation: PlateObservation,
        event_id: str,
        normalized: str,
        plate: str,
        score: float,
        sub_label: str | None,
    ) -> PlateCommit | None:
        commit_id = hashlib.sha256(
            f"{observation.key.camera}:{observation.key.track_id}:"
            f"{observation.key.generation}:{normalized}".encode()
        ).hexdigest()
        evidence_id = hashlib.sha256(f"{commit_id}:lpr".encode()).hexdigest()
        evidence_dir = Path(CLIPS_DIR) / "artifacts" / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / f"{evidence_id}.jpg"
        full_frame_bgr = cv2.cvtColor(
            observation.evidence.frame, cv2.COLOR_YUV2BGR_I420
        )
        ok, encoded = cv2.imencode(
            ".jpg", full_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            return None
        if not evidence_path.exists():
            temporary = evidence_path.with_suffix(".tmp")
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, evidence_path)
        snapshot = (
            base64.b64encode(encoded.tobytes()).decode("ASCII")
            if observation.dedicated_lpr
            else None
        )
        return PlateCommit(
            commit_id=commit_id,
            key=observation.key,
            event_id=event_id,
            camera=observation.key.camera,
            plate=plate,
            score=float(score),
            sub_label=sub_label,
            timestamp=time.time(),
            frame_time=observation.frame_time,
            plate_box=observation.plate_box,
            object_box=observation.plate_box
            if observation.dedicated_lpr
            else observation.object_box,
            evidence_id=evidence_id,
            frame_ref=str(evidence_path),
            frame_width=int(full_frame_bgr.shape[1]),
            frame_height=int(full_frame_bgr.shape[0]),
            obj_data=observation.obj_data,
            dedicated_lpr=observation.dedicated_lpr,
            snapshot=snapshot,
            candidate_id=observation.evidence.candidate_id,
            quality_score=observation.evidence.quality_score,
            quality_components=observation.evidence.quality_components,
            source_role=observation.evidence.source_role.value,
        )

    def _emit_commit(self, commit: PlateCommit) -> None:
        with self._results_lock:
            if len(self._results) >= self.MAX_RESULTS:
                logger.error(
                    "LPR commit queue full; dropping commit %s", commit.commit_id
                )
                return
            self._results.append(commit)

    def _emit_activity(self, activity: PlateActivity) -> None:
        with self._results_lock:
            # Conflate consecutive heartbeats for the same manual event.
            if (
                self._results
                and isinstance(self._results[-1], PlateActivity)
                and self._results[-1].event_id == activity.event_id
            ):
                self._results[-1] = activity
            elif len(self._results) < self.MAX_RESULTS:
                self._results.append(activity)

    def drain_results(self) -> list[PlateCommit | PlateActivity]:
        with self._results_lock:
            results = list(self._results)
            self._results.clear()
        return results

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        """Priority reset invalidates pending and in-flight results immediately."""
        active = getattr(self, "_active_detection_ids", None)
        if active is None:
            active = {}
            self._active_detection_ids = active
        active.setdefault(camera, set()).discard(str(object_id))
        registry = getattr(self, "_passage_registry", None)
        boundaries = (
            registry.retire_raw(camera, str(object_id))
            if registry is not None
            else None
        )
        for passage_id in ([str(object_id)] if boundaries is None else boundaries):
            self._tasks.advance_generation(camera, passage_id)
        self._update_queue_metrics()

    def expire_missing_objects(self, camera: str, active_ids: set[str]) -> None:
        """Reconcile LPR work with the authoritative active detection set."""
        active_ids = {str(object_id) for object_id in active_ids}
        previous = self._active_detection_ids.get(camera, set())
        for object_id in previous - active_ids:
            self.expire_object(object_id, camera)
        self._active_detection_ids[camera] = active_ids

    @staticmethod
    def _release_state(state: PlateTrackState) -> None:
        for variant in state.variants:
            observation = variant.get("observation")
            if observation is not None:
                observation.evidence.release()
        state.variants.clear()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._tasks.wake()
        self._worker.join(timeout=5.0)
        keys = (
            set(self._states)
            | set(self._prepared)
            | self._terminal_keys
            | set(getattr(self, "_last_seen_monotonic", {}))
        )
        for key in keys:
            # Shutdown is cancellation only. _finish_passage never emits.
            self._finish_passage(key, "shutdown", boundary=True)
        self._terminal_keys.clear()
        registry = getattr(self, "_passage_registry", None)
        if registry is not None:
            registry.clear()

    @property
    def pending_count(self) -> int:
        return self._tasks.depth

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def prepared_count(self) -> int:
        return sum(len(candidates) for candidates in self._prepared.values())
