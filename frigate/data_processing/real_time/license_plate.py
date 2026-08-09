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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
from rapidfuzz.distance import JaroWinkler, Levenshtein

from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import CLIPS_DIR
from frigate.data_processing.common.evidence import EvidenceRingBuffer, FrameRef
from frigate.data_processing.common.license_plate.association import (
    is_lpr_track_discontinuity,
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
)
from frigate.data_processing.common.quality import QualitySelector

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
        super().__init__(config, metrics)

        self._tasks = LatestLprTaskQueue(self.MAX_TRACKS)
        self._states: OrderedDict[LprTrackKey, PlateTrackState] = OrderedDict()
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

    def _is_eligible(
        self, obj_data: dict[str, Any] | str, dedicated_lpr: bool
    ) -> tuple[str, str, float] | None:
        camera = str(obj_data) if dedicated_lpr else str(obj_data.get("camera"))
        if (
            camera not in self.config.cameras
            or not self.config.cameras[camera].lpr.enabled
        ):
            return None
        if dedicated_lpr:
            return camera, "dedicated-lpr", datetime.datetime.now().timestamp()

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
        track_id = str(obj_data.get("id"))
        frame_time = float(
            obj_data.get("frame_time") or datetime.datetime.now().timestamp()
        )
        return camera, track_id, frame_time

    def process_frame(
        self,
        obj_data: dict[str, Any] | str,
        frame_ref: FrameRef,
        dedicated_lpr: bool = False,
    ) -> None:
        """Gate, take one owning frame copy, and enqueue latest work per track."""
        eligible = self._is_eligible(obj_data, dedicated_lpr)
        if eligible is None:
            return
        camera, track_id, frame_time = eligible
        generation = self._tasks.generation(camera, track_id)
        key = LprTrackKey(camera, track_id, generation)
        task = LprFrameTask(
            key=key,
            obj_data=str(obj_data) if dedicated_lpr else dict(obj_data),
            frame_ref=frame_ref,
            dedicated_lpr=dedicated_lpr,
            frame_time=frame_time,
        )
        self._tasks.submit(task)
        self._update_queue_metrics()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._tasks.get(timeout=0.5)
            if task is None:
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

            started = time.monotonic()
            lease = None
            try:
                if self.evidence_ring is None or self.quality_selector is None:
                    continue
                lease = self.evidence_ring.acquire(task.frame_ref)
                if lease is None:
                    self.quality_selector.record_reject("lpr", "frame_expired")
                    continue
                observation = self.lpr_process(
                    task.obj_data,
                    lease.frame,
                    task.dedicated_lpr,
                    task.key,
                    task.frame_ref,
                )
                if observation is None:
                    continue
                if not self._tasks.is_current(observation.key):
                    self._increment_metric("lpr_stale_generation_drops")
                    observation.evidence.release()
                    continue
                commit = self._reduce(observation)
                if commit is not None and self._tasks.is_current(commit.key):
                    self._emit_commit(commit)
                elif commit is not None:
                    self._increment_metric("lpr_stale_generation_drops")
            except Exception:
                logger.exception("Error processing realtime LPR task")
            finally:
                if lease is not None:
                    lease.release()
                self._set_metric("lpr_worker_latency", time.monotonic() - started)

    def _expire_state(self, task: LprExpireTask) -> None:
        for key in list(self._states):
            sweep = task.generation < 0 and not self._tasks.is_current(key)
            targeted = (
                key.camera == task.camera
                and key.track_id == task.track_id
                and key.generation < task.generation
            )
            if sweep or targeted:
                self._release_state(self._states.pop(key))
                if self.quality_selector is not None:
                    self.quality_selector.expire(
                        "lpr", key.camera, key.track_id, key.generation
                    )

    @staticmethod
    def _normalized_plate(plate: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def _new_state(self, key: LprTrackKey) -> PlateTrackState:
        while len(self._states) >= self.MAX_STATES:
            _, expired = self._states.popitem(last=False)
            self._release_state(expired)
        state = PlateTrackState(key=key)
        self._states[key] = state
        return state

    def _reduce(self, observation: PlateObservation) -> PlateCommit | None:
        state = self._states.get(observation.key)
        if state is None:
            state = self._new_state(observation.key)
        else:
            self._states.move_to_end(observation.key)

        if (
            observation.dedicated_lpr
            and state.last_seen is not None
            and observation.frame_time - state.last_seen
            > self.config.cameras[observation.key.camera].lpr.expire_time
        ):
            observation = self._tasks.rekey(observation)
            if self.quality_selector is not None:
                observation = replace(
                    observation,
                    evidence=self.quality_selector.rekey(
                        observation.evidence, observation.key.generation
                    ),
                )
            state = self._new_state(observation.key)

        if observation.frame_time in state.seen_frames:
            observation.evidence.release()
            return None

        if (
            not observation.dedicated_lpr
            and state.representative_plate
            and is_lpr_track_discontinuity(
                state.representative_plate,
                observation.plate,
                state.object_box,
                observation.object_box,
                self.cluster_threshold,
            )
        ):
            if (
                state.switch_candidate
                and JaroWinkler.similarity(state.switch_candidate, observation.plate)
                >= self.cluster_threshold
            ):
                state.switch_count += 1
            else:
                state.switch_candidate = observation.plate
                state.switch_count = 1
            if state.switch_count < 2:
                observation.evidence.release()
                return None
            observation = self._tasks.rekey(observation)
            if self.quality_selector is not None:
                observation = replace(
                    observation,
                    evidence=self.quality_selector.rekey(
                        observation.evidence, observation.key.generation
                    ),
                )
            state = self._new_state(observation.key)
        else:
            state.switch_candidate = None
            state.switch_count = 0

        state.seen_frames.add(observation.frame_time)
        state.seen_frame_order.append(observation.frame_time)
        max_variants = max(
            1, int(self.config.cameras[observation.key.camera].detect.fps * 5)
        )
        while len(state.seen_frame_order) > max_variants:
            state.seen_frames.discard(state.seen_frame_order.popleft())

        if (
            self.quality_selector is not None
            and self.config.cameras[observation.key.camera].quality.enabled
        ):
            active_ids = self.quality_selector.active_candidate_ids(
                "lpr",
                observation.key.camera,
                observation.key.track_id,
                observation.key.generation,
            )
            retained_variants = []
            for variant in state.variants:
                prior = variant.get("observation")
                if prior is None or prior.evidence.candidate_id in active_ids:
                    retained_variants.append(variant)
                else:
                    prior.evidence.release()
            state.variants = retained_variants

        state.variants.append(
            {
                "plate": observation.plate,
                "conf": observation.confidence,
                "char_confidences": list(observation.char_confidences),
                "area": observation.text_area,
                "timestamp": observation.frame_time,
                "candidate_id": observation.evidence.candidate_id,
                "quality": observation.evidence.quality_score,
                "observation": observation,
            }
        )
        if len(state.variants) > max_variants:
            removed = state.variants[:-max_variants]
            state.variants = state.variants[-max_variants:]
            for variant in removed:
                prior = variant.get("observation")
                if prior is not None:
                    prior.evidence.release()

        rep_plate, rep_conf, _, rep_area = self._get_cluster_rep(state.variants)
        representative = max(
            (
                variant
                for variant in state.variants
                if JaroWinkler.similarity(variant["plate"], rep_plate)
                >= self.cluster_threshold
            ),
            key=lambda variant: (
                variant["plate"] == rep_plate,
                float(variant["conf"]),
                int(variant["area"]),
                float(variant.get("quality", 0.0)),
                str(variant.get("candidate_id", "")),
            ),
        )["observation"]
        state.representative_plate = rep_plate
        state.object_box = observation.object_box
        state.last_seen = observation.frame_time

        # Raw observations enter consensus before commit filters are applied.
        if rep_conf < self.lpr_config.recognition_threshold:
            return None
        if len(rep_plate) < self.lpr_config.min_plate_length:
            return None
        if self.lpr_config.format:
            try:
                if not re.fullmatch(self.lpr_config.format, rep_plate):
                    return None
            except re.error:
                logger.error("Invalid regex in LPR format configuration")

        normalized = self._normalized_plate(rep_plate)
        if not normalized:
            return None
        if normalized in state.committed_plates:
            if observation.dedicated_lpr and state.event_id is not None:
                self._emit_activity(
                    PlateActivity(
                        event_id=state.event_id,
                        camera=observation.key.camera,
                        frame_time=observation.frame_time,
                        key=observation.key,
                    )
                )
            return None
        support = sum(
            1
            for variant in state.variants
            if JaroWinkler.similarity(variant["plate"], rep_plate)
            >= self.cluster_threshold
        )
        strength = (support, float(rep_conf), int(rep_area))
        if (
            state.committed_strength is not None
            and strength <= state.committed_strength
        ):
            return None

        if state.event_id is None:
            if observation.dedicated_lpr:
                suffix = "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=6)
                )
                state.event_id = f"{datetime.datetime.now().timestamp()}-{suffix}"
            else:
                state.event_id = observation.key.track_id

        sub_label = self._known_plate_label(rep_plate)
        commit = self._materialize_commit(
            representative, state.event_id, normalized, rep_plate, rep_conf, sub_label
        )
        if commit is None:
            return None
        state.committed_plates.add(normalized)
        state.committed_plate = normalized
        state.committed_strength = strength
        return commit

    def _known_plate_label(self, plate: str) -> str | None:
        try:
            return next(
                (
                    label
                    for label, plates in self.lpr_config.known_plates.items()
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
        self._tasks.advance_generation(camera, str(object_id))
        self._update_queue_metrics()

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
        for state in self._states.values():
            self._release_state(state)
        self._states.clear()

    @property
    def pending_count(self) -> int:
        return self._tasks.depth

    @property
    def state_count(self) -> int:
        return len(self._states)
