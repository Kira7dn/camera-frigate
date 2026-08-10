"""Bounded multi-camera capture and batched face recognition pipeline."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

import cv2
import numpy as np

from frigate.const import MODEL_CACHE_DIR
from frigate.data_processing.common.evidence import EvidenceCandidate, EvidenceLease
from frigate.data_processing.common.quality import QualitySelector, QualityThresholds
from frigate.data_processing.common.recognition import (
    RecognitionAttemptLease,
    RecognitionKey,
    RecognitionLifecycle,
    RecognitionPolicy,
    RecognitionStatus,
)
from frigate.util.image import area, calculate_region, yuv_region_2_bgr
from frigate.util.passage_trace import passage_trace

logger = logging.getLogger(__name__)

FACE_CAPTURE_WORKERS = 2
FACE_PREPROCESS_WORKERS = 2
FACE_BATCH_SIZE = 4
FACE_BATCH_FLUSH_SECONDS = 0.4
FACE_CANDIDATE_TTL_SECONDS = 1.5
MAX_TRACKS_PER_CAMERA = 4

FaceKey = tuple[str, str]


@dataclass(frozen=True)
class FaceCaptureRequest:
    camera: str
    event_id: str
    frame_time: float
    generation: int
    person_box: tuple[int, int, int, int]
    evidence_lease: EvidenceLease
    detection_threshold: float
    min_area: int
    requires_face_detection: bool
    attribute_face_box: tuple[int, int, int, int] | None
    vote_count: int
    created_monotonic: float
    quality_enabled: bool
    quality_thresholds: QualityThresholds
    top_k: int
    candidate_collection_seconds: float = FACE_BATCH_FLUSH_SECONDS
    detector_score: float | None = None
    quality: float = 0.0
    lifecycle_policy: RecognitionPolicy = field(default_factory=RecognitionPolicy)

    @property
    def key(self) -> FaceKey:
        return (self.camera, self.event_id)


@dataclass(frozen=True)
class FaceCandidate:
    request: FaceCaptureRequest
    face_box: tuple[int, int, int, int]
    face_frame: np.ndarray
    capture_ms: float
    quality: float
    evidence: EvidenceCandidate

    @property
    def key(self) -> FaceKey:
        return self.request.key

    @property
    def camera(self) -> str:
        return self.request.camera

    @property
    def created_monotonic(self) -> float:
        return self.request.created_monotonic

    @property
    def vote_count(self) -> int:
        return self.request.vote_count


@dataclass(frozen=True)
class PreparedFaceCandidate:
    candidate: FaceCandidate
    aligned_face: np.ndarray
    blur_reduction: float
    alignment_ms: float
    prepared_monotonic: float

    @property
    def key(self) -> FaceKey:
        return self.candidate.key

    @property
    def camera(self) -> str:
        return self.candidate.camera

    @property
    def created_monotonic(self) -> float:
        return self.candidate.created_monotonic

    @property
    def vote_count(self) -> int:
        return self.candidate.vote_count

    @property
    def quality(self) -> float:
        return self.candidate.quality

    @property
    def collection_seconds(self) -> float:
        return self.candidate.request.candidate_collection_seconds


@dataclass(frozen=True)
class FaceRecognitionOutcome:
    candidate: FaceCandidate
    sub_label: str
    score: float
    top2_label: str | None
    top2_score: float
    batch_size: int
    batch_wait_ms: float
    alignment_ms: float
    embedding_ms: float
    completed_monotonic: float
    attempt: RecognitionAttemptLease


T = TypeVar("T")


class LatestFaceCandidateStore(Generic[T]):
    """Thread-safe latest-only keyed store with bounded per-camera ownership."""

    def __init__(
        self,
        max_per_camera: int = MAX_TRACKS_PER_CAMERA,
        ttl_seconds: float = FACE_CANDIDATE_TTL_SECONDS,
        on_drop: Callable[[T, str], None] | None = None,
        prefer_quality: bool = False,
    ) -> None:
        self.max_per_camera = max_per_camera
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[FaceKey, T] = OrderedDict()
        self._condition = threading.Condition()
        self._camera_cursor = 0
        self._on_drop = on_drop
        self.prefer_quality = prefer_quality

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    def keys(self) -> set[FaceKey]:
        with self._condition:
            return set(self._items)

    def submit(self, item: T) -> bool:
        key = getattr(item, "key")
        camera = getattr(item, "camera")
        dropped: list[tuple[T, str]] = []
        accepted = True
        with self._condition:
            previous = self._items.pop(key, None)
            if previous is not None:
                if (
                    self.prefer_quality
                    and self._quality_rank(previous) >= self._quality_rank(item)
                ):
                    self._items[key] = previous
                    dropped.append((item, "lower_quality"))
                    accepted = False
                else:
                    dropped.append((previous, "replaced"))
            if accepted:
                camera_keys = [k for k in self._items if k[0] == camera]
                if len(camera_keys) >= self.max_per_camera:
                    evicted_key = max(
                        camera_keys,
                        key=lambda candidate_key: self._priority(
                            self._items[candidate_key]
                        ),
                    )
                    dropped.append((self._items.pop(evicted_key), "camera_limit"))
                self._items[key] = item
                self._condition.notify_all()
        self._notify_drops(dropped)
        return accepted

    def remove(self, key: FaceKey, reason: str = "removed") -> bool:
        with self._condition:
            item = self._items.pop(key, None)
            self._condition.notify_all()
        if item is not None:
            self._notify_drops([(item, reason)])
            return True
        return False

    def remove_camera_missing(self, camera: str, active_ids: set[str]) -> int:
        with self._condition:
            removed = [
                self._items.pop(key)
                for key in list(self._items)
                if key[0] == camera and key[1] not in active_ids
            ]
            self._condition.notify_all()
        self._notify_drops([(item, "track_ended") for item in removed])
        return len(removed)

    def take_fair(
        self,
        max_items: int,
        timeout: float,
        excluded: set[FaceKey] | None = None,
        flush_seconds: float = 0.0,
        per_item_flush: bool = False,
    ) -> list[T]:
        deadline = time.monotonic() + timeout
        excluded = excluded or set()
        dropped: list[tuple[T, str]] = []
        with self._condition:
            while True:
                now = time.monotonic()
                dropped.extend(self._prune_expired(now))
                eligible = [
                    item for key, item in self._items.items() if key not in excluded
                ]
                ready = (
                    [
                        item
                        for item in eligible
                        if now - float(getattr(item, "created_monotonic"))
                        >= float(getattr(item, "collection_seconds", flush_seconds))
                    ]
                    if per_item_flush
                    else eligible
                )
                oldest_age = max(
                    (
                        now - float(getattr(item, "created_monotonic"))
                        for item in eligible
                    ),
                    default=0.0,
                )
                if ready and (
                    per_item_flush
                    or flush_seconds <= 0
                    or len(ready) >= max_items
                    or oldest_age >= flush_seconds
                ):
                    selected = self._select_fair(ready, max_items)
                    for item in selected:
                        self._items.pop(getattr(item, "key"), None)
                    break
                remaining = deadline - now
                if remaining <= 0:
                    selected = (
                        self._select_fair(ready, max_items) if ready else []
                    )
                    for item in selected:
                        self._items.pop(getattr(item, "key"), None)
                    break
                wait_for = remaining
                if eligible and per_item_flush:
                    wait_for = min(
                        wait_for,
                        max(
                            0.001,
                            min(
                                float(getattr(item, "collection_seconds", flush_seconds))
                                - (now - float(getattr(item, "created_monotonic")))
                                for item in eligible
                            ),
                        ),
                    )
                elif eligible and flush_seconds > 0:
                    wait_for = min(wait_for, max(0.001, flush_seconds - oldest_age))
                self._condition.wait(wait_for)
        self._notify_drops(dropped)
        return selected

    def _prune_expired(self, now: float) -> list[tuple[T, str]]:
        expired = [
            key
            for key, item in self._items.items()
            if now - float(getattr(item, "created_monotonic")) > self.ttl_seconds
        ]
        return [(self._items.pop(key), "ttl") for key in expired]

    @staticmethod
    def _priority(item: T) -> tuple[float, float, float]:
        votes = int(getattr(item, "vote_count", 0))
        vote_rank = 0 if votes == 0 else (1 if votes == 1 else 2)
        return (
            float(vote_rank),
            float(getattr(item, "created_monotonic")),
            -float(getattr(item, "quality", 0.0)),
        )

    @staticmethod
    def _quality_rank(item: T) -> tuple[float, float, str]:
        evidence = getattr(getattr(item, "candidate", item), "evidence", None)
        candidate_id = str(getattr(evidence, "candidate_id", ""))
        return (
            float(getattr(item, "quality", 0.0)),
            -float(getattr(item, "created_monotonic", 0.0)),
            candidate_id,
        )

    def _select_fair(self, eligible: list[T], max_items: int) -> list[T]:
        by_camera: dict[str, list[T]] = {}
        for item in eligible:
            by_camera.setdefault(str(getattr(item, "camera")), []).append(item)
        for items in by_camera.values():
            items.sort(key=self._priority)
        cameras = sorted(by_camera)
        if not cameras:
            return []
        start = self._camera_cursor % len(cameras)
        cameras = cameras[start:] + cameras[:start]
        self._camera_cursor = (start + min(max_items, len(cameras))) % len(cameras)
        selected: list[T] = []
        while len(selected) < max_items:
            made_progress = False
            for camera in cameras:
                if by_camera[camera] and len(selected) < max_items:
                    selected.append(by_camera[camera].pop(0))
                    made_progress = True
            if not made_progress:
                break
        return selected

    def _notify_drops(self, drops: list[tuple[T, str]]) -> None:
        if self._on_drop is None:
            return
        for item, reason in drops:
            self._on_drop(item, reason)


def create_yunet_detector() -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        os.path.join(MODEL_CACHE_DIR, "facedet/facedet.onnx"),
        config="",
        input_size=(320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
    )


def detect_largest_face(
    detector: cv2.FaceDetectorYN,
    image: np.ndarray,
    threshold: float,
    max_height: int = 1080,
) -> tuple[int, int, int, int] | None:
    if image.shape[0] > max_height:
        scale = max_height / image.shape[0]
        image = cv2.resize(image, (int(scale * image.shape[1]), max_height))
    else:
        scale = 1.0
    detector.setInputSize((image.shape[1], image.shape[0]))
    detected = detector.detect(image)
    if detected is None or detected[1] is None:
        return None
    best: tuple[int, int, int, int] | None = None
    for potential in detected[1]:
        if float(potential[-1]) < threshold:
            continue
        x, y, width, height = potential[0:4]
        box = (
            max(0, int(x / scale)),
            max(0, int(y / scale)),
            max(0, int((x + width) / scale)),
            max(0, int((y + height) / scale)),
        )
        if best is None or area(box) > area(best):
            best = box
    return best


def crop_yuv_region_to_bgr(
    frame: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Convert only the square I420 region that owns ``box`` to BGR."""
    height = frame.shape[0] * 2 // 3
    width = frame.shape[1]
    longest = max(4, box[2] - box[0], box[3] - box[1])
    model_size = max(4, longest // 4 * 4)
    region = calculate_region(
        (height, width), *box, model_size=model_size, multiplier=1.0
    )
    normalized_region = (
        int(region[0]),
        int(region[1]),
        int(region[2]),
        int(region[3]),
    )
    return yuv_region_2_bgr(frame, region), normalized_region


class FaceRecognitionPipeline:
    """Two capture workers, two CPU preprocessors, and one batch executor."""

    def __init__(
        self,
        recognizer: Any,
        detector_factory: Callable[[], cv2.FaceDetectorYN] = create_yunet_detector,
        quality_selector: QualitySelector | None = None,
        recognition_lifecycle: RecognitionLifecycle | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.quality_selector = quality_selector
        self.recognition_lifecycle = recognition_lifecycle or RecognitionLifecycle()
        self.metrics: Counter[str] = Counter()
        self._stop = threading.Event()
        self._active_lock = threading.Lock()
        self._active_keys: set[FaceKey] = set()
        self._retry_lock = threading.Lock()
        self._retry_candidates: dict[FaceKey, list[FaceCandidate]] = {}
        self._retry_prepared: dict[FaceKey, list[PreparedFaceCandidate]] = {}
        self._finalized_keys: set[FaceKey] = set()
        self.capture_store = LatestFaceCandidateStore[FaceCaptureRequest](
            on_drop=self._drop_request
        )
        self.candidate_store = LatestFaceCandidateStore[FaceCandidate](
            on_drop=self._drop_candidate
        )
        self.prepared_store = LatestFaceCandidateStore[PreparedFaceCandidate](
            on_drop=self._drop_prepared, prefer_quality=True
        )
        self._results: queue.Queue[FaceRecognitionOutcome] = queue.Queue(maxsize=32)
        self._threads: list[threading.Thread] = []
        for index in range(FACE_CAPTURE_WORKERS):
            detector = detector_factory()
            self._threads.append(
                threading.Thread(
                    target=self._capture_loop,
                    args=(detector,),
                    daemon=True,
                    name=f"face_capture_{index}",
                )
            )
        for index in range(FACE_PREPROCESS_WORKERS):
            landmark_detector = self.recognizer.create_landmark_detector()
            self._threads.append(
                threading.Thread(
                    target=self._preprocess_loop,
                    args=(landmark_detector,),
                    daemon=True,
                    name=f"face_preprocess_{index}",
                )
            )
        self._threads.append(
            threading.Thread(
                target=self._recognition_loop,
                daemon=True,
                name="face_recognition_executor",
            )
        )
        for thread in self._threads:
            thread.start()

    def submit(self, request: FaceCaptureRequest) -> bool:
        key = RecognitionKey(
            "face", request.camera, request.event_id, request.generation
        )
        if request.key in self._finalized_keys or self.recognition_lifecycle.is_terminal(key):
            request.evidence_lease.release()
            self.metrics["terminal_skip"] += 1
            return False
        return self.capture_store.submit(request)

    def finalize(self, key: FaceKey) -> None:
        """Close passage admission and dispatch its best retained candidate."""
        with self._retry_lock:
            self._finalized_keys.add(key)
        self.retry(key)

    def expire(self, key: FaceKey) -> None:
        self.capture_store.remove(key, "track_ended")
        self.candidate_store.remove(key, "track_ended")
        self.prepared_store.remove(key, "track_ended")
        self._release_retry_key(key)
        with self._retry_lock:
            self._finalized_keys.discard(key)

    def expire_missing(self, camera: str, active_ids: set[str]) -> None:
        self.capture_store.remove_camera_missing(camera, active_ids)
        self.candidate_store.remove_camera_missing(camera, active_ids)
        self.prepared_store.remove_camera_missing(camera, active_ids)
        with self._retry_lock:
            missing = {
                key
                for key in (*self._retry_candidates, *self._retry_prepared)
                if key[0] == camera and key[1] not in active_ids
            }
        for key in missing:
            self._release_retry_key(key)

    def retry(self, key: FaceKey) -> bool:
        """Schedule the best retained independent candidate after a failed vote."""
        with self._retry_lock:
            prepared = self._retry_prepared.get(key, [])
            if prepared:
                prepared_item = max(
                    prepared, key=LatestFaceCandidateStore._quality_rank
                )
                prepared.remove(prepared_item)
                if not prepared:
                    self._retry_prepared.pop(key, None)
                return self.prepared_store.submit(prepared_item)
            else:
                candidates = self._retry_candidates.get(key, [])
                if not candidates:
                    return False
                candidate_item = max(
                    candidates, key=LatestFaceCandidateStore._quality_rank
                )
                candidates.remove(candidate_item)
                if not candidates:
                    self._retry_candidates.pop(key, None)
                return self.candidate_store.submit(candidate_item)

    def drain_results(self) -> list[FaceRecognitionOutcome]:
        results: list[FaceRecognitionOutcome] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def pending_count(self) -> int:
        with self._active_lock:
            keys = set(self._active_keys)
        keys.update(self.capture_store.keys())
        keys.update(self.candidate_store.keys())
        keys.update(self.prepared_store.keys())
        with self._retry_lock:
            keys.update(self._retry_candidates)
            keys.update(self._retry_prepared)
        return len(keys)

    def has_pending(self, key: FaceKey) -> bool:
        """Return whether this passage still owns queued or active compute."""
        with self._active_lock:
            if key in self._active_keys:
                return True
        if any(
            key in store.keys()
            for store in (self.capture_store, self.candidate_store, self.prepared_store)
        ):
            return True
        with self._retry_lock:
            return key in self._retry_candidates or key in self._retry_prepared

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        for store in (self.capture_store, self.candidate_store, self.prepared_store):
            for key in store.keys():
                store.remove(key, "shutdown")
        with self._retry_lock:
            retry_keys = set(self._retry_candidates) | set(self._retry_prepared)
        for key in retry_keys:
            self._release_retry_key(key)
        for outcome in self.drain_results():
            outcome.candidate.evidence.release()

    def _capture_loop(self, detector: cv2.FaceDetectorYN) -> None:
        while not self._stop.is_set():
            with self._active_lock:
                excluded = set(self._active_keys)
            jobs = self.capture_store.take_fair(1, 0.1, excluded=excluded)
            if not jobs:
                continue
            job = jobs[0]
            with self._active_lock:
                self._active_keys.add(job.key)
            started = time.monotonic()
            try:
                candidate = self._capture(job, detector)
            except Exception:
                logger.exception("Face capture failed for %s/%s", *job.key)
                self.metrics["capture_error"] += 1
                job.evidence_lease.release()
                self._release(job.key)
                continue
            if candidate is None:
                self.metrics["no_face"] += 1
                job.evidence_lease.release()
                self._release(job.key)
                continue
            candidate = replace(
                candidate, capture_ms=(time.monotonic() - started) * 1000
            )
            self.metrics["captured"] += 1
            self.candidate_store.submit(candidate)
            job.evidence_lease.release()

    def _capture(
        self, job: FaceCaptureRequest, detector: cv2.FaceDetectorYN
    ) -> FaceCandidate | None:
        frame = job.evidence_lease.frame
        if job.requires_face_detection:
            person, region = crop_yuv_region_to_bgr(frame, job.person_box)
            local_box = detect_largest_face(detector, person, job.detection_threshold)
            if local_box is None:
                return None
            face_box = (
                local_box[0] + region[0],
                local_box[1] + region[1],
                local_box[2] + region[0],
                local_box[3] + region[1],
            )
            face_frame = person[
                max(0, local_box[1]) : min(person.shape[0], local_box[3]),
                max(0, local_box[0]) : min(person.shape[1], local_box[2]),
            ].copy()
        else:
            if job.attribute_face_box is None:
                return None
            face_box = job.attribute_face_box
            region_frame, region = crop_yuv_region_to_bgr(frame, face_box)
            local_box = (
                face_box[0] - region[0],
                face_box[1] - region[1],
                face_box[2] - region[0],
                face_box[3] - region[1],
            )
            face_frame = region_frame[
                max(0, local_box[1]) : min(region_frame.shape[0], local_box[3]),
                max(0, local_box[0]) : min(region_frame.shape[1], local_box[2]),
            ].copy()
        if area(face_box) < job.min_area or face_frame.size == 0:
            self.metrics["too_small_or_empty"] += 1
            return None
        if self.quality_selector is None:
            raise RuntimeError("Face quality selector is unavailable")
        evidence = self.quality_selector.select(
            task="face",
            camera=job.camera,
            track_id=job.event_id,
            generation=job.generation,
            frame_ref=job.evidence_lease.ref,
            object_bbox=job.person_box,
            detail_bbox=face_box,
            detail_frame=face_frame,
            thresholds=job.quality_thresholds,
            top_k=job.top_k,
            enabled=job.quality_enabled,
            detector_score=job.detector_score,
        )
        if evidence is None:
            self.metrics["quality_rejected"] += 1
            return None
        passage_trace(
            "recognition_candidate",
            task="face",
            camera=job.camera,
            passage_id=job.event_id,
            recognition_passage_id=job.event_id,
            track_id=job.event_id,
            raw_track_lineage=[job.event_id],
            generation=job.generation,
            candidate_id=evidence.candidate_id,
            evidence_id=evidence.frame_ref.identity,
            frame_id=evidence.frame_ref.frame_id,
            frame_time=job.frame_time,
            bbox=list(face_box),
            object_box=list(job.person_box),
            quality_score=evidence.quality_score,
            quality_components=evidence.quality_components,
            admitted=True,
        )
        return FaceCandidate(
            job, face_box, face_frame, 0.0, evidence.quality_score, evidence
        )

    def _preprocess_loop(self, landmark_detector: Any) -> None:
        while not self._stop.is_set():
            candidates = self.candidate_store.take_fair(1, 0.1)
            if not candidates:
                continue
            candidate = candidates[0]
            started = time.monotonic()
            try:
                aligned, blur = self.recognizer.prepare_face(
                    candidate.face_frame, landmark_detector
                )
            except Exception:
                logger.exception("Face alignment failed for %s/%s", *candidate.key)
                self.metrics["alignment_error"] += 1
                candidate.evidence.release()
                self._release(candidate.key)
                continue
            self.prepared_store.submit(
                PreparedFaceCandidate(
                    candidate,
                    aligned,
                    blur,
                    (time.monotonic() - started) * 1000,
                    time.monotonic(),
                )
            )
            self._release(candidate.key)

    def _recognition_loop(self) -> None:
        while not self._stop.is_set():
            batch = self.prepared_store.take_fair(
                FACE_BATCH_SIZE,
                0.1,
                flush_seconds=FACE_BATCH_FLUSH_SECONDS,
                per_item_flush=True,
            )
            if not batch:
                continue
            admitted: list[tuple[PreparedFaceCandidate, RecognitionAttemptLease]] = []
            for item in batch:
                candidate = item.candidate
                request = candidate.request
                with self._retry_lock:
                    finalized = item.key in self._finalized_keys
                if not finalized:
                    if not self._reserve_prepared_retry(item, request.top_k):
                        candidate.evidence.release()
                    self._release(item.key)
                    continue
                key = RecognitionKey(
                    "face", request.camera, request.event_id, request.generation
                )
                lease, reason = self.recognition_lifecycle.begin_attempt(
                    key,
                    candidate_id=candidate.evidence.candidate_id,
                    frame_time=request.frame_time,
                    detail_bbox=candidate.face_box,
                    quality_score=candidate.evidence.quality_score,
                    policy=request.lifecycle_policy,
                )
                if lease is None:
                    self.metrics[f"lifecycle_{reason}"] += 1
                    candidate.evidence.release()
                    self._release(item.key)
                    if reason == "attempt_budget_exhausted":
                        self.recognition_lifecycle.terminal(
                            key,
                            RecognitionStatus.EXHAUSTED,
                            "insufficient_quality",
                        )
                    continue
                admitted.append((item, lease))
            if not admitted:
                continue
            started = time.monotonic()
            try:
                prepared_batch = [
                    (item.aligned_face, item.blur_reduction) for item, _ in admitted
                ]
                if hasattr(self.recognizer, "classify_prepared_top2_batch"):
                    classified = self.recognizer.classify_prepared_top2_batch(
                        prepared_batch
                    )
                else:
                    classified = self.recognizer.classify_prepared_batch(
                        prepared_batch
                    )
            except Exception:
                logger.exception("Batched face recognition failed")
                self.metrics["embedding_error"] += len(admitted)
                for item, lease in admitted:
                    self.recognition_lifecycle.complete_attempt(
                        lease, reason="inference_error"
                    )
                    if (
                        lease.attempt_index
                        >= item.candidate.request.lifecycle_policy.max_attempts
                    ):
                        self.recognition_lifecycle.terminal(
                            lease.key,
                            RecognitionStatus.EXHAUSTED,
                            "insufficient_quality",
                        )
                    item.candidate.evidence.release()
                    self._release(item.key)
                continue
            embedding_ms = (time.monotonic() - started) * 1000
            completed = time.monotonic()
            for index, (item, lease) in enumerate(admitted):
                result = classified[index] if index < len(classified) else None
                self._release(item.key)
                if result is None:
                    self.metrics["classifier_unavailable"] += 1
                    self.recognition_lifecycle.complete_attempt(
                        lease, reason="no_result"
                    )
                    if (
                        lease.attempt_index
                        >= item.candidate.request.lifecycle_policy.max_attempts
                    ):
                        self.recognition_lifecycle.terminal(
                            lease.key,
                            RecognitionStatus.EXHAUSTED,
                            "insufficient_quality",
                        )
                    item.candidate.evidence.release()
                    continue
                top1_label = (
                    result.top1_label if hasattr(result, "top1_label") else result[0]
                )
                top1_score = float(
                    result.top1_score if hasattr(result, "top1_score") else result[1]
                )
                top2_label = (
                    result.top2_label if hasattr(result, "top2_label") else None
                )
                top2_score = float(
                    result.top2_score if hasattr(result, "top2_score") else 0.0
                )
                if not self.recognition_lifecycle.complete_attempt(
                    lease,
                    result=top1_label,
                    confidence=top1_score,
                ):
                    self.metrics["stale_lifecycle_result"] += 1
                    item.candidate.evidence.release()
                    continue
                outcome = FaceRecognitionOutcome(
                    item.candidate,
                    top1_label,
                    top1_score,
                    top2_label,
                    top2_score,
                    len(admitted),
                    max(0.0, (started - item.prepared_monotonic) * 1000),
                    item.alignment_ms,
                    embedding_ms,
                    completed,
                    lease,
                )
                try:
                    self._results.put_nowait(outcome)
                except queue.Full:
                    try:
                        dropped = self._results.get_nowait()
                        dropped.candidate.evidence.release()
                    except queue.Empty:
                        pass
                    self._results.put_nowait(outcome)
                    self.metrics["result_overwritten"] += 1
            self.metrics["batches"] += 1
            self.metrics["batch_candidates"] += len(admitted)

    def _release(self, key: FaceKey) -> None:
        with self._active_lock:
            self._active_keys.discard(key)

    @staticmethod
    def _candidate_identity(item: FaceCandidate | PreparedFaceCandidate) -> str:
        candidate = item.candidate if isinstance(item, PreparedFaceCandidate) else item
        return candidate.evidence.candidate_id

    def _reserve_retry(
        self, item: FaceCandidate | PreparedFaceCandidate
    ) -> bool:
        candidate = item.candidate if isinstance(item, PreparedFaceCandidate) else item
        limit = candidate.request.top_k
        if isinstance(item, PreparedFaceCandidate):
            return self._reserve_prepared_retry(item, limit)
        return self._reserve_candidate_retry(item, limit)

    def _reserve_prepared_retry(
        self, item: PreparedFaceCandidate, limit: int
    ) -> bool:
        with self._retry_lock:
            retained = self._retry_prepared.setdefault(item.key, [])
            identity = self._candidate_identity(item)
            if any(self._candidate_identity(peer) == identity for peer in retained):
                return False
            retained.append(item)
            retained.sort(key=LatestFaceCandidateStore._quality_rank, reverse=True)
            dropped = retained[limit:]
            del retained[limit:]
        for peer in dropped:
            peer.candidate.evidence.release()
        return all(peer is not item for peer in dropped)

    def _reserve_candidate_retry(self, item: FaceCandidate, limit: int) -> bool:
        with self._retry_lock:
            retained = self._retry_candidates.setdefault(item.key, [])
            identity = self._candidate_identity(item)
            if any(self._candidate_identity(peer) == identity for peer in retained):
                return False
            retained.append(item)
            retained.sort(key=LatestFaceCandidateStore._quality_rank, reverse=True)
            dropped = retained[limit:]
            del retained[limit:]
        for peer in dropped:
            peer.evidence.release()
        return all(peer is not item for peer in dropped)

    def _release_retry_key(self, key: FaceKey) -> None:
        with self._retry_lock:
            candidates = self._retry_candidates.pop(key, [])
            prepared = self._retry_prepared.pop(key, [])
        for item in candidates:
            item.evidence.release()
        for item in prepared:
            item.candidate.evidence.release()

    def _drop_request(self, item: FaceCaptureRequest, reason: str) -> None:
        item.evidence_lease.release()
        self.metrics[f"drop_{reason}"] += 1

    def _drop_candidate(self, item: FaceCandidate, reason: str) -> None:
        if reason not in {"replaced", "lower_quality"} or not self._reserve_retry(item):
            item.evidence.release()
        self.metrics[f"drop_{reason}"] += 1
        self._release(item.key)

    def _drop_prepared(self, item: PreparedFaceCandidate, reason: str) -> None:
        if reason not in {"replaced", "lower_quality"} or not self._reserve_retry(item):
            item.candidate.evidence.release()
        self.metrics[f"drop_{reason}"] += 1
        self._release(item.key)
