"""Bounded multi-camera capture and batched face recognition pipeline."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Callable, Generic, TypeVar

import cv2
import numpy as np

from frigate.const import MODEL_CACHE_DIR
from frigate.data_processing.common.evidence import EvidenceCandidate, EvidenceLease
from frigate.data_processing.common.quality import QualitySelector, QualityThresholds
from frigate.util.image import area, calculate_region, yuv_region_2_bgr

logger = logging.getLogger(__name__)

FACE_CAPTURE_WORKERS = 2
FACE_PREPROCESS_WORKERS = 2
FACE_BATCH_SIZE = 4
FACE_BATCH_FLUSH_SECONDS = 0.1
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
    detector_score: float | None = None
    quality: float = 0.0

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


@dataclass(frozen=True)
class FaceRecognitionOutcome:
    candidate: FaceCandidate
    sub_label: str
    score: float
    batch_size: int
    batch_wait_ms: float
    alignment_ms: float
    embedding_ms: float
    completed_monotonic: float


T = TypeVar("T")


class LatestFaceCandidateStore(Generic[T]):
    """Thread-safe latest-only keyed store with bounded per-camera ownership."""

    def __init__(
        self,
        max_per_camera: int = MAX_TRACKS_PER_CAMERA,
        ttl_seconds: float = FACE_CANDIDATE_TTL_SECONDS,
        on_drop: Callable[[T, str], None] | None = None,
    ) -> None:
        self.max_per_camera = max_per_camera
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[FaceKey, T] = OrderedDict()
        self._condition = threading.Condition()
        self._camera_cursor = 0
        self._on_drop = on_drop

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
        with self._condition:
            previous = self._items.pop(key, None)
            if previous is not None:
                dropped.append((previous, "replaced"))
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
        return True

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
                oldest_age = max(
                    (
                        now - float(getattr(item, "created_monotonic"))
                        for item in eligible
                    ),
                    default=0.0,
                )
                if eligible and (
                    flush_seconds <= 0
                    or len(eligible) >= max_items
                    or oldest_age >= flush_seconds
                ):
                    selected = self._select_fair(eligible, max_items)
                    for item in selected:
                        self._items.pop(getattr(item, "key"), None)
                    break
                remaining = deadline - now
                if remaining <= 0:
                    selected = (
                        self._select_fair(eligible, max_items) if eligible else []
                    )
                    for item in selected:
                        self._items.pop(getattr(item, "key"), None)
                    break
                wait_for = remaining
                if eligible and flush_seconds > 0:
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
    return yuv_region_2_bgr(frame, region), tuple(int(value) for value in region)


class FaceRecognitionPipeline:
    """Two capture workers, two CPU preprocessors, and one batch executor."""

    def __init__(
        self,
        recognizer: Any,
        detector_factory: Callable[[], cv2.FaceDetectorYN] = create_yunet_detector,
        quality_selector: QualitySelector | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.quality_selector = quality_selector
        self.metrics: Counter[str] = Counter()
        self._stop = threading.Event()
        self._active_lock = threading.Lock()
        self._active_keys: set[FaceKey] = set()
        self.capture_store = LatestFaceCandidateStore[FaceCaptureRequest](
            on_drop=self._drop_request
        )
        self.candidate_store = LatestFaceCandidateStore[FaceCandidate](
            on_drop=self._drop_candidate
        )
        self.prepared_store = LatestFaceCandidateStore[PreparedFaceCandidate](
            on_drop=self._drop_prepared
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
        return self.capture_store.submit(request)

    def expire(self, key: FaceKey) -> None:
        self.capture_store.remove(key, "track_ended")
        self.candidate_store.remove(key, "track_ended")
        self.prepared_store.remove(key, "track_ended")

    def expire_missing(self, camera: str, active_ids: set[str]) -> None:
        self.capture_store.remove_camera_missing(camera, active_ids)
        self.candidate_store.remove_camera_missing(camera, active_ids)
        self.prepared_store.remove_camera_missing(camera, active_ids)

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
        return len(keys)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        for store in (self.capture_store, self.candidate_store, self.prepared_store):
            for key in store.keys():
                store.remove(key, "shutdown")
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

    def _recognition_loop(self) -> None:
        while not self._stop.is_set():
            batch = self.prepared_store.take_fair(
                FACE_BATCH_SIZE,
                0.1,
                flush_seconds=FACE_BATCH_FLUSH_SECONDS,
            )
            if not batch:
                continue
            started = time.monotonic()
            try:
                classified = self.recognizer.classify_prepared_batch(
                    [(item.aligned_face, item.blur_reduction) for item in batch]
                )
            except Exception:
                logger.exception("Batched face recognition failed")
                self.metrics["embedding_error"] += len(batch)
                for item in batch:
                    item.candidate.evidence.release()
                    self._release(item.key)
                continue
            embedding_ms = (time.monotonic() - started) * 1000
            completed = time.monotonic()
            for item, result in zip(batch, classified):
                self._release(item.key)
                if result is None:
                    self.metrics["classifier_unavailable"] += 1
                    item.candidate.evidence.release()
                    continue
                outcome = FaceRecognitionOutcome(
                    item.candidate,
                    result[0],
                    result[1],
                    len(batch),
                    max(0.0, (started - item.prepared_monotonic) * 1000),
                    item.alignment_ms,
                    embedding_ms,
                    completed,
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
            self.metrics["batch_candidates"] += len(batch)

    def _release(self, key: FaceKey) -> None:
        with self._active_lock:
            self._active_keys.discard(key)

    def _drop_request(self, item: FaceCaptureRequest, reason: str) -> None:
        item.evidence_lease.release()
        self.metrics[f"drop_{reason}"] += 1

    def _drop_candidate(self, item: FaceCandidate, reason: str) -> None:
        item.evidence.release()
        self.metrics[f"drop_{reason}"] += 1
        self._release(item.key)

    def _drop_prepared(self, item: PreparedFaceCandidate, reason: str) -> None:
        item.candidate.evidence.release()
        self.metrics[f"drop_{reason}"] += 1
        self._release(item.key)
