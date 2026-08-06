"""Bounded background work for event-safe face snapshots."""

import logging
import os
import re
import shutil
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FaceTrackKey = tuple[str, str]
Box = tuple[int, int, int, int]
FACE_EVENT_STAGING_DIR = "/tmp/cache/face-events"
FACE_PROCESS_INTERVAL = 0.5
EXCLUDED_FACE_DIRECTORIES = frozenset({"train", "events", "staging", "face-events"})
_LEGACY_ARTIFACT = re.compile(r"^.+-.+-\d+(?:\.\d+)?\.webp(?:\.tmp-\d+-\d+\.webp)?$")
_FACE_ATTEMPT_IMAGE_EXTENSIONS = frozenset({".webp", ".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class FaceVote:
    """One recognition vote for a face track."""

    sub_label: str
    score: float
    face_area: int


@dataclass(frozen=True)
class FaceRecognitionResult:
    """Recognition identity and snapshot candidate from the same frame."""

    camera: str
    event_id: str
    frame_time: float
    person_box: Box
    face_box: Box
    sub_label: str
    face_score: float
    artifact_path: str

    @property
    def key(self) -> FaceTrackKey:
        return (self.camera, self.event_id)

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable internal message."""
        return {
            "camera": self.camera,
            "event_id": self.event_id,
            "frame_time": self.frame_time,
            "person_box": self.person_box,
            "face_box": self.face_box,
            "sub_label": self.sub_label,
            "face_score": self.face_score,
            "artifact_path": self.artifact_path,
        }


@dataclass
class FaceTrackState:
    """All mutable recognition state for one camera and event pair."""

    last_frame_time: float
    last_box: Box
    last_snapshot_time: float
    votes: list[FaceVote]
    candidate: FaceRecognitionResult | None = None
    last_attempt_time: float = 0.0
    result_emitted: bool = False


def parse_face_attempt_filename(filename: str) -> tuple[str, str] | None:
    """Return the event id and identity encoded in a face-attempt filename."""
    path = Path(filename)
    if path.suffix.lower() not in _FACE_ATTEMPT_IMAGE_EXTENSIONS:
        return None

    parts = path.stem.rsplit("-", 3)
    if len(parts) != 4 or not parts[0]:
        return None

    event_id, timestamp, sub_label, score = parts
    try:
        float(timestamp)
        float(score)
    except ValueError:
        return None

    return event_id, sub_label


def is_unknown_face_attempt(filename: str) -> bool:
    """Return whether a face-attempt filename represents an unknown identity."""
    parsed = parse_face_attempt_filename(filename)
    return parsed is not None and parsed[1] == "unknown"


@dataclass(frozen=True)
class FaceSnapshotJob:
    """Copied recognition frame awaiting one background encoding."""

    camera: str
    event_id: str
    frame_time: float
    person_box: Box
    face_box: Box
    sub_label: str
    face_score: float
    frame: np.ndarray

    @property
    def key(self) -> FaceTrackKey:
        return (self.camera, self.event_id)


@dataclass(frozen=True)
class SnapshotCommitJob:
    """Immutable paths and result consumed by the media committer."""

    result: FaceRecognitionResult
    canonical_path: str
    thumbnail_path: str


@dataclass(frozen=True)
class SnapshotCommitted:
    """Media completion sent to the Event Maintainer DB writer."""

    result: FaceRecognitionResult
    canonical_path: str
    thumbnail_path: str

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable completion message."""
        return {
            **self.result.as_payload(),
            "canonical_path": self.canonical_path,
            "thumbnail_path": self.thumbnail_path,
        }


@dataclass(frozen=True)
class SnapshotFailed:
    """Failed media completion used to release pending active state."""

    result: FaceRecognitionResult
    reason: str

    def as_payload(self) -> dict[str, Any]:
        return {**self.result.as_payload(), "status": "failed", "reason": self.reason}


@dataclass(frozen=True)
class CleanupJob:
    """Artifact cleanup that does not consume a pending object slot."""

    paths: tuple[str, ...]


@dataclass(frozen=True)
class FaceAttemptJob:
    """Copied attempt image and bounded retention settings."""

    frame: np.ndarray
    event_id: str
    timestamp: float
    sub_label: str
    score: float
    face_dir: str
    max_files: int


def box_iou(first: Box, second: Box) -> float:
    """Return intersection over union for two pixel boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def is_track_discontinuity(previous: Box, current: Box, frame_gap: float) -> bool:
    """Detect an implausible identity-preserving track transition."""
    if frame_gap > 2.0:
        return True
    if box_iou(previous, current) >= 0.05:
        return False
    previous_center = (
        (previous[0] + previous[2]) / 2,
        (previous[1] + previous[3]) / 2,
    )
    current_center = (
        (current[0] + current[2]) / 2,
        (current[1] + current[3]) / 2,
    )
    previous_diagonal = max(
        1.0,
        ((previous[2] - previous[0]) ** 2 + (previous[3] - previous[1]) ** 2) ** 0.5,
    )
    center_distance = (
        (current_center[0] - previous_center[0]) ** 2
        + (current_center[1] - previous_center[1]) ** 2
    ) ** 0.5
    return center_distance > previous_diagonal * 1.5


class LatestPerObjectWorker:
    """Run bounded latest-per-object work with an uncounted control queue."""

    def __init__(
        self,
        handler: Callable[[Any], Any | None],
        max_objects: int = 4,
        name: str = "face_snapshot_worker",
        drop_handler: Callable[[Any], None] | None = None,
    ) -> None:
        self._handler = handler
        self._max_objects = max_objects
        self._drop_handler = drop_handler
        self._pending: OrderedDict[FaceTrackKey, Any] = OrderedDict()
        self._control: deque[Any] = deque()
        self._results: deque[Any] = deque()
        self._condition = threading.Condition()
        self._stopping = False
        self._active_key: FaceTrackKey | None = None
        self._counters: Counter[str] = Counter()
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def submit(self, key: FaceTrackKey, job: Any) -> bool:
        """Add or replace work, rejecting new objects after four slots."""
        previous = None
        with self._condition:
            if self._stopping:
                self._counters["rejected"] += 1
                return False
            keys = set(self._pending)
            if self._active_key is not None:
                keys.add(self._active_key)
            if key not in keys and len(keys) >= self._max_objects:
                self._counters["rejected"] += 1
                return False
            previous = self._pending.get(key)
            self._pending[key] = job
            self._pending.move_to_end(key)
            if previous is not None:
                self._counters["replaced"] += 1
            self._condition.notify()
        if previous is not None and self._drop_handler is not None:
            self._drop(previous)
        return True

    def submit_control(self, job: Any) -> bool:
        """Submit release or cleanup work without consuming an object slot."""
        with self._condition:
            if self._stopping:
                return False
            self._control.append(job)
            self._condition.notify()
            return True

    def drain_results(self) -> list[Any]:
        """Return all completed results without blocking."""
        with self._condition:
            results = list(self._results)
            self._results.clear()
            return results

    def stats(self) -> dict[str, int]:
        """Return bounded queue and lifecycle counters."""
        with self._condition:
            return {
                "pending": len(self._pending) + (1 if self._active_key else 0),
                **self._counters,
            }

    def stop(self) -> None:
        """Stop after discarding pending jobs and releasing their artifacts."""
        with self._condition:
            self._stopping = True
            dropped = list(self._pending.values()) + list(self._control)
            self._pending.clear()
            self._control.clear()
            self._condition.notify_all()
        if self._drop_handler is not None:
            for job in dropped:
                self._drop(job)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._control and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                if self._control:
                    key = None
                    job = self._control.popleft()
                else:
                    key, job = self._pending.popitem(last=False)
                    self._active_key = key
            try:
                result = self._handler(job)
            except Exception:
                logger.exception("Face snapshot background job failed")
                with self._condition:
                    self._counters["failed"] += 1
                if self._drop_handler is not None:
                    self._drop(job)
                result = None
            with self._condition:
                if result is not None:
                    self._results.append(result)
                    self._counters["committed"] += 1
                if key is not None:
                    self._active_key = None

    def _drop(self, job: Any) -> None:
        try:
            self._drop_handler(job)  # type: ignore[misc]
        except Exception:
            logger.exception("Face snapshot drop handler failed")


def cleanup_paths(job: CleanupJob) -> None:
    """Remove staging or rejected media paths."""
    for path in job.paths:
        if not path:
            continue
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def write_face_snapshot_artifact(
    job: FaceSnapshotJob, staging_dir: str = FACE_EVENT_STAGING_DIR
) -> FaceRecognitionResult | None:
    """Encode once and atomically publish a recognition staging artifact."""
    folder = staging_dir
    os.makedirs(folder, mode=0o700, exist_ok=True)
    artifact_path = os.path.join(
        folder, f"{job.camera}-{job.event_id}-{job.frame_time}.webp"
    )
    temporary_path = f"{artifact_path}.tmp-{os.getpid()}-{threading.get_ident()}.webp"
    try:
        bgr_frame = cv2.cvtColor(job.frame, cv2.COLOR_YUV2BGR_I420)
        if not cv2.imwrite(temporary_path, bgr_frame):
            return None
        os.replace(temporary_path, artifact_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)
    return FaceRecognitionResult(
        camera=job.camera,
        event_id=job.event_id,
        frame_time=job.frame_time,
        person_box=job.person_box,
        face_box=job.face_box,
        sub_label=job.sub_label,
        face_score=job.face_score,
        artifact_path=artifact_path,
    )


def commit_snapshot_job(job: SnapshotCommitJob) -> SnapshotCommitted:
    """Atomically replace canonical and thumbnail media before completion."""
    image = cv2.imread(job.result.artifact_path)
    if image is None:
        raise OSError("Unable to read face snapshot artifact")
    os.makedirs(os.path.dirname(job.canonical_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.thumbnail_path), exist_ok=True)
    canonical_temp = (
        f"{job.canonical_path}.tmp-{os.getpid()}-{threading.get_ident()}.webp"
    )
    thumbnail_temp = (
        f"{job.thumbnail_path}.tmp-{os.getpid()}-{threading.get_ident()}.webp"
    )
    canonical_backup = f"{job.canonical_path}.bak-{os.getpid()}-{threading.get_ident()}"
    thumbnail_backup = f"{job.thumbnail_path}.bak-{os.getpid()}-{threading.get_ident()}"
    had_canonical = os.path.isfile(job.canonical_path)
    had_thumbnail = os.path.isfile(job.thumbnail_path)
    try:
        if not cv2.imwrite(canonical_temp, image):
            raise OSError("Unable to encode canonical face snapshot")
        height, width = image.shape[:2]
        thumb_height = min(175, height)
        thumb_width = max(1, int(width * thumb_height / max(1, height)))
        thumbnail = cv2.resize(
            image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA
        )
        if not cv2.imwrite(thumbnail_temp, thumbnail):
            raise OSError("Unable to encode face snapshot thumbnail")
        if had_canonical:
            os.replace(job.canonical_path, canonical_backup)
        if had_thumbnail:
            os.replace(job.thumbnail_path, thumbnail_backup)
        os.replace(canonical_temp, job.canonical_path)
        os.replace(thumbnail_temp, job.thumbnail_path)
    except Exception:
        Path(job.canonical_path).unlink(missing_ok=True)
        Path(job.thumbnail_path).unlink(missing_ok=True)
        if had_canonical and os.path.isfile(canonical_backup):
            os.replace(canonical_backup, job.canonical_path)
        if had_thumbnail and os.path.isfile(thumbnail_backup):
            os.replace(thumbnail_backup, job.thumbnail_path)
        raise
    finally:
        Path(canonical_temp).unlink(missing_ok=True)
        Path(thumbnail_temp).unlink(missing_ok=True)
        Path(canonical_backup).unlink(missing_ok=True)
        Path(thumbnail_backup).unlink(missing_ok=True)
        Path(job.result.artifact_path).unlink(missing_ok=True)
    return SnapshotCommitted(job.result, job.canonical_path, job.thumbnail_path)


def write_face_attempt(job: FaceAttemptJob) -> None:
    """Persist and trim attempts outside the recognition thread."""
    folder = os.path.join(job.face_dir, "train")
    os.makedirs(folder, exist_ok=True)
    sub_label = job.sub_label.replace("-", "_")
    path = os.path.join(
        folder, f"{job.event_id}-{job.timestamp}-{sub_label}-{job.score}.webp"
    )
    if not cv2.imwrite(path, job.frame):
        raise OSError("Unable to encode face attempt")
    files = sorted(
        (entry for entry in Path(folder).glob("*.webp") if entry.is_file()),
        key=lambda entry: entry.stat().st_ctime,
        reverse=True,
    )
    for old_file in files[job.max_files :]:
        old_file.unlink(missing_ok=True)


def reap_stale_staging(
    staging_dir: str = FACE_EVENT_STAGING_DIR, max_age_seconds: int = 3600
) -> int:
    """Delete only contract-named temporary staging files older than one hour."""
    folder = Path(staging_dir)
    if not folder.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in folder.glob("*.webp*"):
        is_contract_staging = bool(_LEGACY_ARTIFACT.match(path.name))
        if is_contract_staging and path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def is_face_identity_directory(name: str, path: str) -> bool:
    """Return whether a child of FACE_DIR is a public identity directory."""
    return (
        bool(name)
        and not name.startswith(".")
        and name.lower() not in EXCLUDED_FACE_DIRECTORIES
        and os.path.isdir(path)
    )


def cleanup_legacy_face_events(face_dir: str) -> int:
    """Remove only the obsolete runtime-created FACE_DIR/events artifact folder."""
    legacy = Path(face_dir) / "events"
    if not legacy.is_dir():
        return 0
    entries = list(legacy.iterdir())
    if any(
        not entry.is_file() or not _LEGACY_ARTIFACT.match(entry.name)
        for entry in entries
    ):
        logger.warning("Preserving non-runtime content in legacy face events directory")
        return 0
    for entry in entries:
        entry.unlink(missing_ok=True)
    shutil.rmtree(legacy, ignore_errors=False)
    return len(entries)
