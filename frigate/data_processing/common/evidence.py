"""Bounded raw-frame ownership shared by realtime enrichment pipelines."""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np


class EvidenceSourceRole(str, Enum):
    detect = "detect"
    evidence = "evidence"
    record = "record"


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Stable reference to one raw I420 frame in an evidence ring."""

    camera: str
    source_role: EvidenceSourceRole
    frame_id: str
    frame_time: float
    width: int
    height: int

    @property
    def identity(self) -> str:
        return (
            f"{self.camera}:{self.source_role.value}:{self.frame_id}:"
            f"{self.frame_time:.6f}:{self.width}x{self.height}"
        )


@dataclass(frozen=True, slots=True)
class EvidenceBufferPolicy:
    window_seconds: float = 3.0
    max_bytes: int = 32 * 1024 * 1024
    sample_fps: float = 5.0


@dataclass(slots=True)
class _EvidenceEntry:
    ref: FrameRef
    frame: np.ndarray
    size_bytes: int
    pins: int = 0
    indexed: bool = True


class EvidenceLease:
    """A releasable pin that keeps an evidence frame resident."""

    __slots__ = ("_entry_key", "_released", "_ring")

    def __init__(self, ring: EvidenceRingBuffer, entry_key: str) -> None:
        self._ring = ring
        self._entry_key = entry_key
        self._released = False

    @property
    def ref(self) -> FrameRef:
        return self._ring._lease_ref(self._entry_key, self._released)

    @property
    def frame(self) -> np.ndarray:
        return self._ring._lease_frame(self._entry_key, self._released)

    @property
    def released(self) -> bool:
        return self._released

    def fork(self) -> EvidenceLease:
        if self._released:
            raise RuntimeError("evidence lease has been released")
        lease = self._ring.acquire(self.ref)
        if lease is None:
            raise RuntimeError("evidence frame has expired")
        return lease

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._ring._release(self._entry_key)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Task candidate whose lineage and raw frame are owned together."""

    candidate_id: str
    task: str
    camera: str
    track_id: str
    generation: int
    frame_ref: FrameRef
    object_bbox: tuple[int, int, int, int] | None
    detail_bbox: tuple[int, int, int, int]
    quality_score: float
    quality_components: dict[str, float]
    reject_reasons: tuple[str, ...]
    unavailable_metrics: tuple[str, ...]
    source_role: EvidenceSourceRole
    lease: EvidenceLease

    @property
    def frame(self) -> np.ndarray:
        return self.lease.frame

    def fork(self) -> EvidenceCandidate:
        return EvidenceCandidate(
            candidate_id=self.candidate_id,
            task=self.task,
            camera=self.camera,
            track_id=self.track_id,
            generation=self.generation,
            frame_ref=self.frame_ref,
            object_bbox=self.object_bbox,
            detail_bbox=self.detail_bbox,
            quality_score=self.quality_score,
            quality_components=dict(self.quality_components),
            reject_reasons=self.reject_reasons,
            unavailable_metrics=self.unavailable_metrics,
            source_role=self.source_role,
            lease=self.lease.fork(),
        )

    def release(self) -> None:
        self.lease.release()


class EvidenceRingBuffer:
    """Per-camera time/byte bounded I420 ring with pinned-frame accounting."""

    def __init__(
        self,
        policies: dict[str, EvidenceBufferPolicy] | None = None,
        default_policy: EvidenceBufferPolicy | None = None,
    ) -> None:
        self._policies = policies or {}
        self._default_policy = default_policy or EvidenceBufferPolicy()
        self._entries: dict[str, _EvidenceEntry] = {}
        self._index: dict[tuple[str, str], str] = {}
        self._camera_order: dict[str, OrderedDict[str, None]] = {}
        self._camera_bytes: Counter[str] = Counter()
        self._last_sample_time: dict[str, float] = {}
        self._last_reject_reason: dict[str, str] = {}
        self._counters: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._closed = False

    def _policy(self, camera: str) -> EvidenceBufferPolicy:
        return self._policies.get(camera, self._default_policy)

    def ingest(
        self,
        camera: str,
        source_role: EvidenceSourceRole | str,
        frame_id: str,
        frame_time: float,
        frame: np.ndarray,
    ) -> FrameRef | None:
        """Copy one sampled frame, or return the existing reference on dedupe."""
        role = EvidenceSourceRole(source_role)
        identity = f"{role.value}:{frame_id}:{frame_time:.6f}"
        index_key = (camera, identity)
        with self._lock:
            if self._closed:
                return None
            existing_key = self._index.get(index_key)
            if existing_key is not None:
                self._counters["deduped"] += 1
                return self._entries[existing_key].ref

            policy = self._policy(camera)
            previous_time = self._last_sample_time.get(camera)
            minimum_gap = 1.0 / policy.sample_fps
            if (
                previous_time is not None
                and frame_time >= previous_time
                and frame_time - previous_time + 1e-9 < minimum_gap
            ):
                self._counters["sampled_drops"] += 1
                self._last_reject_reason[camera] = "sample_fps"
                return None

            self._evict_expired(camera, frame_time, policy.window_seconds)
            size_bytes = int(frame.nbytes)
            if size_bytes > policy.max_bytes:
                self._counters["pinned_capacity_drops"] += 1
                self._last_reject_reason[camera] = "buffer_capacity"
                return None
            while self._camera_bytes[camera] + size_bytes > policy.max_bytes:
                if not self._evict_oldest_unpinned(camera, "capacity_evictions"):
                    self._counters["pinned_capacity_drops"] += 1
                    self._last_reject_reason[camera] = "buffer_capacity"
                    return None

            owned = np.ascontiguousarray(frame).copy()
            owned.flags.writeable = False
            height = int(owned.shape[0] * 2 // 3)
            width = int(owned.shape[1])
            ref = FrameRef(camera, role, identity, float(frame_time), width, height)
            entry_key = ref.identity
            self._entries[entry_key] = _EvidenceEntry(ref, owned, size_bytes)
            self._index[index_key] = entry_key
            self._camera_order.setdefault(camera, OrderedDict())[entry_key] = None
            self._camera_bytes[camera] += size_bytes
            self._last_sample_time[camera] = frame_time
            self._counters["ingested"] += 1
            self._last_reject_reason.pop(camera, None)
            return ref

    def last_reject_reason(self, camera: str) -> str | None:
        with self._lock:
            return self._last_reject_reason.get(camera)

    def acquire(self, ref: FrameRef) -> EvidenceLease | None:
        with self._lock:
            entry = self._entries.get(ref.identity)
            if entry is None:
                self._counters["misses"] += 1
                return None
            entry.pins += 1
            return EvidenceLease(self, ref.identity)

    def expire(self, now_by_camera: dict[str, float]) -> None:
        with self._lock:
            for camera, now in now_by_camera.items():
                self._evict_expired(camera, now, self._policy(camera).window_seconds)

    def _evict_expired(self, camera: str, now: float, window: float) -> None:
        order = self._camera_order.get(camera)
        if not order:
            return
        for entry_key in list(order):
            entry = self._entries[entry_key]
            if now < entry.ref.frame_time or now - entry.ref.frame_time <= window:
                continue
            self._unindex(entry_key, "time_evictions")

    def _evict_oldest_unpinned(self, camera: str, counter: str) -> bool:
        order = self._camera_order.get(camera)
        if not order:
            return False
        for entry_key in list(order):
            if self._entries[entry_key].pins == 0:
                self._unindex(entry_key, counter)
                return True
        return False

    def _unindex(self, entry_key: str, counter: str) -> None:
        entry = self._entries[entry_key]
        if entry.indexed:
            entry.indexed = False
            self._index.pop(
                (entry.ref.camera, entry.ref.frame_id),
                None,
            )
            self._camera_order.get(entry.ref.camera, {}).pop(entry_key, None)
            self._counters[counter] += 1
        if entry.pins == 0:
            self._drop_entry(entry_key)

    def _drop_entry(self, entry_key: str) -> None:
        entry = self._entries.pop(entry_key, None)
        if entry is not None:
            self._camera_bytes[entry.ref.camera] -= entry.size_bytes

    def _lease_ref(self, entry_key: str, released: bool) -> FrameRef:
        with self._lock:
            if released or entry_key not in self._entries:
                raise RuntimeError("evidence frame has expired")
            return self._entries[entry_key].ref

    def _lease_frame(self, entry_key: str, released: bool) -> np.ndarray:
        with self._lock:
            if released or entry_key not in self._entries:
                raise RuntimeError("evidence frame has expired")
            return self._entries[entry_key].frame

    def _release(self, entry_key: str) -> None:
        with self._lock:
            entry = self._entries.get(entry_key)
            if entry is None:
                return
            entry.pins = max(0, entry.pins - 1)
            if entry.pins == 0 and not entry.indexed:
                self._drop_entry(entry_key)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cameras: dict[str, dict[str, int]] = {}
            for camera in set(self._camera_bytes) | set(self._camera_order):
                resident = [e for e in self._entries.values() if e.ref.camera == camera]
                cameras[camera] = {
                    "frames": len(resident),
                    "bytes": int(self._camera_bytes[camera]),
                    "pinned": sum(1 for entry in resident if entry.pins > 0),
                    "leases": sum(entry.pins for entry in resident),
                }
            return {"cameras": cameras, **dict(self._counters)}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for entry_key in list(self._entries):
                entry = self._entries[entry_key]
                entry.indexed = False
                if entry.pins == 0:
                    self._drop_entry(entry_key)
            self._index.clear()
            self._camera_order.clear()
