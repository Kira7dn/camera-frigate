"""Bounded on-demand I420 evidence storage."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from ..domain.contracts import EvidenceReference


class EvidenceUnavailableError(KeyError):
    pass


class EvidenceCapacityError(RuntimeError):
    pass


@dataclass(slots=True)
class _Entry:
    reference: EvidenceReference
    data: bytes
    shape: tuple[int, int]
    pins: int = 0


class EvidenceRing:
    """Per-camera FIFO ring that never evicts pinned evidence."""

    def __init__(
        self,
        node_id: str,
        camera_id: str,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        ttl_seconds: float = 45,
    ) -> None:
        if max_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("evidence capacity and TTL must be positive")
        self.node_id = node_id
        self.camera_id = camera_id
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0

    @property
    def byte_size(self) -> int:
        return self._bytes

    @property
    def pinned_count(self) -> int:
        return sum(entry.pins > 0 for entry in self._entries.values())

    def put(
        self,
        *,
        stream_epoch: str,
        frame_seq: int,
        data: bytes,
        shape: tuple[int, int],
        now: float | None = None,
    ) -> EvidenceReference:
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("I420 shape must be positive height/width")
        expected = shape[0] * shape[1] * 3 // 2
        if len(data) != expected:
            raise ValueError("I420 byte length does not match shape")
        if len(data) > self.max_bytes:
            raise EvidenceCapacityError("frame exceeds per-camera evidence budget")
        now = time.time() if now is None else now
        self.expire(now)
        reference = EvidenceReference.create(
            node_id=self.node_id,
            camera_id=self.camera_id,
            stream_epoch=stream_epoch,
            frame_seq=frame_seq,
            data=data,
            expiry_unix_ms=int((now + self.ttl_seconds) * 1000),
        )
        self._evict_for(len(data))
        self._entries[reference.evidence_id] = _Entry(reference, data, shape)
        self._bytes += len(data)
        return reference

    def get(self, evidence_id: str, now: float | None = None) -> tuple[EvidenceReference, bytes, tuple[int, int]]:
        now = time.time() if now is None else now
        entry = self._entries.get(evidence_id)
        if entry is None:
            raise EvidenceUnavailableError(evidence_id)
        if entry.reference.expiry_unix_ms <= int(now * 1000) and entry.pins == 0:
            self._remove(evidence_id)
            raise EvidenceUnavailableError(evidence_id)
        if hashlib.sha256(entry.data).hexdigest() != entry.reference.sha256:
            raise EvidenceUnavailableError("evidence checksum mismatch")
        self._entries.move_to_end(evidence_id)
        return entry.reference, entry.data, entry.shape

    def pin(self, evidence_id: str) -> EvidenceReference:
        entry = self._entries.get(evidence_id)
        if entry is None:
            raise EvidenceUnavailableError(evidence_id)
        entry.pins += 1
        reference = entry.reference
        entry.reference = EvidenceReference(
            reference.evidence_id,
            reference.byte_length,
            reference.sha256,
            reference.expiry_unix_ms,
            True,
        )
        return entry.reference

    def release(self, evidence_id: str) -> None:
        entry = self._entries.get(evidence_id)
        if entry is None:
            return
        entry.pins = max(0, entry.pins - 1)

    def expire(self, now: float | None = None) -> int:
        now_ms = int((time.time() if now is None else now) * 1000)
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.pins == 0 and entry.reference.expiry_unix_ms <= now_ms
        ]
        for key in expired:
            self._remove(key)
        return len(expired)

    def _evict_for(self, byte_length: int) -> None:
        while self._bytes + byte_length > self.max_bytes:
            victim = next(
                (key for key, entry in self._entries.items() if entry.pins == 0),
                None,
            )
            if victim is None:
                raise EvidenceCapacityError("evidence ring is full of pinned frames")
            self._remove(victim)

    def _remove(self, evidence_id: str) -> None:
        entry = self._entries.pop(evidence_id, None)
        if entry is not None:
            self._bytes -= len(entry.data)
