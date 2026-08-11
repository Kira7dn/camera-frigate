"""Bounded raw I420 evidence used by the external recognition service."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..contracts import TrackedObservation

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RawI420Evidence:
    evidence_id: str
    data: bytes
    shape: tuple[int, ...]
    dtype: str
    layout: str
    byte_length: int
    expiry_unix_ms: int

    def validate(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if self.layout != "I420":
            raise ValueError("unsupported evidence layout")
        if self.dtype != "uint8":
            raise ValueError("unsupported evidence dtype")
        if len(self.shape) != 2 or any(value <= 0 for value in self.shape):
            raise ValueError("invalid evidence shape")
        if self.byte_length != len(self.data):
            raise ValueError("evidence byte length mismatch")
        if self.byte_length > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence exceeds byte limit")
        expected = int(np.prod(self.shape, dtype=np.int64))
        if expected != self.byte_length:
            raise ValueError("evidence shape does not match byte length")
        if self.expiry_unix_ms <= int(time.time() * 1000):
            raise ValueError("evidence expired")


class RawI420EvidenceResolver:
    """Resolve validated request-owned bytes without filesystem access."""

    @contextmanager
    def resolve(self, observation: TrackedObservation):
        evidence = observation.evidence_ref
        if not isinstance(evidence, RawI420Evidence):
            raise TypeError("raw I420 evidence is required")
        evidence.validate()
        yield np.frombuffer(evidence.data, dtype=np.uint8).reshape(evidence.shape)

    def stats(self) -> dict[str, int]:
        return {"pinned": 0}
