"""Shared bounded lifecycle for realtime recognition tasks."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecognitionStatus(str, Enum):
    SEARCHING = "SEARCHING"
    ACCEPTED = "ACCEPTED"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True, slots=True)
class RecognitionKey:
    task: str
    camera: str
    passage_id: str
    generation: int

    @property
    def track_id(self) -> str:
        """Compatibility alias; lifecycle ownership is the physical passage."""
        return self.passage_id


@dataclass(frozen=True, slots=True)
class RecognitionPolicy:
    max_attempts: int = 3
    min_candidate_interval_seconds: float = 0.4
    max_candidate_bbox_iou: float = 0.90


@dataclass(frozen=True, slots=True)
class RecognitionAttemptLease:
    key: RecognitionKey
    attempt_index: int
    candidate_id: str
    frame_time: float
    detail_bbox: tuple[int, int, int, int]
    quality_score: float
    started_monotonic: float


@dataclass(slots=True)
class RecognitionAttempt:
    attempt_index: int
    candidate_id: str
    frame_time: float
    detail_bbox: tuple[int, int, int, int]
    quality_score: float
    started_monotonic: float
    completed_monotonic: float | None = None
    result: Any = None
    confidence: float | None = None
    confidence_type: str | None = None
    latency_ms: float | None = None
    reason: str = "inference_started"


@dataclass(slots=True)
class RecognitionState:
    key: RecognitionKey
    status: RecognitionStatus = RecognitionStatus.SEARCHING
    attempts: list[RecognitionAttempt] = field(default_factory=list)
    in_flight: set[int] = field(default_factory=set)
    terminal_reason: str | None = None
    terminal_monotonic: float | None = None


class RecognitionLifecycle:
    """Thread-safe attempt accounting, diversity admission, and terminal state."""

    def __init__(self) -> None:
        self._states: dict[RecognitionKey, RecognitionState] = {}
        self._lock = threading.RLock()
        self._counters: Counter[str] = Counter()

    @staticmethod
    def _bbox_iou(
        left: tuple[int, int, int, int], right: tuple[int, int, int, int]
    ) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
        right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    def is_terminal(self, key: RecognitionKey) -> bool:
        with self._lock:
            state = self._states.get(key)
            return state is not None and state.status != RecognitionStatus.SEARCHING

    def status(self, key: RecognitionKey) -> RecognitionStatus:
        with self._lock:
            state = self._states.get(key)
            return state.status if state else RecognitionStatus.SEARCHING

    def begin_attempt(
        self,
        key: RecognitionKey,
        *,
        candidate_id: str,
        frame_time: float,
        detail_bbox: tuple[int, int, int, int],
        quality_score: float,
        policy: RecognitionPolicy,
    ) -> tuple[RecognitionAttemptLease | None, str | None]:
        """Consume budget only when the caller is immediately starting inference."""
        with self._lock:
            state = self._states.setdefault(key, RecognitionState(key))
            if state.status != RecognitionStatus.SEARCHING:
                self._counters["terminal_skips"] += 1
                return None, "terminal"
            if len(state.attempts) >= policy.max_attempts:
                self._counters["budget_skips"] += 1
                return None, "attempt_budget_exhausted"
            for previous in state.attempts:
                if previous.candidate_id == candidate_id:
                    self._counters["duplicate_skips"] += 1
                    return None, "duplicate_candidate"
                if (
                    abs(previous.frame_time - frame_time) + 1e-9
                    < policy.min_candidate_interval_seconds
                    and self._bbox_iou(previous.detail_bbox, detail_bbox)
                    > policy.max_candidate_bbox_iou + 1e-9
                ):
                    self._counters["diversity_skips"] += 1
                    return None, "insufficient_diversity"
            index = len(state.attempts) + 1
            started = time.monotonic()
            state.attempts.append(
                RecognitionAttempt(
                    index,
                    candidate_id,
                    frame_time,
                    detail_bbox,
                    quality_score,
                    started,
                )
            )
            state.in_flight.add(index)
            self._counters["attempts_started"] += 1
            self._counters[f"attempts_started:{key.task}"] += 1
            return (
                RecognitionAttemptLease(
                    key,
                    index,
                    candidate_id,
                    frame_time,
                    detail_bbox,
                    quality_score,
                    started,
                ),
                None,
            )

    def complete_attempt(
        self,
        lease: RecognitionAttemptLease,
        *,
        result: Any = None,
        confidence: float | None = None,
        confidence_type: str | None = None,
        reason: str = "inference_completed",
    ) -> bool:
        """Complete an attempt, returning false for stale or terminal results."""
        with self._lock:
            state = self._states.get(lease.key)
            if state is None or lease.attempt_index not in state.in_flight:
                self._counters["stale_results"] += 1
                return False
            state.in_flight.remove(lease.attempt_index)
            attempt = state.attempts[lease.attempt_index - 1]
            completed = time.monotonic()
            attempt.completed_monotonic = completed
            attempt.latency_ms = (completed - attempt.started_monotonic) * 1000
            attempt.result = result
            attempt.confidence = confidence
            attempt.confidence_type = confidence_type
            attempt.reason = reason
            self._counters["attempts_completed"] += 1
            self._counters[f"attempts_completed:{lease.key.task}"] += 1
            self._counters[f"compute_ms:{lease.key.task}"] += int(attempt.latency_ms)
            if state.status != RecognitionStatus.SEARCHING:
                self._counters["stale_results"] += 1
                return False
            return True

    def terminal(
        self,
        key: RecognitionKey,
        status: RecognitionStatus,
        reason: str,
    ) -> bool:
        if status == RecognitionStatus.SEARCHING:
            raise ValueError("terminal status must be ACCEPTED or EXHAUSTED")
        with self._lock:
            state = self._states.setdefault(key, RecognitionState(key))
            if state.status != RecognitionStatus.SEARCHING:
                return False
            state.status = status
            state.terminal_reason = reason
            state.terminal_monotonic = time.monotonic()
            self._counters["pending_cancellations"] += len(state.in_flight)
            state.in_flight.clear()
            self._counters[status.value.lower()] += 1
            self._counters[f"terminal_reason:{reason}"] += 1
            self._counters[f"terminal_attempts:{key.task}"] += len(state.attempts)
            self._counters[
                f"terminal_passages:{key.task}:{len(state.attempts)}_attempts"
            ] += 1
            if status == RecognitionStatus.ACCEPTED:
                self._counters["early_stop"] += 1
            return True

    def record_skip(self, task: str, reason: str) -> None:
        """Record an adapter skip without creating a per-track metric label."""
        with self._lock:
            self._counters[f"{reason}_skips"] += 1
            self._counters[f"{reason}_skips:{task}"] += 1

    def attempts(self, key: RecognitionKey) -> tuple[RecognitionAttempt, ...]:
        with self._lock:
            state = self._states.get(key)
            return tuple(state.attempts) if state else ()

    def expire(self, key: RecognitionKey, reason: str = "track_expired") -> bool:
        """Terminate and remove a track generation, invalidating in-flight leases."""
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return False
            if state.status == RecognitionStatus.SEARCHING:
                state.status = RecognitionStatus.EXHAUSTED
                state.terminal_reason = reason
                self._counters["exhausted"] += 1
                self._counters[f"terminal_reason:{reason}"] += 1
            self._counters["pending_cancellations"] += len(state.in_flight)
            del self._states[key]
            return True

    def shutdown(self) -> None:
        with self._lock:
            for key in list(self._states):
                self.expire(key, "shutdown")

    def stats(self) -> dict[str, int]:
        with self._lock:
            result = dict(self._counters)
            result["in_flight"] = sum(
                len(state.in_flight) for state in self._states.values()
            )
            result["active_lifecycles"] = sum(
                state.status == RecognitionStatus.SEARCHING
                for state in self._states.values()
            )
            result["tombstones"] = sum(
                state.status != RecognitionStatus.SEARCHING
                for state in self._states.values()
            )
            # Kept for the existing stats consumer; unlike the old value this
            # excludes non-owning terminal tombstones.
            result["active_tracks"] = result["active_lifecycles"]
            return result
