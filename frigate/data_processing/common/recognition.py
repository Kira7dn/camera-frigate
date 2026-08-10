"""Shared bounded lifecycle for realtime recognition tasks."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

ImageRank = tuple[float, float, float, float, float, int, str]


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


@dataclass(frozen=True, slots=True)
class RecognitionOutcome:
    """One candidate result with complete lineage and a deterministic rank."""

    task: str
    candidate_id: str
    image_rank: ImageRank
    payload: Any
    valid: bool
    result_rank: tuple[Any, ...]
    invalid_reason: str | None = None
    # Kept as explicit metadata so a reducer cannot accidentally publish a
    # payload from one candidate with the lineage of another.  Adapters that
    # already carry a richer payload may leave this unset for compatibility.
    frame_id: str | None = None
    detail_bbox: tuple[int, int, int, int] | None = None


class BestResultReducer:
    """Keep at most three outcomes and select the highest-ranked valid one."""

    def __init__(self, task: str, max_outcomes: int = 3) -> None:
        if task not in {"lpr", "face"}:
            raise ValueError("task must be 'lpr' or 'face'")
        if not 1 <= max_outcomes <= 3:
            raise ValueError("max_outcomes must be between one and three")
        self.task = task
        self.max_outcomes = max_outcomes
        self._outcomes: list[RecognitionOutcome] = []

    def add(self, outcome: RecognitionOutcome) -> None:
        if outcome.task != self.task:
            raise ValueError("outcome task does not match reducer task")
        if any(item.candidate_id == outcome.candidate_id for item in self._outcomes):
            raise ValueError("candidate outcome may only be added once")
        if len(self._outcomes) >= self.max_outcomes:
            raise ValueError("outcome budget exhausted")
        self._outcomes.append(outcome)

    @property
    def outcomes(self) -> tuple[RecognitionOutcome, ...]:
        return tuple(self._outcomes)

    def winner(self) -> RecognitionOutcome | None:
        valid = [outcome for outcome in self._outcomes if outcome.valid]
        return max(valid, key=lambda outcome: outcome.result_rank, default=None)

    def exhausted_reason(self) -> str:
        if self.task == "lpr":
            return "insufficient_quality"
        reasons = {outcome.invalid_reason for outcome in self._outcomes}
        if "ambiguous_identity" in reasons:
            return "ambiguous_identity"
        if "unknown" in reasons:
            return "unknown"
        return "insufficient_quality"


def lpr_result_outcome(
    *,
    candidate_id: str,
    image_rank: ImageRank,
    payload: Any,
    character_scores: tuple[float, ...],
    recognition_threshold: float,
    length_valid: bool,
    format_valid: bool,
    recognized_text_area: int,
    frame_id: str | None = None,
    detail_bbox: tuple[int, int, int, int] | None = None,
) -> RecognitionOutcome:
    mean_score = sum(character_scores) / len(character_scores) if character_scores else 0.0
    min_score = min(character_scores, default=0.0)
    valid = (
        bool(character_scores)
        and mean_score >= recognition_threshold
        and length_valid
        and format_valid
    )
    return RecognitionOutcome(
        task="lpr",
        candidate_id=candidate_id,
        image_rank=image_rank,
        payload=payload,
        valid=valid,
        result_rank=(
            bool(length_valid and format_valid),
            min_score,
            mean_score,
            image_rank,
            int(recognized_text_area),
            candidate_id,
        ),
        invalid_reason=None if valid else "insufficient_quality",
        frame_id=frame_id,
        detail_bbox=detail_bbox,
    )


def face_result_outcome(
    *,
    candidate_id: str,
    image_rank: ImageRank,
    payload: Any,
    top1_score: float,
    top2_score: float,
    recognition_threshold: float,
    min_identity_margin: float,
    image_quality_valid: bool,
    margin_scale: float | None = None,
    frame_id: str | None = None,
    detail_bbox: tuple[int, int, int, int] | None = None,
) -> RecognitionOutcome:
    margin = top1_score - top2_score
    scale = max(float(margin_scale or min_identity_margin), 1e-9)
    valid = (
        image_quality_valid
        and top1_score >= recognition_threshold
        and margin >= min_identity_margin
    )
    if not image_quality_valid:
        reason = "insufficient_quality"
    elif margin < min_identity_margin:
        reason = "ambiguous_identity"
    elif top1_score < recognition_threshold:
        reason = "unknown"
    else:
        reason = None
    return RecognitionOutcome(
        task="face",
        candidate_id=candidate_id,
        image_rank=image_rank,
        payload=payload,
        valid=valid,
        result_rank=(
            bool(valid),
            max(0.0, min(1.0, margin / scale)),
            top1_score,
            image_rank,
            candidate_id,
        ),
        invalid_reason=reason,
        frame_id=frame_id,
        detail_bbox=detail_bbox,
    )


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
            result.setdefault("early_stop", 0)
            return result
