"""Tests for bounded ordered recognition execution."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

from frigate.application.recognition import (
    FacePolicy,
    LprPolicy,
    RawRecognition,
    RecognitionCore,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcomeStatus,
    RecognitionTask,
    TrackedObservation,
    TrackKey,
)
from frigate.application.recognition.executor import AsyncRecognitionExecutor


class Evidence:
    def resolve(self, observation):
        return nullcontext(observation.evidence_ref)


class Model:
    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate

    def recognize(self, task, observation, evidence):
        if self.gate is not None:
            self.gate.wait(1)
        return RawRecognition(str(evidence), 0.95, area=1000)


def core_factory(model: Model | None = None):
    return lambda: RecognitionCore(
        model or Model(),
        Evidence(),
        LprPolicy(5, 0.9),
        FacePolicy(0.8, 0.9),
    )


def observation(key: TrackKey, sequence: int) -> TrackedObservation:
    return TrackedObservation(
        RecognitionTask.LPR,
        key,
        float(sequence),
        (0, 0, 100, 100),
        evidence_ref=f"P{sequence}",
    )


def job(
    key: TrackKey,
    sequence: int,
    *,
    operation: RecognitionOperation = RecognitionOperation.OBSERVE,
    deadline: float | None = None,
) -> RecognitionJob:
    return RecognitionJob(
        f"job-{sequence}-{operation.value}",
        "client",
        "service",
        key,
        sequence,
        operation,
        observation(key, sequence)
        if operation is RecognitionOperation.OBSERVE
        else None,
        deadline,
        "event_end" if operation is RecognitionOperation.END_TRACK else "",
    )


def wait_outcomes(executor: AsyncRecognitionExecutor, count: int):
    deadline = time.monotonic() + 2
    outcomes = []
    while len(outcomes) < count and time.monotonic() < deadline:
        outcomes.extend(executor.drain_outcomes())
        time.sleep(0.005)
    assert len(outcomes) == count
    return outcomes


def test_ordered_observations_and_end_match_synchronous_core():
    key = TrackKey("front", "stream", "track")
    executor = AsyncRecognitionExecutor(core_factory(), "service")
    assert executor.submit_nowait(job(key, 0)).accepted
    assert executor.submit_nowait(job(key, 1)).accepted
    assert executor.submit_nowait(
        job(key, 2, operation=RecognitionOperation.END_TRACK)
    ).accepted

    outcomes = wait_outcomes(executor, 3)
    assert [outcome.sequence for outcome in outcomes] == [0, 1, 2]
    assert [outcome.status for outcome in outcomes] == [
        RecognitionOutcomeStatus.SUCCEEDED,
        RecognitionOutcomeStatus.SUCCEEDED,
        RecognitionOutcomeStatus.ENDED,
    ]
    assert outcomes[1].updates[0].raw_value == "P1"
    assert executor.shutdown()
    assert executor.stats["sessions"] == 0


def test_queue_full_rejects_without_blocking_and_preserves_control_capacity():
    key = TrackKey("front", "stream", "track")
    gate = threading.Event()
    executor = AsyncRecognitionExecutor(
        core_factory(Model(gate)),
        "service",
        observation_capacity=1,
        control_capacity=1,
    )
    assert executor.submit_nowait(job(key, 0)).accepted
    deadline = time.monotonic() + 1
    while executor.stats["queue_depth"] and time.monotonic() < deadline:
        time.sleep(0.005)
    assert executor.submit_nowait(job(key, 1)).accepted
    started = time.monotonic()
    receipt = executor.submit_nowait(job(key, 2))
    assert time.monotonic() - started < 0.1
    assert not receipt.accepted
    assert receipt.reason == "queue_full"
    assert receipt.retryable
    assert executor.submit_nowait(
        job(key, 3, operation=RecognitionOperation.END_TRACK)
    ).accepted
    gate.set()
    wait_outcomes(executor, 3)
    assert executor.stats["queue_full"] == 1
    assert executor.shutdown()


def test_deadline_after_model_call_discards_state_update():
    key = TrackKey("front", "stream", "track")
    gate = threading.Event()
    executor = AsyncRecognitionExecutor(core_factory(Model(gate)), "service")
    assert executor.submit_nowait(
        job(key, 0, deadline=time.monotonic() + 0.02)
    ).accepted
    time.sleep(0.03)
    gate.set()
    first = wait_outcomes(executor, 1)[0]
    assert first.status is RecognitionOutcomeStatus.DEADLINE_EXCEEDED
    assert not first.updates

    assert executor.submit_nowait(job(key, 1)).accepted
    second = wait_outcomes(executor, 1)[0]
    assert second.status is RecognitionOutcomeStatus.SUCCEEDED
    assert second.updates[0].metadata["history_size"] == 1
    executor.submit_nowait(job(key, 2, operation=RecognitionOperation.END_TRACK))
    wait_outcomes(executor, 1)
    assert executor.shutdown()


def test_epoch_mismatch_and_duplicate_cancel_fail_closed():
    key = TrackKey("front", "stream", "track")
    executor = AsyncRecognitionExecutor(core_factory(), "service")
    wrong = RecognitionJob(
        "wrong",
        "client",
        "old-service",
        key,
        0,
        RecognitionOperation.OBSERVE,
        observation(key, 0),
    )
    receipt = executor.submit_nowait(wrong)
    assert not receipt.accepted
    assert receipt.reason == "epoch_mismatch"
    assert not receipt.retryable
    assert executor.cancel("missing")
    assert not executor.cancel("missing")
    assert executor.shutdown()


def test_observation_not_in_frame_returns_typed_skip_reason():
    key = TrackKey("front", "stream", "track")
    executor = AsyncRecognitionExecutor(core_factory(), "service")
    skipped = TrackedObservation(
        RecognitionTask.LPR,
        key,
        1.0,
        (0, 0, 100, 100),
        observed_in_frame=False,
        evidence_ref="frame",
    )
    submitted = RecognitionJob(
        "skipped",
        "client",
        "service",
        key,
        0,
        RecognitionOperation.OBSERVE,
        skipped,
    )
    assert executor.submit_nowait(submitted).accepted
    outcome = wait_outcomes(executor, 1)[0]
    assert outcome.status is RecognitionOutcomeStatus.SUCCEEDED
    assert outcome.reason == "observation_not_in_frame"
    assert not outcome.updates
    assert not outcome.artifacts
    assert executor.shutdown()
