"""Bounded ordered execution around the synchronous recognition core."""

from __future__ import annotations

import hashlib
import queue
import threading
import time
from collections.abc import Callable

from .contracts import (
    JobReceipt,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
    TrackKey,
)
from .core import RecognitionCore


class _Partition:
    def __init__(
        self,
        core: RecognitionCore,
        observation_capacity: int,
        control_capacity: int,
    ) -> None:
        self.core = core
        self.observation_capacity = observation_capacity
        self.queue: queue.Queue[RecognitionJob | None] = queue.Queue(
            maxsize=observation_capacity + control_capacity
        )
        self.observation_depth = 0


class AsyncRecognitionExecutor:
    """Execute recognition jobs without blocking producers or changing core policy."""

    def __init__(
        self,
        core_factory: Callable[[], RecognitionCore],
        service_epoch: str,
        *,
        partitions: int = 1,
        observation_capacity: int = 128,
        control_capacity: int = 64,
        outcome_capacity: int = 128,
        shutdown_drain: float = 10.0,
    ) -> None:
        if partitions <= 0:
            raise ValueError("partitions must be positive")
        if min(observation_capacity, control_capacity, outcome_capacity) <= 0:
            raise ValueError("queue capacities must be positive")
        self.service_epoch = service_epoch
        self._partitions = [
            _Partition(
                core_factory(),
                max(1, observation_capacity // partitions),
                max(1, control_capacity // partitions),
            )
            for _ in range(partitions)
        ]
        self._outcomes: queue.Queue[RecognitionOutcome] = queue.Queue(
            maxsize=outcome_capacity
        )
        self._shutdown_drain = shutdown_drain
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._started = False
        self._closing = False
        self._threads: list[threading.Thread] = []
        self._metrics = {
            "accepted": 0,
            "queue_full": 0,
            "cancelled": 0,
            "deadline_exceeded": 0,
            "failed": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            stats = dict(self._metrics)
            stats.update(
                {
                    "queue_depth": sum(item.queue.qsize() for item in self._partitions),
                    "outcome_depth": self._outcomes.qsize(),
                    "in_flight": sum(
                        item.core.stats["in_flight"] for item in self._partitions
                    ),
                    "sessions": sum(
                        item.core.stats["sessions"] for item in self._partitions
                    ),
                    "evidence_pinned": sum(
                        item.core.stats["evidence_pinned"] for item in self._partitions
                    ),
                }
            )
            return stats

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            if self._closing:
                raise RuntimeError("executor is closing")
            self._started = True
            for index, partition in enumerate(self._partitions):
                thread = threading.Thread(
                    target=self._run,
                    args=(partition,),
                    name=f"recognition-executor-{index}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def submit_nowait(self, job: RecognitionJob) -> JobReceipt:
        if job.service_epoch != self.service_epoch:
            return JobReceipt(
                job.job_id, self.service_epoch, False, "epoch_mismatch", False
            )
        if self._closing:
            return JobReceipt(job.job_id, self.service_epoch, False, "shutdown", True)
        self.start()
        partition = self._partition(job.key)
        with self._lock:
            if (
                job.operation is RecognitionOperation.OBSERVE
                and partition.observation_depth >= partition.observation_capacity
            ):
                self._metrics["queue_full"] += 1
                return JobReceipt(
                    job.job_id, self.service_epoch, False, "queue_full", True
                )
            try:
                partition.queue.put_nowait(job)
            except queue.Full:
                self._metrics["queue_full"] += 1
                return JobReceipt(
                    job.job_id, self.service_epoch, False, "queue_full", True
                )
            if job.operation is RecognitionOperation.OBSERVE:
                partition.observation_depth += 1
            elif job.operation is RecognitionOperation.CANCEL:
                self._cancelled.add(job.target_job_id or "")
            self._metrics["accepted"] += 1
        return JobReceipt(job.job_id, self.service_epoch, True)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancelled:
                return False
            self._cancelled.add(job_id)
            return True

    def get_outcome_nowait(self) -> RecognitionOutcome:
        outcome = self._outcomes.get_nowait()
        self._outcomes.task_done()
        return outcome

    def drain_outcomes(self) -> tuple[RecognitionOutcome, ...]:
        outcomes = []
        while True:
            try:
                outcomes.append(self.get_outcome_nowait())
            except queue.Empty:
                return tuple(outcomes)

    def drain(self, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if all(
                partition.queue.unfinished_tasks == 0 for partition in self._partitions
            ):
                return True
            time.sleep(0.005)
        return all(
            partition.queue.unfinished_tasks == 0 for partition in self._partitions
        )

    def shutdown(self, deadline: float | None = None) -> bool:
        with self._lock:
            if self._closing:
                return not any(thread.is_alive() for thread in self._threads)
            self._closing = True
        end = (
            deadline
            if deadline is not None
            else time.monotonic() + self._shutdown_drain
        )
        drained = self.drain(end)
        for partition in self._partitions:
            remaining = max(0.0, end - time.monotonic())
            try:
                partition.queue.put(None, timeout=remaining)
            except queue.Full:
                drained = False
        for thread in self._threads:
            thread.join(max(0.0, end - time.monotonic()))
        for index, partition in enumerate(self._partitions):
            if index >= len(self._threads) or not self._threads[index].is_alive():
                partition.core.shutdown()
        return drained and not any(thread.is_alive() for thread in self._threads)

    def _partition(self, key: TrackKey) -> _Partition:
        identity = f"{key.camera_id}\0{key.stream_epoch}\0{key.track_id}".encode()
        index = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        return self._partitions[index % len(self._partitions)]

    def _run(self, partition: _Partition) -> None:
        while True:
            job = partition.queue.get()
            try:
                if job is None:
                    return
                if job.operation is RecognitionOperation.OBSERVE:
                    with self._lock:
                        partition.observation_depth -= 1
                self._execute(partition.core, job)
            finally:
                partition.queue.task_done()

    def _execute(self, core: RecognitionCore, job: RecognitionJob) -> None:
        if job.operation is RecognitionOperation.CANCEL:
            self.cancel(job.target_job_id or "")
            self._emit(job, RecognitionOutcomeStatus.CANCELLED, reason="cancelled")
            return
        if job.operation is RecognitionOperation.END_TRACK:
            core.end_track(job.key, job.reason or "end_track")
            self._emit(job, RecognitionOutcomeStatus.ENDED, reason=job.reason)
            return
        if self._is_cancelled(job.job_id):
            self._record_and_emit(job, RecognitionOutcomeStatus.CANCELLED, "cancelled")
            return
        if self._expired(job):
            self._record_and_emit(
                job, RecognitionOutcomeStatus.DEADLINE_EXCEEDED, "deadline_exceeded"
            )
            return
        try:
            updates, artifacts, execution_reason = core.observe_guarded_with_artifacts(
                job.observation,  # type: ignore[arg-type]
                lambda: not self._is_cancelled(job.job_id) and not self._expired(job),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            with self._lock:
                self._metrics["failed"] += 1
            self._emit(
                job, RecognitionOutcomeStatus.FAILED, reason=type(error).__name__
            )
            return
        if self._is_cancelled(job.job_id):
            self._record_and_emit(job, RecognitionOutcomeStatus.CANCELLED, "cancelled")
        elif self._expired(job):
            self._record_and_emit(
                job, RecognitionOutcomeStatus.DEADLINE_EXCEEDED, "deadline_exceeded"
            )
        else:
            self._emit(
                job,
                RecognitionOutcomeStatus.SUCCEEDED,
                updates=updates,
                artifacts=artifacts,
                reason=execution_reason,
            )

    def _record_and_emit(
        self,
        job: RecognitionJob,
        status: RecognitionOutcomeStatus,
        reason: str,
    ) -> None:
        with self._lock:
            self._metrics[status.value] += 1
        self._emit(job, status, reason=reason)

    def _emit(
        self,
        job: RecognitionJob,
        status: RecognitionOutcomeStatus,
        *,
        updates=(),
        artifacts=(),
        reason: str = "",
    ) -> None:
        outcome = RecognitionOutcome(
            job.job_id,
            job.client_id,
            self.service_epoch,
            job.key,
            job.sequence,
            status,
            tuple(updates),
            reason,
            False,
            tuple(artifacts),
        )
        while True:
            try:
                self._outcomes.put(outcome, timeout=0.05)
                return
            except queue.Full:
                continue

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    @staticmethod
    def _expired(job: RecognitionJob) -> bool:
        return (
            job.deadline_monotonic is not None
            and time.monotonic() >= job.deadline_monotonic
        )
