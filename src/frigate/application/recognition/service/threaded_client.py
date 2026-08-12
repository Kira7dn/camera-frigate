"""Thread-owned bridge from Frigate's synchronous loop to the async gRPC client."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import TypeAlias

from ..contracts import (
    JobReceipt,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcome,
)
from .grpc_client import RecognitionGrpcClient, TlsClientConfig
from .v1 import recognition_pb2 as pb


@dataclass(frozen=True, slots=True)
class ClientResult:
    receipt: JobReceipt | None = None
    outcome: RecognitionOutcome | None = None


@dataclass(frozen=True, slots=True)
class _FaceCommand:
    request: pb.FaceLibraryRequest
    response: queue.Queue[pb.FaceLibraryResponse | BaseException]


_Command: TypeAlias = RecognitionJob | _FaceCommand | None


class ThreadedRecognitionClient:
    """Own an asyncio client without ever blocking the frame producer."""

    def __init__(
        self,
        endpoint: str,
        client_id: str,
        config_json: str,
        *,
        tls: TlsClientConfig,
        deadline: float,
        observation_capacity: int,
        control_capacity: int,
        outcome_capacity: int,
    ) -> None:
        self._client = RecognitionGrpcClient(
            endpoint,
            client_id,
            tls=tls,
            rpc_deadline=deadline,
            outcome_capacity=outcome_capacity,
        )
        self._config_json = config_json
        self._commands: queue.Queue[_Command] = queue.Queue(
            maxsize=observation_capacity + control_capacity
        )
        self._observation_capacity = observation_capacity
        self._observation_depth = 0
        self._results: queue.Queue[ClientResult] = queue.Queue(maxsize=outcome_capacity)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="recognition-grpc-client", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(deadline + 1):
            raise TimeoutError("recognition service startup timed out")
        if self._startup_error is not None:
            raise RuntimeError(
                "recognition service startup failed"
            ) from self._startup_error

    @property
    def service_epoch(self) -> str:
        return self._client.service_epoch

    @property
    def stats(self) -> dict[str, int]:
        return {
            "queue_depth": self._commands.qsize(),
            "outcome_depth": self._results.qsize(),
            "healthy": int(self._client.healthy),
        }

    def submit_nowait(self, job: RecognitionJob) -> JobReceipt:
        with self._lock:
            if self._closed:
                return JobReceipt(
                    job.job_id, self.service_epoch, False, "closed", False
                )
            if (
                job.operation is RecognitionOperation.OBSERVE
                and self._observation_depth >= self._observation_capacity
            ):
                return JobReceipt(
                    job.job_id, self.service_epoch, False, "queue_full", True
                )
            try:
                self._commands.put_nowait(job)
            except queue.Full:
                return JobReceipt(
                    job.job_id, self.service_epoch, False, "queue_full", True
                )
            if job.operation is RecognitionOperation.OBSERVE:
                self._observation_depth += 1
        return JobReceipt(job.job_id, self.service_epoch, True)

    def drain_results(self) -> tuple[ClientResult, ...]:
        values = []
        while True:
            try:
                values.append(self._results.get_nowait())
            except queue.Empty:
                return tuple(values)

    def manage_face_library(
        self, request: pb.FaceLibraryRequest, timeout: float
    ) -> pb.FaceLibraryResponse:
        response: queue.Queue[pb.FaceLibraryResponse | BaseException] = queue.Queue(1)
        try:
            self._commands.put_nowait(_FaceCommand(request, response))
        except queue.Full:
            return pb.FaceLibraryResponse(
                success=False, message="Recognition control queue is full"
            )
        try:
            result = response.get(timeout=timeout)
        except queue.Empty:
            return pb.FaceLibraryResponse(
                success=False, message="Recognition control request timed out"
            )
        if isinstance(result, BaseException):
            return pb.FaceLibraryResponse(
                success=False, message="Recognition service is unavailable"
            )
        return result

    def close(self, timeout: float) -> bool:
        with self._lock:
            if self._closed:
                return not self._thread.is_alive()
            self._closed = True
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            return False
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        try:
            await self._client.connect()
            await self._client.configure(self._config_json)
        except BaseException as error:
            self._startup_error = error
            self._ready.set()
            return
        self._ready.set()
        receiver = asyncio.create_task(self._receive_outcomes())
        try:
            while True:
                job = await asyncio.to_thread(self._commands.get)
                if job is None:
                    break
                if isinstance(job, _FaceCommand):
                    try:
                        response = await self._client.manage_face_library(job.request)
                    except BaseException as error:
                        response = error
                    await asyncio.to_thread(job.response.put, response)
                    continue
                if job.operation is RecognitionOperation.OBSERVE:
                    with self._lock:
                        self._observation_depth -= 1
                try:
                    receipt = await self._client.submit(job)
                except (asyncio.InvalidStateError, RuntimeError, OSError) as error:
                    # The bidirectional stream may finish between dequeue and
                    # write during a service restart. Convert that transport
                    # loss into the typed lifecycle result owned by Frigate;
                    # never let the client thread die with an unobserved job.
                    self._client._healthy = False
                    receipt = JobReceipt(
                        job.job_id,
                        self.service_epoch,
                        False,
                        "service_disconnected",
                        False,
                    )
                await asyncio.to_thread(
                    self._results.put, ClientResult(receipt=receipt)
                )
        finally:
            await self._client.close()
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def _receive_outcomes(self) -> None:
        while True:
            outcome = await self._client.get_outcome()
            await asyncio.to_thread(self._results.put, ClientResult(outcome=outcome))
