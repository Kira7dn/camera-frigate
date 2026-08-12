"""Fail-closed asynchronous client for the recognition service."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import grpc
from grpc import aio

from ..contracts import (
    JobReceipt,
    RecognitionJob,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
)
from .evidence import MAX_EVIDENCE_BYTES
from .grpc_server import is_loopback_bind
from .v1 import recognition_pb2 as pb
from .v1 import recognition_pb2_grpc as pb_grpc
from .wire import envelope_from_job, outcome_from_proto, receipt_from_proto


@dataclass(frozen=True, slots=True)
class TlsClientConfig:
    root_ca: bytes
    certificate: bytes
    private_key: bytes
    server_name: str | None = None


class RecognitionGrpcClient:
    """Maintain one explicit gRPC topology without transport fallback."""

    def __init__(
        self,
        endpoint: str,
        client_id: str,
        *,
        tls: TlsClientConfig | None = None,
        rpc_deadline: float = 5.0,
        outcome_capacity: int = 128,
    ) -> None:
        if not endpoint or not client_id:
            raise ValueError("endpoint and client_id are required")
        if not is_loopback_bind(endpoint) and tls is None:
            raise ValueError("non-loopback recognition endpoint requires mTLS")
        self.endpoint = endpoint
        self.client_id = client_id
        self._tls = tls
        self._rpc_deadline = rpc_deadline
        self._outcomes: asyncio.Queue[RecognitionOutcome] = asyncio.Queue(
            maxsize=outcome_capacity
        )
        self._channel: aio.Channel | None = None
        self._stub: pb_grpc.RecognitionServiceStub | None = None
        self._stream = None
        self._receiver: asyncio.Task[None] | None = None
        self._hello = asyncio.Event()
        self._service_epoch = ""
        self._receipt_waiters: dict[str, asyncio.Future[JobReceipt]] = {}
        self._pending: dict[str, RecognitionJob] = {}
        self._write_lock = asyncio.Lock()
        self._healthy = False

    @property
    def service_epoch(self) -> str:
        return self._service_epoch

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def connect(self) -> str:
        if self._channel is not None:
            return self._service_epoch
        message_limit = MAX_EVIDENCE_BYTES * 2
        options = [
            ("grpc.max_receive_message_length", message_limit),
            ("grpc.max_send_message_length", message_limit),
        ]
        if self._tls is None:
            self._channel = aio.insecure_channel(
                self.endpoint, options=tuple(options)
            )
        else:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self._tls.root_ca,
                private_key=self._tls.private_key,
                certificate_chain=self._tls.certificate,
            )
            if self._tls.server_name:
                options.append(("grpc.ssl_target_name_override", self._tls.server_name))
            self._channel = aio.secure_channel(
                self.endpoint, credentials, options=tuple(options)
            )
        self._stub = pb_grpc.RecognitionServiceStub(self._channel)
        await asyncio.wait_for(self._channel.channel_ready(), self._rpc_deadline)
        self._stream = self._stub.Recognize()
        self._receiver = asyncio.create_task(self._receive())
        await asyncio.wait_for(self._hello.wait(), self._rpc_deadline)
        self._healthy = True
        return self._service_epoch

    async def configure(self, config_json: str) -> str:
        await self.connect()
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        response = await self._stub.Configure(  # type: ignore[union-attr]
            pb.ConfigureRequest(
                client_id=self.client_id,
                config_json=config_json,
                config_hash=config_hash,
            ),
            timeout=self._rpc_deadline,
        )
        if response.service_epoch != self._service_epoch:
            await self._fail_pending("service_epoch_changed")
            self._service_epoch = response.service_epoch
        return response.config_hash

    async def capabilities(self) -> pb.CapabilitiesResponse:
        await self.connect()
        return await self._stub.GetCapabilities(  # type: ignore[union-attr]
            pb.CapabilitiesRequest(), timeout=self._rpc_deadline
        )

    async def submit(self, job: RecognitionJob) -> JobReceipt:
        await self.connect()
        if job.service_epoch != self._service_epoch:
            return JobReceipt(
                job.job_id, self._service_epoch, False, "epoch_mismatch", False
            )
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[JobReceipt] = loop.create_future()
        self._receipt_waiters[job.job_id] = waiter
        self._pending[job.job_id] = job
        try:
            async with self._write_lock:
                await self._stream.write(envelope_from_job(job))
            receipt = await asyncio.wait_for(waiter, self._rpc_deadline)
        except (TimeoutError, grpc.RpcError):
            self._pending.pop(job.job_id, None)
            return JobReceipt(
                job.job_id, self._service_epoch, False, "unavailable", False
            )
        finally:
            self._receipt_waiters.pop(job.job_id, None)
        if not receipt.accepted:
            self._pending.pop(job.job_id, None)
        return receipt

    def get_outcome_nowait(self) -> RecognitionOutcome:
        return self._outcomes.get_nowait()

    async def get_outcome(self) -> RecognitionOutcome:
        return await self._outcomes.get()

    async def manage_face_library(
        self, request: pb.FaceLibraryRequest
    ) -> pb.FaceLibraryResponse:
        await self.connect()
        return await self._stub.ManageFaceLibrary(  # type: ignore[union-attr]
            request, timeout=self._rpc_deadline
        )

    async def close(self) -> None:
        if self._stream is not None:
            await self._stream.done_writing()
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        await self._fail_pending("client_closed")
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
        self._healthy = False

    async def _receive(self) -> None:
        try:
            async for envelope in self._stream:
                response = envelope.WhichOneof("response")
                if response == "hello":
                    epoch = envelope.hello.service_epoch
                    if self._service_epoch and self._service_epoch != epoch:
                        await self._fail_pending("service_epoch_changed")
                    self._service_epoch = epoch
                    self._hello.set()
                elif response == "receipt":
                    receipt = receipt_from_proto(envelope.receipt)
                    waiter = self._receipt_waiters.get(receipt.job_id)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(receipt)
                elif response == "outcome":
                    outcome = outcome_from_proto(envelope.outcome)
                    if outcome.service_epoch != self._service_epoch:
                        continue
                    self._pending.pop(outcome.job_id, None)
                    await self._outcomes.put(outcome)
        except grpc.RpcError:
            self._healthy = False
            await self._fail_pending("service_disconnected")
        finally:
            self._healthy = False

    async def _fail_pending(self, reason: str) -> None:
        for job in tuple(self._pending.values()):
            await self._outcomes.put(
                RecognitionOutcome(
                    job.job_id,
                    job.client_id,
                    self._service_epoch,
                    job.key,
                    job.sequence,
                    RecognitionOutcomeStatus.FAILED,
                    reason=reason,
                    retryable=False,
                )
            )
        self._pending.clear()
