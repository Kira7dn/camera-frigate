"""gRPC transport for the external recognition runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import grpc
from grpc import aio

from frigate.application.recognition.executor import AsyncRecognitionExecutor
from . import health, health_pb2, health_pb2_grpc
from .evidence import MAX_EVIDENCE_BYTES
from .v1 import recognition_pb2 as pb
from .v1 import recognition_pb2_grpc as pb_grpc
from .wire import job_from_envelope, outcome_to_proto, receipt_to_proto

FaceControl = Callable[[pb.FaceLibraryRequest], Awaitable[pb.FaceLibraryResponse]]


@dataclass(frozen=True, slots=True)
class TlsServerConfig:
    certificate: bytes
    private_key: bytes
    client_ca: bytes
    allowed_client_identities: frozenset[str]


class RecognitionGrpcService(pb_grpc.RecognitionServiceServicer):
    """Expose one bounded executor without taking Event or media ownership."""

    def __init__(
        self,
        executor: AsyncRecognitionExecutor,
        *,
        face_control: FaceControl | None = None,
        mtls_required: bool = True,
        allowed_client_identities: frozenset[str] = frozenset(),
        config_hash: str = "",
        dedupe_capacity: int = 4096,
        dedupe_ttl: float = 60.0,
    ) -> None:
        self.executor = executor
        self._face_control = face_control
        self._mtls_required = mtls_required
        self._allowed_client_identities = allowed_client_identities
        self._dedupe_capacity = dedupe_capacity
        self._dedupe_ttl = dedupe_ttl
        self._config_hash = config_hash
        self._dedupe: OrderedDict[
            tuple[str, str], tuple[float, pb.RecognitionOutcome]
        ] = OrderedDict()
        self._pending: dict[
            tuple[str, str], list[asyncio.Queue[pb.ServerEnvelope]]
        ] = {}
        self._dispatcher: asyncio.Task[None] | None = None

    async def Recognize(self, request_iterator, context):
        await self._authorize(context)
        self._ensure_dispatcher()
        responses: asyncio.Queue[pb.ServerEnvelope] = asyncio.Queue()
        stream_jobs: set[tuple[str, str]] = set()
        await responses.put(
            pb.ServerEnvelope(
                hello=pb.ServiceHello(
                    service_epoch=self.executor.service_epoch,
                    config_hash=self._config_hash,
                )
            )
        )

        async def receive() -> None:
            async for envelope in request_iterator:
                await self._accept(envelope, responses, stream_jobs)

        receiver = asyncio.create_task(receive())
        try:
            while not receiver.done() or stream_jobs or not responses.empty():
                try:
                    response = await asyncio.wait_for(responses.get(), 0.1)
                except TimeoutError:
                    if receiver.done() and not stream_jobs:
                        break
                    continue
                if response.HasField("outcome"):
                    stream_jobs.discard(
                        (response.outcome.client_id, response.outcome.job_id)
                    )
                yield response
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            for key in stream_jobs:
                subscribers = self._pending.get(key, [])
                self._pending[key] = [
                    item for item in subscribers if item is not responses
                ]
                if not self._pending[key]:
                    self._pending.pop(key, None)

    async def Configure(self, request, context):
        await self._authorize(context)
        try:
            parsed = json.loads(request.config_json)
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid config JSON")
        if not isinstance(parsed, dict):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "config must be an object"
            )
        actual_hash = hashlib.sha256(request.config_json.encode()).hexdigest()
        if request.config_hash != actual_hash:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "config hash mismatch"
            )
        if self._config_hash and self._config_hash != actual_hash:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "config change requires service restart "
                f"(service={self._config_hash[:12]} client={actual_hash[:12]})",
            )
        self._config_hash = actual_hash
        return pb.ConfigureResponse(
            service_epoch=self.executor.service_epoch,
            config_hash=actual_hash,
        )

    async def GetCapabilities(self, request, context):
        await self._authorize(context)
        return pb.CapabilitiesResponse(
            schema_version="1",
            service_epoch=self.executor.service_epoch,
            max_evidence_bytes=MAX_EVIDENCE_BYTES,
            tasks=[pb.RECOGNITION_TASK_FACE, pb.RECOGNITION_TASK_LPR],
            mtls_required=self._mtls_required,
        )

    async def ManageFaceLibrary(self, request, context):
        await self._authorize(context)
        if self._face_control is None:
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "face library control is unavailable"
            )
            raise RuntimeError("face library control is unavailable")
        return await self._face_control(request)

    async def _accept(
        self,
        envelope: pb.ClientEnvelope,
        responses: asyncio.Queue[pb.ServerEnvelope],
        stream_jobs: set[tuple[str, str]],
    ) -> None:
        self._expire_dedupe()
        try:
            job = job_from_envelope(envelope)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            request = envelope.WhichOneof("request")
            value = getattr(envelope, request) if request else None
            await responses.put(
                pb.ServerEnvelope(
                    receipt=pb.JobReceipt(
                        job_id=getattr(value, "job_id", ""),
                        service_epoch=self.executor.service_epoch,
                        accepted=False,
                        reason=str(error),
                        retryable=False,
                    )
                )
            )
            return
        key = (job.client_id, job.job_id)
        cached = self._dedupe.get(key)
        if cached is not None:
            await responses.put(
                pb.ServerEnvelope(
                    receipt=pb.JobReceipt(
                        job_id=job.job_id,
                        service_epoch=self.executor.service_epoch,
                        accepted=True,
                        reason="duplicate_completed",
                    )
                )
            )
            await responses.put(pb.ServerEnvelope(outcome=cached[1]))
            stream_jobs.add(key)
            return
        if key in self._pending:
            self._pending[key].append(responses)
            stream_jobs.add(key)
            await responses.put(
                pb.ServerEnvelope(
                    receipt=pb.JobReceipt(
                        job_id=job.job_id,
                        service_epoch=self.executor.service_epoch,
                        accepted=True,
                        reason="duplicate_pending",
                    )
                )
            )
            return
        self._pending[key] = [responses]
        receipt = self.executor.submit_nowait(job)
        await responses.put(pb.ServerEnvelope(receipt=receipt_to_proto(receipt)))
        if receipt.accepted:
            stream_jobs.add(key)
        else:
            self._pending.pop(key, None)

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch_outcomes())

    async def _dispatch_outcomes(self) -> None:
        while True:
            for outcome in self.executor.drain_outcomes():
                proto = outcome_to_proto(outcome)
                key = (outcome.client_id, outcome.job_id)
                self._dedupe[key] = (time.monotonic() + self._dedupe_ttl, proto)
                self._dedupe.move_to_end(key)
                while len(self._dedupe) > self._dedupe_capacity:
                    self._dedupe.popitem(last=False)
                for response_queue in self._pending.pop(key, []):
                    await response_queue.put(pb.ServerEnvelope(outcome=proto))
            await asyncio.sleep(0.005)

    def _expire_dedupe(self) -> None:
        now = time.monotonic()
        while self._dedupe:
            key, (expiry, _) = next(iter(self._dedupe.items()))
            if expiry > now:
                break
            self._dedupe.pop(key)

    async def close(self) -> None:
        """Stop outcome dispatch and drain the executor."""
        await asyncio.to_thread(self.executor.shutdown)
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)

    async def _authorize(self, context) -> None:
        if not self._allowed_client_identities:
            return
        auth_context = context.auth_context()
        identities = {
            value.decode(errors="replace")
            for key in ("x509_common_name", "x509_subject_alternative_name")
            for value in auth_context.get(key, ())
        }
        if not identities.intersection(self._allowed_client_identities):
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "client is not allowed"
            )


def is_loopback_bind(bind: str) -> bool:
    host = bind.rsplit(":", 1)[0].strip("[]")
    return host in {"127.0.0.1", "::1", "localhost"}


def create_grpc_server(
    service: RecognitionGrpcService,
    bind: str = "127.0.0.1:50051",
    *,
    tls: TlsServerConfig | None = None,
) -> tuple[aio.Server, health.HealthServicer]:
    """Create a server and fail closed for non-loopback plaintext binds."""
    if not is_loopback_bind(bind) and tls is None:
        raise ValueError("non-loopback recognition bind requires mTLS")
    if tls is not None and not tls.allowed_client_identities:
        raise ValueError("mTLS requires a client identity allowlist")
    server = aio.server(
        options=(
            ("grpc.max_receive_message_length", MAX_EVIDENCE_BYTES * 2),
            ("grpc.max_send_message_length", MAX_EVIDENCE_BYTES * 2),
        )
    )
    pb_grpc.add_RecognitionServiceServicer_to_server(service, server)
    health_service = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    health_service.set("", health_pb2.HealthCheckResponse.SERVING)
    health_service.set(
        "camera.recognition.v1.RecognitionService",
        health_pb2.HealthCheckResponse.SERVING,
    )
    if tls is None:
        server.add_insecure_port(bind)
    else:
        credentials = grpc.ssl_server_credentials(
            ((tls.private_key, tls.certificate),),
            root_certificates=tls.client_ca,
            require_client_auth=True,
        )
        server.add_secure_port(bind, credentials)
    return server, health_service
