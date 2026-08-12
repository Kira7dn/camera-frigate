"""mTLS-only camera.tracker.v1 service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import grpc
from grpc import aio

from ..evidence import EvidenceRing, EvidenceUnavailableError
from ..journal import EdgeJournal
from ..media import MediaAuthority
from .v1 import tracker_pb2 as pb
from .v1 import tracker_pb2_grpc as pb_grpc
from .wire import update_to_proto

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ServerTlsConfig:
    certificate: bytes
    private_key: bytes
    client_ca: bytes
    allowed_client_identities: frozenset[str]


class TrackerGrpcService(pb_grpc.TrackerServiceServicer):
    def __init__(
        self,
        *,
        node_id: str,
        node_epoch: str,
        journal: EdgeJournal,
        evidence: dict[str, EvidenceRing],
        media: MediaAuthority,
        detector_runtimes: tuple[str, ...] = (),
        config_hash: str = "",
        control: Callable[[str, int, dict[str, object]], tuple[bool, str, dict[str, object] | None]] | None = None,
        allowed_client_identities: frozenset[str] = frozenset(),
        camera_health: Callable[[], tuple[dict[str, object], ...]] | None = None,
    ) -> None:
        self.node_id = node_id
        self.node_epoch = node_epoch
        self.journal = journal
        self.evidence = evidence
        self.media = media
        self.detector_runtimes = detector_runtimes
        self.config_hash = config_hash
        self.control = control
        self.allowed_client_identities = allowed_client_identities
        self.camera_health = camera_health
        self.degraded = False

    async def Connect(self, request_iterator, context):
        await self._authorize(context)
        last_sent = 0
        yield pb.EdgeEnvelope(
            hello=pb.EdgeHello(
                node_id=self.node_id,
                node_epoch=self.node_epoch,
                schema_version=SCHEMA_VERSION,
                config_hash=self.config_hash,
            )
        )
        async for envelope in request_iterator:
            request = envelope.WhichOneof("request")
            if request == "hello":
                if envelope.hello.schema_version != SCHEMA_VERSION:
                    yield pb.EdgeEnvelope(
                        failure=pb.LifecycleFailure(
                            code="schema_mismatch", retryable=False
                        )
                    )
                    continue
                for update in self.journal.replay(
                    envelope.hello.replay_after_sequence
                ):
                    yield pb.EdgeEnvelope(update=update_to_proto(update))
                    last_sent = update.journal_sequence
            elif request == "ack":
                ack = envelope.ack
                if ack.node_id != self.node_id or ack.node_epoch != self.node_epoch:
                    yield pb.EdgeEnvelope(
                        failure=pb.LifecycleFailure(
                            code="ack_epoch_mismatch", retryable=False
                        )
                    )
                elif not self.journal.ack(ack.journal_sequence, ack.event_id):
                    yield pb.EdgeEnvelope(
                        failure=pb.LifecycleFailure(
                            code="ack_unknown", retryable=False
                        )
                    )
            elif request == "health":
                for update in self.journal.replay(last_sent):
                    yield pb.EdgeEnvelope(update=update_to_proto(update))
                    last_sent = update.journal_sequence
                yield pb.EdgeEnvelope(
                    health=pb.NodeHealth(
                        node_id=self.node_id,
                        node_epoch=self.node_epoch,
                        ready=True,
                        degraded=self.degraded,
                        pending_ack=self.journal.pending_count,
                        spool_bytes=self.journal.logical_bytes,
                        pinned_evidence=sum(
                            ring.pinned_count for ring in self.evidence.values()
                        ),
                    )
                )

    async def Configure(self, request, context):
        await self._authorize(context)
        digest = hashlib.sha256(request.config_json.encode()).hexdigest()
        if digest != request.config_hash:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "config_hash_mismatch")
        try:
            json.loads(request.config_json)
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_config_json")
        self.config_hash = digest
        return pb.ConfigureResponse(
            node_id=self.node_id,
            node_epoch=self.node_epoch,
            config_hash=digest,
        )

    async def GetCapabilities(self, request, context):
        await self._authorize(context)
        response = pb.CapabilitiesResponse(
            schema_version=SCHEMA_VERSION,
            node_id=self.node_id,
            node_epoch=self.node_epoch,
            detector_runtimes=self.detector_runtimes,
            max_evidence_bytes_per_camera=max(
                (ring.max_bytes for ring in self.evidence.values()), default=0
            ),
            max_media_chunk_bytes=1024 * 1024,
            mtls_required=True,
            ptz_supported=self.control is not None,
        )
        if self.camera_health is not None:
            response.cameras.extend(
                pb.CameraHealth(**camera) for camera in self.camera_health()
            )
        return response

    async def GetEvidence(self, request, context):
        await self._authorize(context)
        for ring in self.evidence.values():
            try:
                reference, data, shape = ring.get(request.evidence_id)
                return pb.EvidenceResponse(
                    evidence_id=reference.evidence_id,
                    data=data,
                    shape=shape,
                    layout="I420",
                    sha256=reference.sha256,
                    expiry_unix_ms=reference.expiry_unix_ms,
                )
            except EvidenceUnavailableError:
                continue
        try:
            reference, data, shape = self.journal.get_evidence(request.evidence_id)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "evidence_unavailable")
        return pb.EvidenceResponse(
            evidence_id=request.evidence_id,
            data=data,
            shape=shape,
            layout="I420",
            sha256=reference.sha256,
            expiry_unix_ms=reference.expiry_unix_ms,
        )

    async def StreamMedia(self, request, context):
        await self._authorize(context)
        try:
            manifest = self.media.manifest(request.media_id)
        except KeyError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "media_unavailable")
        length = request.length if request.HasField("length") else None
        if length is not None and request.offset + length > manifest.byte_size:
            await context.abort(grpc.StatusCode.OUT_OF_RANGE, "invalid_media_range")
        data = self.media.read_range(request.media_id, request.offset, length)
        chunk_size = 1024 * 1024
        for index in range(0, len(data), chunk_size):
            chunk = data[index : index + chunk_size]
            yield pb.MediaChunk(
                media_id=request.media_id,
                offset=request.offset + index,
                data=chunk,
                eof=index + len(chunk) == len(data),
            )

    async def ControlCamera(self, request, context):
        await self._authorize(context)
        if request.expected_node_epoch != self.node_epoch:
            return pb.ControlResponse(
                command_id=request.command_id,
                accepted=False,
                reason="node_epoch_mismatch",
                node_epoch=self.node_epoch,
            )
        if self.control is None:
            return pb.ControlResponse(
                command_id=request.command_id,
                accepted=False,
                reason="control_unavailable",
                node_epoch=self.node_epoch,
            )
        try:
            payload = json.loads(request.payload_json or "{}")
        except json.JSONDecodeError:
            return pb.ControlResponse(
                command_id=request.command_id,
                accepted=False,
                reason="invalid_payload_json",
                node_epoch=self.node_epoch,
            )
        accepted, reason, patch = self.control(
            request.camera_id, request.operation, payload
        )
        response = pb.ControlResponse(
            command_id=request.command_id,
            accepted=accepted,
            reason=reason,
            node_epoch=self.node_epoch,
        )
        if patch is not None:
            response.config_patch_json = json.dumps(patch, sort_keys=True)
        return response

    async def _authorize(self, context) -> None:
        auth_context = context.auth_context()
        identities = {
            value.decode(errors="replace")
            for key in ("x509_common_name", "x509_subject_alternative_name")
            for value in auth_context.get(key, ())
        }
        if not identities.intersection(self.allowed_client_identities):
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "tracker client is not allowed"
            )


async def start_secure_server(
    bind: str,
    service: TrackerGrpcService,
    tls: ServerTlsConfig,
) -> aio.Server:
    """Start a private server; plaintext is intentionally unsupported."""
    if not tls.allowed_client_identities:
        raise ValueError("tracker mTLS requires a client identity allowlist")
    if tls.allowed_client_identities != service.allowed_client_identities:
        raise ValueError("tracker service and TLS identity allowlists must match")
    server = aio.server()
    pb_grpc.add_TrackerServiceServicer_to_server(service, server)
    credentials = grpc.ssl_server_credentials(
        [(tls.private_key, tls.certificate)],
        root_certificates=tls.client_ca,
        require_client_auth=True,
    )
    if server.add_secure_port(bind, credentials) == 0:
        raise RuntimeError(f"unable to bind secure tracker endpoint {bind}")
    await server.start()
    return server
