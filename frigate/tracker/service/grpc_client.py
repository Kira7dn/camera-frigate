"""Frigate-side mTLS tracker client with ordered ACK after canonical ingest."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import grpc
from grpc import aio

from ..host import TrackerHostIngest
from .grpc_server import SCHEMA_VERSION
from .v1 import tracker_pb2 as pb
from .v1 import tracker_pb2_grpc as pb_grpc
from .wire import update_from_proto


@dataclass(frozen=True, slots=True)
class TlsClientConfig:
    root_ca: bytes
    certificate: bytes
    private_key: bytes
    server_name: str


class TrackerGrpcClient:
    def __init__(
        self,
        endpoint: str,
        client_id: str,
        tls: TlsClientConfig,
        ingest: TrackerHostIngest,
        *,
        deadline: float = 5,
        output_capacity: int = 256,
        durable_sequence: Callable[[str, str], int] | None = None,
    ) -> None:
        if not endpoint or not client_id:
            raise ValueError("tracker endpoint and client_id are required")
        self.endpoint = endpoint
        self.client_id = client_id
        self.tls = tls
        self.ingest = ingest
        self.deadline = deadline
        self._outgoing: asyncio.Queue[pb.MainEnvelope | None] = asyncio.Queue(
            maxsize=output_capacity
        )
        self._channel: aio.Channel | None = None
        self._stub: pb_grpc.TrackerServiceStub | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._server_hello = asyncio.Event()
        self._durable_sequence = durable_sequence
        self.node_id = ""
        self.node_epoch = ""
        self.config_hash = ""
        self.healthy = False

    async def connect(self, replay_after_sequence: int | None = None) -> tuple[str, str]:
        credentials = grpc.ssl_channel_credentials(
            root_certificates=self.tls.root_ca,
            private_key=self.tls.private_key,
            certificate_chain=self.tls.certificate,
        )
        options = (("grpc.ssl_target_name_override", self.tls.server_name),)
        self._channel = aio.secure_channel(
            self.endpoint, credentials, options=options
        )
        await asyncio.wait_for(self._channel.channel_ready(), self.deadline)
        self._stub = pb_grpc.TrackerServiceStub(self._channel)
        stream = self._stub.Connect(self._request_iterator())
        self._receiver = asyncio.create_task(self._receive(stream))
        await asyncio.wait_for(self._server_hello.wait(), self.deadline)
        if replay_after_sequence is None:
            replay_after_sequence = (
                self._durable_sequence(self.node_id, self.node_epoch)
                if self._durable_sequence is not None
                else 0
            )
        self.ingest.seed_sequence(
            self.node_id, self.node_epoch, replay_after_sequence
        )
        await self._outgoing.put(
            pb.MainEnvelope(
                hello=pb.ClientHello(
                    client_id=self.client_id,
                    schema_version=SCHEMA_VERSION,
                    replay_after_sequence=replay_after_sequence,
                )
            )
        )
        self._heartbeat = asyncio.create_task(self._send_heartbeat())
        return self.node_id, self.node_epoch

    async def _request_iterator(self):
        while True:
            envelope = await self._outgoing.get()
            if envelope is None:
                return
            yield envelope

    async def _receive(self, stream) -> None:
        async for envelope in stream:
            response = envelope.WhichOneof("response")
            if response == "hello":
                if envelope.hello.schema_version != SCHEMA_VERSION:
                    raise RuntimeError("tracker schema mismatch")
                self.node_id = envelope.hello.node_id
                self.node_epoch = envelope.hello.node_epoch
                self.config_hash = envelope.hello.config_hash
                self.healthy = True
                self._server_hello.set()
            elif response == "update":
                ack = self.ingest.accept(update_from_proto(envelope.update))
                await self._outgoing.put(
                    pb.MainEnvelope(
                        ack=pb.JournalAck(
                            node_id=ack.node_id,
                            node_epoch=ack.node_epoch,
                            journal_sequence=ack.journal_sequence,
                            event_id=ack.event_id,
                        )
                    )
                )
            elif response == "failure":
                raise RuntimeError(f"tracker failure: {envelope.failure.code}")

    async def _send_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(1)
            await self._outgoing.put(pb.MainEnvelope(health=pb.HealthRequest()))

    async def capabilities(self) -> pb.CapabilitiesResponse:
        if self._stub is None:
            await self.connect()
        return await self._stub.GetCapabilities(  # type: ignore[union-attr]
            pb.CapabilitiesRequest(), timeout=self.deadline
        )

    async def get_evidence(self, evidence_id: str) -> pb.EvidenceResponse:
        if self._stub is None:
            await self.connect()
        return await self._stub.GetEvidence(  # type: ignore[union-attr]
            pb.EvidenceRequest(evidence_id=evidence_id), timeout=self.deadline
        )

    async def stream_media(
        self, media_id: str, *, offset: int = 0, length: int | None = None
    ) -> bytes:
        if self._stub is None:
            await self.connect()
        request = pb.MediaRequest(media_id=media_id, offset=offset)
        if length is not None:
            request.length = length
        output = bytearray()
        async for chunk in self._stub.StreamMedia(  # type: ignore[union-attr]
            request, timeout=self.deadline
        ):
            output.extend(chunk.data)
        return bytes(output)

    async def close(self) -> None:
        self.healthy = False
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        await self._outgoing.put(None)
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        if self._channel is not None:
            await self._channel.close()
