"""Private gRPC transport and Frigate-main adapter for tracker updates."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

import grpc
from grpc import aio

from extension.topology.compiler import PlatformTopologyPlan
from extension.tracker.runtime import (
    TrackerJournal,
    TrackerOperation,
    TrackerUpdate,
    tracker_config_fingerprint,
)
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.infrastructure.comms.events_updater import EventUpdatePublisher
from frigate.infrastructure.config import FrigateConfig
from frigate.models import EdgeMediaManifest, EventObservation, TrackerJournalEntry

logger = logging.getLogger(__name__)
SERVICE = "camera.tracker.v1.TrackerService"


def _encode(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _decode(value: bytes) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("tracker message must be an object")
    return decoded


@dataclass(frozen=True, slots=True)
class TrackerTls:
    certificate: bytes
    private_key: bytes
    peer_ca: bytes


@dataclass(frozen=True, slots=True)
class ClientTls:
    ca: bytes
    certificate: bytes
    private_key: bytes
    server_name: str


class TrackerService:
    """Stream durable producer updates over a private mTLS endpoint."""

    def __init__(
        self,
        node_id: str,
        node_epoch: str,
        journal: TrackerJournal,
        config_hash: str,
        camera_health: Callable[[], tuple[dict[str, object], ...]],
        allowed_clients: frozenset[str] = frozenset(),
    ) -> None:
        self.node_id = node_id
        self.node_epoch = node_epoch
        self.journal = journal
        self.config_hash = config_hash
        self.camera_health = camera_health
        self.allowed_clients = allowed_clients
        self.degraded = False
        self._lock = threading.Lock()
        self._subscribers: set[
            tuple[asyncio.AbstractEventLoop, asyncio.Queue[TrackerUpdate]]
        ] = set()

    def publish(self, update: TrackerUpdate) -> TrackerUpdate:
        """Persist then notify connected main runtimes."""
        persisted = self.journal.append(update)
        with self._lock:
            subscribers = tuple(self._subscribers)
        for loop, updates in subscribers:
            loop.call_soon_threadsafe(self._offer, updates, persisted)
        return persisted

    @staticmethod
    def _offer(
        updates: asyncio.Queue[TrackerUpdate], update: TrackerUpdate
    ) -> None:
        try:
            updates.put_nowait(update)
        except asyncio.QueueFull:
            # The durable journal is replayed by the next health request.
            pass

    async def _authorize(self, context: aio.ServicerContext) -> None:
        if not self.allowed_clients:
            return
        common_names = context.auth_context().get("x509_common_name", ())
        identities = {
            value.decode(errors="replace") if isinstance(value, bytes) else str(value)
            for value in common_names
        }
        if identities.isdisjoint(self.allowed_clients):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "client_not_allowed")

    async def capabilities(
        self, request: bytes, context: aio.ServicerContext
    ) -> bytes:
        await self._authorize(context)
        cameras = self.camera_health()
        return _encode(
            {
                "schema_version": 1,
                "node_id": self.node_id,
                "node_epoch": self.node_epoch,
                "config_hash": self.config_hash,
                "health": {
                    "ready": True,
                    "degraded": self.degraded,
                    "pending_ack": self.journal.pending_count,
                },
                "cameras": list(cameras),
            }
        )

    async def connect(
        self, requests: AsyncIterator[bytes], context: aio.ServicerContext
    ) -> AsyncIterator[bytes]:
        await self._authorize(context)
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[TrackerUpdate] = asyncio.Queue(maxsize=256)
        subscriber = (loop, updates)
        with self._lock:
            self._subscribers.add(subscriber)
        yield _encode(
            {
                "type": "hello",
                "node_id": self.node_id,
                "node_epoch": self.node_epoch,
                "config_hash": self.config_hash,
            }
        )
        request_task = asyncio.ensure_future(anext(requests))
        update_task = asyncio.create_task(updates.get())
        last_sent = 0
        try:
            while True:
                done, _ = await asyncio.wait(
                    (request_task, update_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if request_task in done:
                    try:
                        request = _decode(request_task.result())
                    except StopAsyncIteration:
                        return
                    request_task = asyncio.ensure_future(anext(requests))
                    kind = request.get("type")
                    if kind in {"hello", "health"}:
                        after = (
                            int(request.get("replay_after_sequence", 0))
                            if kind == "hello"
                            else last_sent
                        )
                        for update in self.journal.replay(after):
                            yield _encode(
                                {"type": "update", "update": update.to_json()}
                            )
                            last_sent = update.journal_sequence
                    elif kind == "ack":
                        self.journal.acknowledge(
                            int(request["journal_sequence"]),
                            str(request["event_id"]),
                            str(request["node_epoch"]),
                        )
                if update_task in done:
                    update = update_task.result()
                    update_task = asyncio.create_task(updates.get())
                    if update.journal_sequence > last_sent:
                        yield _encode(
                            {"type": "update", "update": update.to_json()}
                        )
                        last_sent = update.journal_sequence
        finally:
            request_task.cancel()
            update_task.cancel()
            await asyncio.gather(request_task, update_task, return_exceptions=True)
            with self._lock:
                self._subscribers.discard(subscriber)

    def handler(self) -> grpc.GenericRpcHandler:
        return grpc.method_handlers_generic_handler(
            SERVICE,
            {
                "GetCapabilities": grpc.unary_unary_rpc_method_handler(
                    self.capabilities,
                    request_deserializer=lambda value: value,
                    response_serializer=lambda value: value,
                ),
                "Connect": grpc.stream_stream_rpc_method_handler(
                    self.connect,
                    request_deserializer=lambda value: value,
                    response_serializer=lambda value: value,
                ),
            },
        )


async def start_server(
    bind: str, service: TrackerService, tls: TrackerTls | None = None
) -> aio.Server:
    """Start the private tracker endpoint."""
    server = aio.server()
    server.add_generic_rpc_handlers((service.handler(),))
    if tls is None:
        server.add_insecure_port(bind)
    else:
        credentials = grpc.ssl_server_credentials(
            ((tls.private_key, tls.certificate),),
            root_certificates=tls.peer_ca,
            require_client_auth=True,
        )
        server.add_secure_port(bind, credentials)
    await server.start()
    return server


class TrackerIngestError(RuntimeError):
    pass


class TrackerHostIngest:
    """Validate producer ordering and lifecycle before canonical commit."""

    def __init__(
        self,
        camera_owners: Mapping[str, str],
        commit: Callable[[TrackerUpdate], None],
    ) -> None:
        self.camera_owners = dict(camera_owners)
        self.commit = commit
        self.last_sequences: dict[str, int] = {}
        self.active: dict[tuple[str, str, str, str], str] = {}
        self.accepted: dict[tuple[str, str, int], str] = {}

    def seed(self, node_id: str, sequence: int) -> None:
        self.last_sequences[node_id] = max(
            self.last_sequences.get(node_id, 0), sequence
        )

    def accept(self, update: TrackerUpdate) -> None:
        if self.camera_owners.get(update.camera_id) != update.node_id:
            raise TrackerIngestError("camera_owner_mismatch")
        accepted_key = (
            update.node_id,
            update.node_epoch,
            update.journal_sequence,
        )
        previous = self.accepted.get(accepted_key)
        if previous is not None:
            if previous != update.event_id:
                raise TrackerIngestError("sequence_event_conflict")
            return
        expected = self.last_sequences.get(update.node_id, 0) + 1
        if update.journal_sequence != expected:
            raise TrackerIngestError("journal_sequence_gap")
        track_key = (
            update.node_id,
            update.camera_id,
            update.stream_epoch,
            update.track_id,
        )
        active_event = self.active.get(track_key)
        if update.operation is TrackerOperation.START:
            if active_event is not None:
                raise TrackerIngestError("duplicate_track_start")
        elif active_event != update.event_id:
            raise TrackerIngestError("update_without_active_start")
        self.commit(update)
        if update.operation is TrackerOperation.START:
            self.active[track_key] = update.event_id
        elif update.operation is TrackerOperation.END:
            self.active.pop(track_key, None)
        self.last_sequences[update.node_id] = update.journal_sequence
        self.accepted[accepted_key] = update.event_id


class TrackerCanonicalStore:
    """Persist accepted producer identity after EventProcessor commit."""

    def __init__(self, database: Any) -> None:
        self.database = database

    def accept(self, update: TrackerUpdate) -> None:
        payload = json.loads(update.to_json())
        now = datetime.datetime.now(datetime.UTC)
        with self.database.atomic():
            existing = TrackerJournalEntry.get_or_none(
                (TrackerJournalEntry.node_id == update.node_id)
                & (TrackerJournalEntry.node_epoch == update.node_epoch)
                & (TrackerJournalEntry.journal_sequence == update.journal_sequence)
            )
            if existing is not None:
                if existing.event_id != update.event_id:
                    raise TrackerIngestError("durable_sequence_conflict")
                return
            TrackerJournalEntry.create(
                node_id=update.node_id,
                node_epoch=update.node_epoch,
                journal_sequence=update.journal_sequence,
                camera_id=update.camera_id,
                stream_epoch=update.stream_epoch,
                event_id=update.event_id,
                operation=update.operation.value,
                payload=payload,
                accepted_at=now,
            )
            key = f"{update.node_id}:{update.node_epoch}:{update.journal_sequence}"
            EventObservation.create(
                observation_id=hashlib.sha256(key.encode()).hexdigest(),
                event_id=update.event_id,
                kind={
                    TrackerOperation.START: "tracker_start",
                    TrackerOperation.UPDATE: "tracker_update",
                    TrackerOperation.END: "event_ended",
                }[update.operation],
                observed_at=now,
                frame_time=update.frame_time,
                evidence_id=None,
                payload=payload,
                expires_at=now + datetime.timedelta(days=2),
            )

    def last_sequence(self, node_id: str) -> int:
        row = (
            TrackerJournalEntry.select(TrackerJournalEntry.journal_sequence)
            .where(TrackerJournalEntry.node_id == node_id)
            .order_by(TrackerJournalEntry.journal_sequence.desc())
            .first()
        )
        return 0 if row is None else int(row.journal_sequence)


class TrackerMaintainer(threading.Thread):
    """Connect Frigate main to every tracker node in the compiled topology."""

    def __init__(
        self,
        config: FrigateConfig,
        topology: PlatformTopologyPlan,
        database: Any,
        event_update_queue: Queue,
        event_commit_queue: Queue,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(name="tracker_maintainer")
        self.config = config
        self.topology = topology
        self.store = TrackerCanonicalStore(database)
        self.event_update_queue = event_update_queue
        self.event_commit_queue = event_commit_queue
        self.stop_event = stop_event
        self.publisher = EventUpdatePublisher()
        self.shutdown = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.channels: dict[str, aio.Channel] = {}
        self.ingests = {
            node_id: TrackerHostIngest(topology.camera_owners, self._commit)
            for node_id in topology.tracker_nodes
        }

    @staticmethod
    def _tls(node: Any) -> ClientTls:
        return ClientTls(
            Path(node.tls.ca).read_bytes(),
            Path(node.tls.certificate).read_bytes(),
            Path(node.tls.key).read_bytes(),
            node.tls.server_name,
        )

    @staticmethod
    def _event_state(update: TrackerUpdate) -> EventStateEnum:
        return {
            TrackerOperation.START: EventStateEnum.start,
            TrackerOperation.UPDATE: EventStateEnum.update,
            TrackerOperation.END: EventStateEnum.end,
        }[update.operation]

    def _event_data(self, update: TrackerUpdate) -> dict[str, Any]:
        data = dict(update.state)
        data.update(
            {
                "id": update.event_id,
                "camera": update.camera_id,
                "frame_time": update.frame_time,
                "label": update.label,
                "score": update.score,
                "box": [
                    update.bbox.left,
                    update.bbox.top,
                    update.bbox.right,
                    update.bbox.bottom,
                ],
                "raw_track_id": update.track_id,
                "tracker_node_id": update.node_id,
                "tracker_node_epoch": update.node_epoch,
                "tracker_stream_epoch": update.stream_epoch,
                "tracker_journal_sequence": update.journal_sequence,
                "observed_in_frame": False,
            }
        )
        data.setdefault("start_time", update.frame_time)
        data.setdefault("end_time", None)
        if update.operation is TrackerOperation.END:
            data["end_time"] = update.frame_time
        return data

    def _commit(self, update: TrackerUpdate) -> None:
        node = self.config.tracker[update.node_id]
        receipt = uuid.uuid4().hex
        canonical = (
            EventTypeEnum.tracked_object,
            self._event_state(update),
            update.camera_id,
            "",
            self._event_data(update),
        )
        self.event_update_queue.put((*canonical, receipt), timeout=node.deadline)
        self.publisher.publish(canonical)
        deadline = time.monotonic() + node.deadline
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("tracker_event_commit_timeout")
            try:
                received, success, detail = self.event_commit_queue.get(
                    timeout=remaining
                )
            except queue.Empty as error:
                raise TimeoutError("tracker_event_commit_timeout") from error
            if received != receipt:
                raise TrackerIngestError("tracker_event_receipt_order_mismatch")
            if not success:
                raise TrackerIngestError(f"tracker_event_commit_failed:{detail}")
            break
        self.store.accept(update)

    async def _run_node(self, node_id: str) -> None:
        node = self.config.tracker[node_id]
        ingest = self.ingests[node_id]
        while not self.stop_event.is_set() and not self.shutdown.is_set():
            channel: aio.Channel | None = None
            heartbeat: asyncio.Task[None] | None = None
            try:
                tls = await asyncio.to_thread(self._tls, node)
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=tls.ca,
                    private_key=tls.private_key,
                    certificate_chain=tls.certificate,
                )
                channel = aio.secure_channel(
                    node.endpoint,
                    credentials,
                    options=(("grpc.ssl_target_name_override", tls.server_name),),
                )
                self.channels[node_id] = channel
                await asyncio.wait_for(channel.channel_ready(), node.deadline)
                connect = channel.stream_stream(
                    f"/{SERVICE}/Connect",
                    request_serializer=lambda value: value,
                    response_deserializer=lambda value: value,
                )
                call = connect()
                hello = _decode(await asyncio.wait_for(call.read(), node.deadline))
                if hello.get("type") != "hello" or hello.get("node_id") != node_id:
                    raise TrackerIngestError("tracker_hello_mismatch")
                if hello.get("config_hash") != tracker_config_fingerprint(
                    self.config, node_id
                ):
                    raise TrackerIngestError("tracker_config_hash_mismatch")
                durable_sequence = self.store.last_sequence(node_id)
                ingest.seed(node_id, durable_sequence)
                await call.write(
                    _encode(
                        {
                            "type": "hello",
                            "replay_after_sequence": durable_sequence,
                        }
                    )
                )

                async def send_heartbeat() -> None:
                    while True:
                        await asyncio.sleep(1)
                        await call.write(_encode({"type": "health"}))

                heartbeat = asyncio.create_task(send_heartbeat())
                async for raw in call:
                    message = _decode(raw)
                    if message.get("type") != "update":
                        continue
                    update = TrackerUpdate.from_json(str(message["update"]))
                    await asyncio.to_thread(ingest.accept, update)
                    await call.write(
                        _encode(
                            {
                                "type": "ack",
                                "node_epoch": update.node_epoch,
                                "journal_sequence": update.journal_sequence,
                                "event_id": update.event_id,
                            }
                        )
                    )
            except (grpc.RpcError, OSError, RuntimeError, ValueError):
                if not self.stop_event.is_set() and not self.shutdown.is_set():
                    logger.exception("Tracker node %s disconnected", node_id)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                self.channels.pop(node_id, None)
                if channel is not None:
                    await channel.close()
            if not self.stop_event.is_set() and not self.shutdown.is_set():
                await asyncio.sleep(min(1.0, node.deadline))

    def run(self) -> None:
        if not self.topology.tracker_nodes:
            return

        async def run_all() -> None:
            self.loop = asyncio.get_running_loop()
            await asyncio.gather(
                *(self._run_node(node_id) for node_id in self.topology.tracker_nodes)
            )

        asyncio.run(run_all())

    def request_stop(self) -> None:
        self.shutdown.set()

    def control_camera(
        self, camera_id: str, operation: str, payload: dict[str, object]
    ) -> bool:
        # Control is intentionally unavailable until its integration path is tested.
        return False

    async def fetch_media(
        self,
        node_id: str,
        media_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        raise RuntimeError("tracker_media_unavailable")

    def close(self) -> None:
        self.publisher.stop()


@dataclass(frozen=True, slots=True)
class EdgeMediaRange:
    manifest: EdgeMediaManifest
    offset: int
    length: int | None


def resolve_event_media(
    event_id: str, media_type: str, range_header: str | None = None
) -> EdgeMediaRange | None:
    manifest = (
        EdgeMediaManifest.select()
        .where(
            (EdgeMediaManifest.event_id == event_id)
            & (EdgeMediaManifest.media_type == media_type)
        )
        .order_by(EdgeMediaManifest.end_time.desc())
        .first()
    )
    if manifest is None:
        return None
    if not range_header:
        return EdgeMediaRange(manifest, 0, None)
    if not range_header.startswith("bytes=") or "," in range_header:
        raise ValueError("invalid_media_range")
    start_text, end_text = range_header[6:].split("-", 1)
    if not start_text:
        raise ValueError("suffix_ranges_not_supported")
    start = int(start_text)
    end = int(end_text) if end_text else int(manifest.byte_size) - 1
    if start < 0 or end < start or end >= int(manifest.byte_size):
        raise ValueError("invalid_media_range")
    return EdgeMediaRange(manifest, start, end - start + 1)


def resolve_media_id(media_id: str) -> EdgeMediaManifest | None:
    return EdgeMediaManifest.get_or_none(EdgeMediaManifest.media_id == media_id)
