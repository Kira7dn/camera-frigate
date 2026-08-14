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
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

import grpc
from grpc import aio
from playhouse.sqliteq import SqliteQueueDatabase

from extension.topology.compiler import PlatformTopologyPlan
from extension.tracker.runtime import (
    TrackerJournal,
    TrackerOperation,
    TrackerUpdate,
    tracker_config_fingerprint,
)
from frigate.application.events.canonical import EventAggregator
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.infrastructure.comms.events_updater import EventUpdatePublisher
from frigate.infrastructure.config import FrigateConfig
from frigate.models import EdgeMediaManifest, TrackerJournalEntry
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)
SERVICE = "camera.tracker.v1.TrackerService"
PROTOCOL_VERSION = 2


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
        media_reader: Callable[[str, int, int | None], bytes] | None = None,
        active_lifecycles: Callable[[], int] = lambda: 0,
    ) -> None:
        self.node_id = node_id
        self.node_epoch = node_epoch
        self.journal = journal
        self.config_hash = config_hash
        self.camera_health = camera_health
        self.allowed_clients = allowed_clients
        self.media_reader = media_reader
        self.active_lifecycles = active_lifecycles
        self.degraded = False

    def publish(self, update: TrackerUpdate) -> TrackerUpdate:
        """Persist an update; the connection sender drains the outbox."""
        return self.journal.append(update)

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
                "protocol_version": PROTOCOL_VERSION,
                "node_id": self.node_id,
                "node_epoch": self.node_epoch,
                "config_hash": self.config_hash,
                "health": {
                    "ready": True,
                    "degraded": self.degraded,
                    "pending_ack": self.journal.pending_count_for_epoch(self.node_epoch),
                    "pinned_evidence": 0,
                    "active_lifecycles": self.active_lifecycles(),
                },
                "cameras": list(cameras),
            }
        )

    async def connect(
        self, requests: AsyncIterator[bytes], context: aio.ServicerContext
    ) -> AsyncIterator[bytes]:
        await self._authorize(context)
        yield _encode(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "node_id": self.node_id,
                "node_epoch": self.node_epoch,
                "config_hash": self.config_hash,
            }
        )
        request_task = asyncio.ensure_future(anext(requests))
        try:
            start = _decode(await request_task)
            request_task = asyncio.ensure_future(anext(requests))
            if (
                start.get("type") != "session_start"
                or int(start.get("protocol_version", 0)) != PROTOCOL_VERSION
                or str(start.get("node_epoch")) != self.node_epoch
            ):
                raise TrackerIngestError("tracker_session_mismatch")
            acknowledged = int(start.get("ack_sequence", 0))
            self.journal.acknowledge_through(self.node_epoch, acknowledged)
            while True:
                update = self.journal.next_pending(self.node_epoch, acknowledged)
                if update is not None:
                    sequence = update.journal_sequence
                    yield _encode({"type": "update", "update": update.to_json()})
                    while acknowledged < sequence:
                        request = _decode(await request_task)
                        request_task = asyncio.ensure_future(anext(requests))
                        if request.get("type") == "ack":
                            ack_sequence = int(request.get("ack_sequence", -1))
                            if ack_sequence != sequence:
                                raise TrackerIngestError(
                                    "tracker_ack_sequence_mismatch"
                                )
                            self.journal.acknowledge_through(
                                self.node_epoch, ack_sequence
                            )
                            acknowledged = ack_sequence
                        elif request.get("type") != "health":
                            raise TrackerIngestError("tracker_ack_without_update")
                    continue
                done, _ = await asyncio.wait(
                    (request_task,), timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                if request_task in done:
                    try:
                        request = _decode(request_task.result())
                    except StopAsyncIteration:
                        return
                    request_task = asyncio.ensure_future(anext(requests))
                    kind = request.get("type")
                    if kind == "ack":
                        sequence = int(request.get("ack_sequence", -1))
                        if sequence <= acknowledged:
                            continue
                        raise TrackerIngestError("tracker_ack_without_update")
        finally:
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

    async def fetch_media(
        self, request: bytes, context: aio.ServicerContext
    ) -> AsyncIterator[bytes]:
        """Stream a bounded range of edge-owned media."""
        await self._authorize(context)
        if self.media_reader is None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "tracker_media_unavailable")
        value = _decode(request)
        try:
            data = await asyncio.to_thread(
                self.media_reader,
                str(value["media_id"]),
                int(value.get("offset", 0)),
                None if value.get("length") is None else int(value["length"]),
            )
        except FileNotFoundError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "tracker_media_not_found")
        except (KeyError, TypeError, ValueError):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_media_range")
        for offset in range(0, len(data), 64 * 1024):
            yield data[offset : offset + 64 * 1024]

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
                "FetchMedia": grpc.unary_stream_rpc_method_handler(
                    self.fetch_media,
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
        self.last_sequences: dict[tuple[str, str], int] = {}
        self.active: dict[tuple[str, str, str, str], str] = {}
        self.accepted: dict[tuple[str, str, int], str] = {}

    def start_epoch(self, node_id: str, node_epoch: str, sequence: int = 0) -> None:
        self.last_sequences[(node_id, node_epoch)] = sequence
        self.active = {
            key: value
            for key, value in self.active.items()
            if key[0] != node_id
        }

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
        expected = self.last_sequences.get(
            (update.node_id, update.node_epoch), 0
        ) + 1
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
        self.last_sequences[(update.node_id, update.node_epoch)] = (
            update.journal_sequence
        )
        self.accepted[accepted_key] = update.event_id


class TrackerCanonicalStore:
    """Persist accepted producer identity after EventProcessor commit."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self.aggregator = EventAggregator()

    def accept(self, update: TrackerUpdate) -> None:
        payload = json.loads(update.to_json())
        now = datetime.datetime.now(datetime.UTC)
        transaction = (
            nullcontext()
            if isinstance(self.database, SqliteQueueDatabase)
            else self.database.atomic()
        )
        with transaction:
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
            self.aggregator.observe(
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
            )
            for manifest in update.media:
                existing_media = EdgeMediaManifest.get_or_none(
                    EdgeMediaManifest.media_id == manifest.media_id
                )
                if existing_media is not None:
                    if (
                        existing_media.event_id != update.event_id
                        or existing_media.sha256 != manifest.sha256
                    ):
                        raise TrackerIngestError("durable_media_conflict")
                    continue
                EdgeMediaManifest.create(
                    media_id=manifest.media_id,
                    node_id=update.node_id,
                    camera_id=manifest.camera_id,
                    event_id=manifest.event_id,
                    media_type=manifest.media_type,
                    codec=manifest.codec,
                    start_time=manifest.start_time,
                    end_time=manifest.end_time,
                    byte_size=manifest.byte_size,
                    sha256=manifest.sha256,
                    expires_at=datetime.datetime.fromtimestamp(
                        manifest.expiry_unix_ms / 1000, datetime.UTC
                    ),
                )

    def last_sequence(self, node_id: str, node_epoch: str | None = None) -> int:
        predicate = TrackerJournalEntry.node_id == node_id
        if node_epoch is not None:
            predicate &= TrackerJournalEntry.node_epoch == node_epoch
        row = (
            TrackerJournalEntry.select(TrackerJournalEntry.journal_sequence)
            .where(predicate)
            .order_by(TrackerJournalEntry.journal_sequence.desc())
            .first()
        )
        return 0 if row is None else int(row.journal_sequence)

    def active_lifecycles(
        self, node_id: str, node_epoch: str | None = None
    ) -> dict[tuple[str, str, str, str], str]:
        """Rebuild active tracks after a Frigate-main restart."""
        active: dict[tuple[str, str, str, str], str] = {}
        predicate = TrackerJournalEntry.node_id == node_id
        if node_epoch is not None:
            predicate &= TrackerJournalEntry.node_epoch == node_epoch
        rows = (
            TrackerJournalEntry.select()
            .where(predicate)
            .order_by(TrackerJournalEntry.journal_sequence.asc())
        )
        for row in rows:
            payload = row.payload
            key = (
                row.node_id,
                row.camera_id,
                row.stream_epoch,
                str(payload["track_id"]),
            )
            if row.operation == TrackerOperation.START.value:
                active[key] = row.event_id
            elif row.operation == TrackerOperation.END.value:
                active.pop(key, None)
        return active


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
        self.frame_manager = SharedMemoryFrameManager()
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
        region = data.get("region")
        if not isinstance(region, list | tuple) or len(region) != 4:
            regions = update.region.get("boxes", ())
            region = next(
                (
                    candidate
                    for candidate in regions
                    if isinstance(candidate, list | tuple) and len(candidate) == 4
                ),
                [
                    update.bbox.left,
                    update.bbox.top,
                    update.bbox.right,
                    update.bbox.bottom,
                ],
            )
            data["region"] = list(region)
        data.setdefault("start_time", update.frame_time)
        data.setdefault("end_time", None)
        if update.operation is TrackerOperation.END:
            data["end_time"] = update.frame_time
        if any(item.media_type == "clip" for item in update.media):
            data["has_clip"] = True
        if any(item.media_type == "snapshot" for item in update.media):
            data["has_snapshot"] = True
        return data

    @staticmethod
    def _unavailable_frame_name(update: TrackerUpdate) -> str:
        """Return a valid missing SHM name for updates without transferred evidence."""
        return (
            f"tracker_unavailable_{update.node_id}_{update.journal_sequence}_"
            f"{update.event_id}"
        )

    def _commit(self, update: TrackerUpdate) -> None:
        node = self.config.tracker[update.node_id]
        receipt = uuid.uuid4().hex
        frame_name = self._unavailable_frame_name(update)
        created_frame = False
        event_data = self._event_data(update)
        evidence = next(
            (item for item in update.media if item.media_type == "recognition_frame"),
            None,
        )
        if evidence is not None:
            if self.loop is None:
                raise RuntimeError("tracker_media_unavailable")
            future = asyncio.run_coroutine_threadsafe(
                self.fetch_media(update.node_id, evidence.media_id), self.loop
            )
            content = future.result(timeout=node.deadline)
            if (
                len(content) != evidence.byte_size
                or hashlib.sha256(content).hexdigest() != evidence.sha256
            ):
                raise TrackerIngestError("tracker_media_integrity_mismatch")
            expected_size = int(
                self.config.cameras[update.camera_id].frame_shape_yuv[0]
                * self.config.cameras[update.camera_id].frame_shape_yuv[1]
            )
            if len(content) != expected_size:
                raise TrackerIngestError("tracker_recognition_frame_size_mismatch")
            frame_name = f"recognition_tracker_{uuid.uuid4().hex}"
            frame_buffer = self.frame_manager.create(frame_name, len(content))
            frame_buffer[:] = content
            del frame_buffer
            created_frame = True
            event_data["_recognition_evidence_owned"] = True
            event_data["observed_in_frame"] = True
        canonical = (
            EventTypeEnum.tracked_object,
            self._event_state(update),
            update.camera_id,
            frame_name,
            event_data,
        )
        try:
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
        except Exception:
            if created_frame:
                self.frame_manager.delete(frame_name)
            raise
        else:
            if created_frame:
                # Keep the creator handle alive until the consumer acknowledges the
                # event. Windows removes named shared memory when its last handle
                # closes, while POSIX keeps it until unlink.
                self.frame_manager.close(frame_name)
        self.store.accept(update)

    async def _run_node(self, node_id: str) -> None:
        node = self.config.tracker[node_id]
        ingest = self.ingests[node_id]
        connect_failure_logged = False
        while not self.stop_event.is_set() and not self.shutdown.is_set():
            channel: aio.Channel | None = None
            heartbeat: asyncio.Task[None] | None = None
            session_started = False
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
                if (
                    hello.get("type") != "hello"
                    or hello.get("node_id") != node_id
                    or int(hello.get("protocol_version", 0)) != PROTOCOL_VERSION
                ):
                    raise TrackerIngestError("tracker_hello_mismatch")
                if hello.get("config_hash") != tracker_config_fingerprint(
                    self.config, node_id
                ):
                    raise TrackerIngestError("tracker_config_hash_mismatch")
                node_epoch = str(hello["node_epoch"])
                durable_sequence = self.store.last_sequence(node_id, node_epoch)
                ingest.start_epoch(node_id, node_epoch, durable_sequence)
                ingest.active = self.store.active_lifecycles(node_id, node_epoch)
                await call.write(
                    _encode(
                        {
                            "type": "session_start",
                            "protocol_version": PROTOCOL_VERSION,
                            "node_epoch": node_epoch,
                            "ack_sequence": durable_sequence,
                        }
                    )
                )
                session_started = True
                connect_failure_logged = False
                logger.info("Tracker node %s connected", node_id)

                async def send_heartbeat() -> None:
                    while True:
                        await asyncio.sleep(1)
                        await call.write(_encode({"type": "health"}))

                heartbeat = asyncio.create_task(send_heartbeat())
                while True:
                    raw = await call.read()
                    if raw is aio.EOF:
                        break
                    message = _decode(raw)
                    if message.get("type") != "update":
                        continue
                    update = TrackerUpdate.from_json(str(message["update"]))
                    await asyncio.to_thread(ingest.accept, update)
                    await call.write(
                        _encode(
                            {
                                "type": "ack",
                                "ack_sequence": update.journal_sequence,
                            }
                        )
                    )
            except (grpc.RpcError, OSError, RuntimeError, ValueError) as error:
                if not self.stop_event.is_set() and not self.shutdown.is_set():
                    if session_started:
                        logger.warning(
                            "Tracker node %s connection interrupted; retrying: %s",
                            node_id,
                            error,
                        )
                    elif not connect_failure_logged:
                        logger.warning(
                            "Tracker node %s is not ready; retrying: %s",
                            node_id,
                            error,
                        )
                        connect_failure_logged = True
                    else:
                        logger.debug(
                            "Tracker node %s is still unavailable: %s", node_id, error
                        )
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
        async def fetch() -> bytes:
            channel = self.channels.get(node_id)
            if channel is None:
                raise RuntimeError("tracker_media_unavailable")
            method = channel.unary_stream(
                f"/{SERVICE}/FetchMedia",
                request_serializer=lambda value: value,
                response_deserializer=lambda value: value,
            )
            call = method(
                _encode({"media_id": media_id, "offset": offset, "length": length})
            )
            chunks = bytearray()
            async for chunk in call:
                chunks.extend(chunk)
            return bytes(chunks)

        if self.loop is None:
            raise RuntimeError("tracker_media_unavailable")
        if asyncio.get_running_loop() is self.loop:
            return await fetch()
        future = asyncio.run_coroutine_threadsafe(fetch(), self.loop)
        return await asyncio.wrap_future(future)

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
