"""Private gRPC transport and Frigate-main adapter for tracker updates."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
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

import cv2
import grpc
from grpc import aio
from playhouse.sqliteq import SqliteQueueDatabase

from extension.topology.compiler import PlatformTopologyPlan
from extension.tracker.runtime import (
    MediaManifest,
    TrackerJournal,
    TrackerOperation,
    TrackerUpdate,
    tracker_config_fingerprint,
)
from frigate.application.events.canonical import (
    EventAggregator,
    RenderSpec,
    as_utc,
    normalized_xyxy,
)
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.infrastructure.comms.events_updater import EventUpdatePublisher
from frigate.infrastructure.config import FrigateConfig
from frigate.models import (
    EdgeMediaManifest,
    Event,
    EventEvidence,
    ReviewSegment,
    TrackerJournalEntry,
)
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)
SERVICE = "camera.tracker.v1.TrackerService"
PROTOCOL_VERSION = 2
PRODUCER_MEDIA_MAX_BYTES = 128 * 1024 * 1024


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


async def start_producer_server(
    bind: str, service: ProducerIngressService
) -> aio.Server:
    """Start the same private gRPC service for Safety producer ingress."""
    server = aio.server()
    server.add_generic_rpc_handlers((service.handler(),))
    server.add_insecure_port(bind)
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


class ProducerIngressService:
    """Frigate-main ingress for non-native producers.

    The existing private gRPC service is also the only producer ingress.  Safety
    uses these two methods; no HTTP event API or live-frame lookup is involved.
    Media is uploaded as binary gRPC chunks and the event message carries only
    its validated manifest.
    """

    def __init__(
        self,
        ingest: TrackerHostIngest,
        media_root: str | Path,
        health: Callable[[], tuple[dict[str, object], ...]],
    ) -> None:
        self.ingest = ingest
        self.media_root = Path(media_root)
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.health = health
        self.media_paths: dict[str, Path] = {}
        self.media_manifests: dict[str, MediaManifest] = {}
        self._lock = threading.Lock()

    async def capabilities(self, request: bytes, context: aio.ServicerContext) -> bytes:
        cameras = self.health()
        return _encode(
            {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "node_id": "frigate-main",
                "health": {"ready": True, "degraded": False},
                "cameras": list(cameras),
            }
        )

    async def upload_media(
        self, requests: AsyncIterator[bytes], context: aio.ServicerContext
    ) -> bytes:
        try:
            header = _decode(await anext(requests))
            media_id = str(header["media_id"])
            event_id = str(header["event_id"])
            camera_id = str(header["camera_id"])
            media_type = str(header["media_type"])
            codec = str(header["codec"])
            byte_size = int(header["byte_size"])
            sha256 = str(header["sha256"])
            start_time = float(header["start_time"])
            end_time = float(header["end_time"])
        except (StopAsyncIteration, KeyError, TypeError, ValueError):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_media_manifest")
        if not media_id or not event_id or not camera_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid_media_identity")
        if media_type not in {"snapshot_jpg", "clip"}:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unsupported_media_type")
        if codec not in {"jpeg", "h264", "mp4"}:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unsupported_media_codec")
        if byte_size <= 0 or byte_size > PRODUCER_MEDIA_MAX_BYTES:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "invalid_media_size")

        temporary = self.media_root / f".{media_id}.tmp"
        target = self.media_root / media_id
        digest = hashlib.sha256()
        received = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in requests:
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > byte_size:
                        raise ValueError("media_size_overflow")
                    digest.update(chunk)
                    handle.write(chunk)
            if received != byte_size or digest.hexdigest() != sha256:
                raise ValueError("media_integrity_mismatch")
            os.replace(temporary, target)
        except (OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

        manifest = MediaManifest(
            media_id=media_id,
            event_id=event_id,
            camera_id=camera_id,
            media_type=media_type,
            codec=codec,
            start_time=start_time,
            end_time=end_time,
            byte_size=byte_size,
            sha256=sha256,
            expiry_unix_ms=int((time.time() + 3600) * 1000),
        )
        with self._lock:
            previous = self.media_manifests.get(media_id)
            if previous is not None and (
                previous.event_id != event_id or previous.sha256 != sha256
            ):
                await context.abort(grpc.StatusCode.ALREADY_EXISTS, "media_id_conflict")
            self.media_paths[media_id] = target
            self.media_manifests[media_id] = manifest
        return _encode({"ok": True, "media_id": media_id, "sha256": sha256})

    async def publish(self, request: bytes, context: aio.ServicerContext) -> bytes:
        try:
            update = TrackerUpdate.from_json(_decode(request)["update"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid_update:{error}")
        if update.source_type != "safety":
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "producer_source_not_allowed")
        try:
            await asyncio.to_thread(self.ingest.accept, update)
        except TrackerIngestError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except (OSError, RuntimeError, ValueError) as error:
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
        return _encode({"ok": True, "event_id": update.event_id, "sequence": update.journal_sequence})

    def handler(self) -> grpc.GenericRpcHandler:
        return grpc.method_handlers_generic_handler(
            SERVICE,
            {
                "GetCapabilities": grpc.unary_unary_rpc_method_handler(
                    self.capabilities,
                    request_deserializer=lambda value: value,
                    response_serializer=lambda value: value,
                ),
                "UploadMedia": grpc.stream_unary_rpc_method_handler(
                    self.upload_media,
                    request_deserializer=lambda value: value,
                    response_serializer=lambda value: value,
                ),
                "Publish": grpc.unary_unary_rpc_method_handler(
                    self.publish,
                    request_deserializer=lambda value: value,
                    response_serializer=lambda value: value,
                ),
            },
        )
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
        producer_cameras = {
            camera: "safety"
            for camera in topology.safety_cameras
            if camera not in topology.camera_owners
        }
        self.producer_ingest = TrackerHostIngest(
            producer_cameras, self._commit_producer
        )
        self.producer_service: ProducerIngressService | None = None
        self.producer_server_bind = os.environ.get(
            "PRODUCER_GRPC_BIND", "0.0.0.0:50052"
        )
        self.producer_event_aggregator = EventAggregator()

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
        if any(item.media_type == "snapshot_jpg" for item in update.media):
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

    def _producer_media_path(self, media_id: str) -> Path | None:
        service = self.producer_service
        if service is None:
            return None
        with service._lock:
            return service.media_paths.get(media_id)

    def _persist_producer_manifest(self, update: TrackerUpdate, manifest: MediaManifest) -> None:
        existing = EdgeMediaManifest.get_or_none(
            EdgeMediaManifest.media_id == manifest.media_id
        )
        if existing is not None:
            if existing.event_id != update.event_id or existing.sha256 != manifest.sha256:
                raise TrackerIngestError("durable_media_conflict")
            return
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

    def _commit_producer(self, update: TrackerUpdate) -> None:
        """Commit Safety evidence synchronously using the canonical Frigate owner."""
        if update.source_type != "safety":
            raise TrackerIngestError("producer_source_mismatch")
        camera_config = self.config.cameras.get(update.camera_id)
        if camera_config is None or camera_config.media_mode.value != "external":
            raise TrackerIngestError("producer_camera_not_external")

        snapshot_manifest = next(
            (item for item in update.media if item.media_type == "snapshot_jpg"), None
        )
        clip_manifest = next(
            (item for item in update.media if item.media_type == "clip"), None
        )
        if update.operation is TrackerOperation.START and snapshot_manifest is None:
            raise TrackerIngestError("producer_start_requires_snapshot")
        if update.operation is TrackerOperation.END and clip_manifest is None:
            raise TrackerIngestError("producer_end_requires_clip")

        for manifest in update.media:
            if manifest.event_id != update.event_id or manifest.camera_id != update.camera_id:
                raise TrackerIngestError("producer_media_identity_mismatch")
            path = self._producer_media_path(manifest.media_id)
            if path is None or not path.is_file():
                raise TrackerIngestError("producer_media_unavailable")
            if path.stat().st_size != manifest.byte_size:
                raise TrackerIngestError("producer_media_size_mismatch")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.sha256:
                raise TrackerIngestError("producer_media_integrity_mismatch")
            self._persist_producer_manifest(update, manifest)

        snapshot_path = (
            self._producer_media_path(snapshot_manifest.media_id)
            if snapshot_manifest is not None
            else None
        )
        width = int(camera_config.detect.width or 0)
        height = int(camera_config.detect.height or 0)
        if snapshot_path is not None:
            image = cv2.imread(str(snapshot_path), cv2.IMREAD_COLOR)
            if image is None:
                raise TrackerIngestError("producer_snapshot_decode_failed")
            width, height = int(image.shape[1]), int(image.shape[0])
        if width <= 0 or height <= 0:
            raise TrackerIngestError("producer_snapshot_dimensions_missing")
        bbox = normalized_xyxy(
            (update.bbox.left, update.bbox.top, update.bbox.right, update.bbox.bottom),
            width,
            height,
        )
        evidence_id: str | None = None
        if snapshot_manifest is not None and snapshot_path is not None:
            evidence_id = hashlib.sha256(
                f"{update.event_id}:{snapshot_manifest.media_id}".encode()
            ).hexdigest()
            evidence = self.producer_event_aggregator.add_evidence(
                evidence_id=evidence_id,
                event_id=update.event_id,
                frame_ref=str(snapshot_path),
                frame_time=update.frame_time,
                width=width,
                height=height,
                boxes=[
                    {
                        "role": "object",
                        "label": update.label,
                        "score": update.score,
                        "normalized_xyxy": bbox,
                    }
                ],
                technical={
                    "source_type": update.source_type,
                    "media_id": snapshot_manifest.media_id,
                    "sha256": snapshot_manifest.sha256,
                    "codec": snapshot_manifest.codec,
                },
            )
            if evidence is None:
                raise TrackerIngestError("producer_evidence_not_durable")

        now = datetime.datetime.fromtimestamp(update.frame_time, datetime.UTC)
        existing = Event.get_or_none(Event.id == update.event_id)
        if existing is None:
            Event.insert(
                {
                    Event.id: update.event_id,
                    Event.label: update.label,
                    Event.sub_label: "camera-safety",
                    Event.camera: update.camera_id,
                    Event.start_time: now,
                    Event.end_time: now if update.operation is TrackerOperation.END else None,
                    Event.top_score: update.score,
                    Event.score: update.score,
                    Event.false_positive: False,
                    Event.zones: [],
                    Event.thumbnail: "",
                    Event.has_clip: clip_manifest is not None,
                    Event.has_snapshot: snapshot_manifest is not None,
                    Event.region: bbox,
                    Event.box: bbox,
                    Event.area: max(1, int((bbox[2] - bbox[0]) * width * (bbox[3] - bbox[1]) * height)),
                    Event.plus_id: "",
                    Event.model_hash: "external",
                    Event.detector_type: "safety",
                    Event.model_type: "external",
                    Event.data: {
                        "type": "producer",
                        "source_type": update.source_type,
                        "score": update.score,
                        "box": bbox,
                        "frame_time": update.frame_time,
                        "producer_node_id": update.node_id,
                    },
                    Event.state: "ACTIVE",
                    Event.revision: 0,
                }
            ).execute()
        else:
            if existing.camera != update.camera_id or existing.label != update.label:
                raise TrackerIngestError("producer_event_identity_mismatch")
            values: dict[Any, Any] = {
                Event.score: update.score,
                Event.top_score: max(float(existing.top_score or 0), update.score),
                Event.has_snapshot: bool(existing.has_snapshot or snapshot_manifest),
                Event.has_clip: bool(existing.has_clip or clip_manifest),
                Event.box: bbox,
                Event.region: bbox,
            }
            if update.operation is TrackerOperation.END:
                values[Event.end_time] = now
            Event.update(values).where(Event.id == update.event_id).execute()

        observation_id = hashlib.sha256(
            f"{update.node_id}:{update.node_epoch}:{update.journal_sequence}".encode()
        ).hexdigest()
        self.producer_event_aggregator.observe(
            observation_id=observation_id,
            event_id=update.event_id,
            kind="event_ended" if update.operation is TrackerOperation.END else "producer_update",
            observed_at=now,
            frame_time=update.frame_time,
            evidence_id=evidence_id,
            payload={
                "source_type": update.source_type,
                "camera": update.camera_id,
                "label": update.label,
                "score": update.score,
                "bbox": bbox,
            },
        )
        # The START/UPDATE snapshot is the canonical alert evidence. END owns
        # the terminal clip and state, but must not replace the ACTIVE image
        # with a post-clear frame where the subject may have left the scene.
        if evidence_id is not None and update.operation is not TrackerOperation.END:
            event = Event.get_by_id(update.event_id)
            label = event.display_label or event.label
            artifact = self.producer_event_aggregator.media.materialize(
                RenderSpec(update.event_id, event.revision, evidence_id),
                EventEvidence.get_by_id(evidence_id),
                label,
            )
            if artifact is None:
                raise TrackerIngestError("producer_canonical_artifact_unavailable")
            Event.update(
                state="ACTIVE" if update.operation is not TrackerOperation.END else "FINALIZING",
                canonical_evidence_id=evidence_id,
                canonical_artifact_id=artifact.id,
                display_label=label,
            ).where(Event.id == update.event_id).execute()
        if update.operation is TrackerOperation.END:
            artifact = self.producer_event_aggregator.finalize(update.event_id)
            event = Event.get_by_id(update.event_id)
            if artifact is not None:
                ReviewSegment.insert(
                    id=update.event_id,
                    camera=update.camera_id,
                    start_time=as_utc(event.start_time).timestamp(),
                    end_time=as_utc(event.end_time).timestamp(),
                    severity="alert",
                    thumb_path=artifact.path,
                    data={
                        "detections": [update.event_id],
                        "objects": [update.label],
                        "verified_objects": [],
                        "sub_labels": [],
                        "zones": [],
                        "audio": [],
                        "thumb_time": update.frame_time,
                        "metadata": {
                            "source_type": update.source_type,
                            "artifact_id": artifact.id,
                        },
                    },
                ).on_conflict_ignore().execute()

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
        if not self.topology.tracker_nodes and not self.producer_ingest.camera_owners:
            return

        async def run_all() -> None:
            self.loop = asyncio.get_running_loop()
            self.producer_service = ProducerIngressService(
                self.producer_ingest,
                Path("/media/frigate") / "producer-media",
                lambda: tuple(
                    {"camera_id": camera, "ready": True}
                    for camera in self.producer_ingest.camera_owners
                ),
            )
            producer_server = await start_producer_server(
                self.producer_server_bind, self.producer_service
            )
            try:
                if self.topology.tracker_nodes:
                    await asyncio.gather(
                        *(self._run_node(node_id) for node_id in self.topology.tracker_nodes)
                    )
                else:
                    while not self.stop_event.is_set() and not self.shutdown.is_set():
                        await asyncio.sleep(0.2)
            finally:
                await producer_server.stop(2)

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
        if node_id == "safety":
            path = self._producer_media_path(media_id)
            if path is None or not path.is_file():
                raise RuntimeError("producer_media_unavailable")
            data = await asyncio.to_thread(path.read_bytes)
            if offset < 0 or offset > len(data):
                raise ValueError("invalid_media_range")
            return data[offset:] if length is None else data[offset : offset + length]

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


class ProducerTransportError(RuntimeError):
    """Producer gRPC transport failed; caller must retry the same identity."""


class ProducerClient:
    """Small synchronous client for the shared Frigate producer ingress."""

    def __init__(self, endpoint: str, node_id: str, node_epoch: str | None = None) -> None:
        self.endpoint = endpoint
        self.node_id = node_id
        self.node_epoch = node_epoch or uuid.uuid4().hex
        self.stream_epoch = uuid.uuid4().hex
        self.sequence = 0
        self.channel = grpc.insecure_channel(endpoint)
        self._publish = self.channel.unary_unary(
            f"/{SERVICE}/Publish",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        self._upload = self.channel.stream_unary(
            f"/{SERVICE}/UploadMedia",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        self._capabilities = self.channel.unary_unary(
            f"/{SERVICE}/GetCapabilities",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )

    def ready(self, timeout: float = 2.0) -> bool:
        try:
            value = _decode(self._capabilities(b"{}", timeout=timeout))
            return bool(value.get("health", {}).get("ready"))
        except (grpc.RpcError, ValueError, TypeError):
            return False

    def upload_media(self, manifest: MediaManifest, content: bytes, timeout: float = 10.0) -> None:
        if len(content) != manifest.byte_size:
            raise ProducerTransportError("media_size_mismatch")
        if hashlib.sha256(content).hexdigest() != manifest.sha256:
            raise ProducerTransportError("media_integrity_mismatch")
        header = {
            "media_id": manifest.media_id,
            "event_id": manifest.event_id,
            "camera_id": manifest.camera_id,
            "media_type": manifest.media_type,
            "codec": manifest.codec,
            "start_time": manifest.start_time,
            "end_time": manifest.end_time,
            "byte_size": manifest.byte_size,
            "sha256": manifest.sha256,
        }

        def messages():
            yield _encode(header)
            for offset in range(0, len(content), 64 * 1024):
                yield content[offset : offset + 64 * 1024]

        try:
            response = _decode(self._upload(messages(), timeout=timeout))
            if not response.get("ok"):
                raise ProducerTransportError("media_upload_rejected")
        except (grpc.RpcError, ValueError, TypeError) as error:
            raise ProducerTransportError(f"media_upload_failed:{error}") from error

    def publish(self, update: TrackerUpdate, timeout: float = 10.0) -> None:
        try:
            response = _decode(
                self._publish(
                    _encode({"update": update.to_json()}), timeout=timeout
                )
            )
            if not response.get("ok"):
                raise ProducerTransportError("producer_update_rejected")
        except (grpc.RpcError, ValueError, TypeError) as error:
            raise ProducerTransportError(f"producer_publish_failed:{error}") from error

    def close(self) -> None:
        self.channel.close()


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
