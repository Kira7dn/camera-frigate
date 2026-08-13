"""Frigate-side maintainer for the compiled external tracker topology.

This module only crosses the process/transport boundary. Canonical Event behavior
still executes in EventProcessor and recognition still consumes EventUpdatePublisher.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

from extension.topology.compiler import PlatformTopologyPlan
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.infrastructure.comms.events_updater import EventUpdatePublisher
from frigate.infrastructure.config import FrigateConfig
from frigate.util.image import SharedMemoryFrameManager

from ..config.fingerprint import (
    canonical_tracker_config_json,
    tracker_config_fingerprint,
)
from ..domain.contracts import TrackerOperation, TrackerUpdate
from ..service.grpc_client import TlsClientConfig, TrackerGrpcClient
from ..service.v1 import tracker_pb2 as pb
from .canonical import TrackerCanonicalStore
from .ingest import TrackerHostIngest

logger = logging.getLogger(__name__)


class TrackerMaintainer(threading.Thread):
    """Apply and maintain the compiled external tracker topology."""

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
        if topology.tracker_nodes and config.runtime.topology_role != "main":
            raise ValueError(
                "external tracker topology must be materialized for the main runtime"
            )
        self.config = config
        self.topology = topology
        self.stop_event = stop_event
        self.event_update_queue = event_update_queue
        self.event_commit_queue = event_commit_queue
        self.store = TrackerCanonicalStore(database)
        self.event_publisher = EventUpdatePublisher()
        self.frame_manager = SharedMemoryFrameManager()
        self._shutdown = threading.Event()
        self._commit_lock = threading.Lock()
        self._clients: dict[str, TrackerGrpcClient] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ingests = {
            node_id: TrackerHostIngest(topology.camera_owners, self._commit)
            for node_id in topology.tracker_nodes
        }

    @staticmethod
    def _tls(node: Any) -> TlsClientConfig:
        return TlsClientConfig(
            root_ca=Path(node.tls.ca).read_bytes(),
            certificate=Path(node.tls.certificate).read_bytes(),
            private_key=Path(node.tls.key).read_bytes(),
            server_name=node.tls.server_name,
        )

    async def _prepare_evidence(
        self, client: TrackerGrpcClient, update: TrackerUpdate
    ) -> TrackerUpdate:
        evidence = update.evidence
        if evidence is None:
            return update
        response = await client.get_evidence(evidence.evidence_id)
        payload = bytes(response.data)
        if response.evidence_id != evidence.evidence_id:
            raise RuntimeError("evidence_identity_mismatch")
        if response.layout != "I420" or len(response.shape) != 2:
            raise RuntimeError("unsupported_evidence_layout")
        if len(payload) != evidence.byte_length:
            raise RuntimeError("evidence_length_mismatch")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != evidence.sha256 or response.sha256 != evidence.sha256:
            raise RuntimeError("evidence_checksum_mismatch")
        frame_name = self._frame_name(update)
        target = self.frame_manager.create(frame_name, len(payload))
        target[:] = payload
        self.frame_manager.close(frame_name)
        return update

    def _prepare_callback(
        self, client_ref: dict[str, TrackerGrpcClient]
    ):
        async def prepare(update: TrackerUpdate) -> TrackerUpdate:
            return await self._prepare_evidence(client_ref["client"], update)

        return prepare

    @staticmethod
    def _frame_name(update: TrackerUpdate) -> str:
        return (
            f"edge-{update.node_id}-{update.camera_id}-"
            f"{update.stream_epoch}-{update.frame_seq}"
        )

    @staticmethod
    def _event_state(update: TrackerUpdate) -> EventStateEnum:
        return {
            TrackerOperation.START: EventStateEnum.start,
            TrackerOperation.UPDATE: EventStateEnum.update,
            TrackerOperation.END: EventStateEnum.end,
        }[update.operation]

    def _event_data(self, update: TrackerUpdate, frame_name: str) -> dict[str, Any]:
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
                "attributes": update.attributes,
                "current_zones": list(update.current_zones),
                "entered_zones": list(update.entered_zones),
                "path_data": [list(point) for point in update.path],
                "current_estimated_speed": update.speed or 0,
                "raw_track_id": update.track_id,
                "tracker_node_id": update.node_id,
                "tracker_node_epoch": update.node_epoch,
                "tracker_stream_epoch": update.stream_epoch,
                "tracker_journal_sequence": update.journal_sequence,
                "observed_in_frame": update.evidence is not None,
            }
        )
        data.setdefault("start_time", update.frame_time)
        data.setdefault("end_time", None)
        if update.operation is TrackerOperation.END:
            data["end_time"] = update.frame_time
        if frame_name:
            data["_recognition_evidence_owned"] = True
        return data

    def _wait_for_receipt(self, receipt_id: str, deadline: float) -> None:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("tracker_event_commit_timeout")
            try:
                received_id, success, detail = self.event_commit_queue.get(
                    timeout=remaining
                )
            except queue.Empty as error:
                raise TimeoutError("tracker_event_commit_timeout") from error
            if received_id != receipt_id:
                raise RuntimeError("tracker_event_receipt_order_mismatch")
            if not success:
                raise RuntimeError(f"tracker_event_commit_failed:{detail}")
            return

    def _commit(self, update: TrackerUpdate) -> None:
        node = self.config.tracker[update.node_id]
        frame_name = self._frame_name(update) if update.evidence is not None else ""
        data = self._event_data(update, frame_name)
        state = self._event_state(update)
        receipt_id = uuid.uuid4().hex
        canonical = (
            EventTypeEnum.tracked_object,
            state,
            update.camera_id,
            frame_name,
            data,
        )
        with self._commit_lock:
            try:
                self.event_update_queue.put(
                    (*canonical, receipt_id), timeout=node.deadline
                )
                self.event_publisher.publish(canonical)
                self._wait_for_receipt(
                    receipt_id, time.monotonic() + node.deadline
                )
                self.store.accept(update)
            except Exception:
                if frame_name:
                    self.frame_manager.delete(frame_name)
                raise

    async def _run_node(self, node_id: str) -> None:
        node = self.config.tracker[node_id]
        tls = await asyncio.to_thread(self._tls, node)
        ingest = self._ingests[node_id]
        while not self.stop_event.is_set() and not self._shutdown.is_set():
            client: TrackerGrpcClient | None = None
            try:
                client_ref: dict[str, TrackerGrpcClient] = {}
                client = TrackerGrpcClient(
                    node.endpoint,
                    f"frigate-main-{node_id}",
                    tls,
                    ingest,
                    deadline=node.deadline,
                    output_capacity=node.output_capacity,
                    durable_sequence=self.store.last_sequence,
                    prepare=self._prepare_callback(client_ref),
                )
                client_ref["client"] = client
                self._clients[node_id] = client
                await client.connect(
                    config_json=canonical_tracker_config_json(self.config, node_id),
                    config_hash=tracker_config_fingerprint(self.config, node_id),
                )
                while (
                    client.healthy
                    and not self.stop_event.is_set()
                    and not self._shutdown.is_set()
                ):
                    await asyncio.sleep(0.2)
            except Exception:
                if not self.stop_event.is_set() and not self._shutdown.is_set():
                    logger.exception("Tracker node %s disconnected", node_id)
            finally:
                self._clients.pop(node_id, None)
                if client is not None:
                    await client.close()
            if not self.stop_event.is_set() and not self._shutdown.is_set():
                await asyncio.sleep(min(1.0, node.deadline))

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(self._run_node(node_id) for node_id in self.topology.tracker_nodes)
        )

    def run(self) -> None:
        if not self.topology.tracker_nodes:
            return
        asyncio.run(self._run())

    def request_stop(self) -> None:
        self._shutdown.set()

    async def fetch_media(
        self,
        node_id: str,
        media_id: str,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        loop = self._loop
        if loop is None:
            raise RuntimeError("tracker_host_unavailable")

        async def fetch() -> bytes:
            client = self._clients.get(node_id)
            if client is None or not client.healthy:
                raise RuntimeError("tracker_edge_unavailable")
            return await client.stream_media(media_id, offset=offset, length=length)

        future = asyncio.run_coroutine_threadsafe(fetch(), loop)
        return await asyncio.wrap_future(future)

    def control_camera(
        self, camera_id: str, operation: str, payload: dict[str, object]
    ) -> bool:
        node_id = self.topology.camera_owners.get(camera_id)
        loop = self._loop
        if node_id is None or loop is None:
            return False
        operations = {
            "enable": pb.CONTROL_OPERATION_ENABLE,
            "disable": pb.CONTROL_OPERATION_DISABLE,
            "manual_ptz": pb.CONTROL_OPERATION_MANUAL_PTZ,
            "preset": pb.CONTROL_OPERATION_PRESET,
            "calibrate": pb.CONTROL_OPERATION_CALIBRATE,
            "topology_drain": pb.CONTROL_OPERATION_TOPOLOGY_DRAIN,
        }
        node = self.config.tracker[node_id]

        def apply_config_patch(response: pb.ControlResponse) -> None:
            if not response.HasField("config_patch_json"):
                return
            patch = json.loads(response.config_patch_json)
            expected_key = (
                f"cameras.{camera_id}.onvif.autotracking.movement_weights"
            )
            if set(patch) != {expected_key} or not isinstance(
                patch[expected_key], str
            ):
                raise RuntimeError("invalid_tracker_config_patch")
            self.config.cameras[
                camera_id
            ].onvif.autotracking.movement_weights = patch[expected_key]

        async def control() -> bool:
            client = self._clients.get(node_id)
            if client is None:
                return False
            runtime_patch: dict[str, object] = {}
            if operation in ("enable", "disable"):
                enabled = operation == "enable"
                if payload.get("target") == "autotracking":
                    path = f"cameras.{camera_id}.onvif.autotracking.enabled"
                    runtime_patch[path] = enabled
                else:
                    path = f"cameras.{camera_id}.enabled"
                    runtime_patch[path] = enabled
            response = await client.control_camera(
                camera_id, operations[operation], payload
            )
            if not response.accepted:
                return False
            if runtime_patch:
                if payload.get("target") == "autotracking":
                    self.config.cameras[camera_id].onvif.autotracking.enabled = bool(
                        runtime_patch[
                            f"cameras.{camera_id}.onvif.autotracking.enabled"
                        ]
                    )
                else:
                    self.config.cameras[camera_id].enabled = bool(
                        runtime_patch[f"cameras.{camera_id}.enabled"]
                    )
            apply_config_patch(response)
            return bool(response.accepted)

        future = asyncio.run_coroutine_threadsafe(control(), loop)
        try:
            return bool(future.result(timeout=node.deadline))
        except Exception:
            logger.exception("Tracker control failed camera=%s", camera_id)
            return False

    def close(self) -> None:
        self.event_publisher.stop()
