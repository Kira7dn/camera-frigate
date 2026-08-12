"""Managed same-host tracker node assembled from original Frigate components."""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import uuid
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

from peewee import SqliteDatabase

from frigate.domain.camera import CameraMetrics, PTZMetrics
from frigate.domain.camera.maintainer import CameraMaintainer
from frigate.domain.camera.runtime import (
    CAMERA_RUNTIME_MODELS,
    camera_runtime_config,
    ensure_runtime_dirs,
    start_detector_runtime,
)
from frigate.infrastructure.comms.zmq_proxy import ZmqProxy
from frigate.infrastructure.config import FrigateConfig
from frigate.domain.object_detection.base import ObjectDetectProcess
from frigate.infrastructure.output.output import OutputProcess
from frigate.domain.ptz.autotrack import PtzAutoTrackerThread
from frigate.domain.ptz.onvif import OnvifController
from frigate.domain.record.record import RecordProcess
from camera_platform.topology.compiler import compile_topology
from frigate.util.builtin import empty_and_close_queue
from frigate.util.image import UntrackedSharedMemory

from .processor import EdgeTrackedObjectProcessor
from .evidence import EvidenceRing
from .journal import EdgeJournal, SpoolFullError
from .media import MediaAuthority
from .producer import ProducerContext, TrackerProducerCore

logger = logging.getLogger(__name__)


class _EdgeDispatcher:
    """PTZ status sink; edge telemetry is carried by TrackerService health."""

    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        logger.debug("edge status topic=%s retain=%s payload=%s", topic, retain, payload)


class _FrameConsumer(threading.Thread):
    def __init__(
        self,
        frames: Queue,
        processors: dict[str, EdgeTrackedObjectProcessor],
        stop_event: MpEvent,
    ) -> None:
        super().__init__(name="tracker_edge_frames")
        self.frames = frames
        self.processors = processors
        self.stop_event = stop_event
        self.degraded = False
        self.detail = ""

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                camera, frame_name, frame_time, objects, motion, regions = (
                    self.frames.get(timeout=1)
                )
            except queue.Empty:
                continue
            processor = self.processors.get(camera)
            if processor is not None:
                try:
                    processor.process(
                        frame_name, frame_time, objects, motion, regions
                    )
                except SpoolFullError:
                    self.degraded = True
                    self.detail = "spool_full"

    def finish_active(self, code: str, detail: str) -> None:
        for processor in self.processors.values():
            processor.fail_active(code, detail, retryable=False)


class TrackerNodeRuntime:
    """Own capture/detect/Norfair/CameraState/PTZ for one configured edge node."""

    def __init__(
        self,
        config: FrigateConfig,
        node_id: str,
        manager: SyncManager,
        stop_event: MpEvent,
        *,
        spool_dir: str | Path,
        media_dir: str | Path,
    ) -> None:
        topology = compile_topology(config)
        if (
            config.runtime.topology_role != "tracker"
            or config.runtime.topology_node_id != node_id
        ):
            raise ValueError(
                f"tracker node {node_id} requires its compiled tracker runtime view"
            )
        node_plan = topology.node(node_id)
        node = config.tracker[node_id]
        if not set(node.cameras).issubset(config.cameras):
            raise ValueError("tracker node references an unknown camera")
        self.config = camera_runtime_config(
            config, edge_node_id=node_id, topology=topology
        )
        cameras = self.config.cameras
        self.node_id = node_plan.node_id
        self.node_epoch = uuid.uuid4().hex
        self.node_config = node
        self.manager = manager
        self.stop_event = stop_event
        self.detection_queue: Queue = mp.Queue(maxsize=max(4, len(cameras) * 4))
        self.frames_queue: Queue = mp.Queue(maxsize=max(4, len(cameras) * 2))
        self.camera_metrics = manager.dict()
        self.ptz_metrics: dict[str, PTZMetrics] = {}
        self._config_patch_lock = threading.Lock()
        self._config_patches: dict[str, dict[str, object]] = {}
        self.detectors: dict[str, ObjectDetectProcess] = {}
        self.detection_shms: list[UntrackedSharedMemory] = []
        ensure_runtime_dirs(self.config)
        spool_root = Path(spool_dir)
        spool_root.mkdir(parents=True, exist_ok=True)
        edge_database = spool_root / "recordings.db"
        self.config.database.path = str(edge_database)
        self.edge_database = SqliteDatabase(edge_database)
        self.edge_database.bind(CAMERA_RUNTIME_MODELS)
        self.edge_database.create_tables(CAMERA_RUNTIME_MODELS, safe=True)
        self.edge_database.close()
        self.journal = EdgeJournal(
            spool_root / "journal.db",
            max_bytes=node.spool.max_bytes,
            retention_seconds=node.spool.retention,
        )
        self.recovered_failures = self.journal.recover_active()
        self.media = MediaAuthority(media_dir)
        self.evidence: dict[str, EvidenceRing] = {
            camera: EvidenceRing(
                node_id,
                camera,
                max_bytes=node.evidence.memory_bytes_per_camera,
                ttl_seconds=node.evidence.ttl,
            )
            for camera in cameras
        }
        for camera, camera_config in cameras.items():
            self.camera_metrics[camera] = CameraMetrics(manager)
            self.ptz_metrics[camera] = PTZMetrics(
                autotracker_enabled=camera_config.onvif.autotracking.enabled
            )
        self.onvif = OnvifController(self.config, self.ptz_metrics)
        self.dispatcher = _EdgeDispatcher()
        self.ptz = PtzAutoTrackerThread(
            self.config,
            self.onvif,
            self.ptz_metrics,
            self.dispatcher,  # type: ignore[arg-type]
            stop_event,
            self._capture_config_patch,
        )
        self.processors: dict[str, EdgeTrackedObjectProcessor] = {}
        self.detection_proxy = ZmqProxy()
        self.recording = RecordProcess(self.config, stop_event)
        self.output = OutputProcess(self.config, stop_event)
        self.consumer = _FrameConsumer(
            self.frames_queue, self.processors, stop_event
        )
        self.cameras = CameraMaintainer(
            self.config,
            self.detection_queue,
            self.frames_queue,
            self.camera_metrics,
            self.ptz_metrics,
            stop_event,
            manager,
        )
        self._drained = False

    def _capture_config_patch(
        self, camera: str, patch: dict[str, object]
    ) -> None:
        with self._config_patch_lock:
            self._config_patches[camera] = patch

    def pop_config_patch(self, camera: str) -> dict[str, object] | None:
        with self._config_patch_lock:
            return self._config_patches.pop(camera, None)

    def _start_processors(self) -> None:
        for camera in self.config.cameras:
            context = ProducerContext(
                self.node_id,
                self.node_epoch,
                camera,
                uuid.uuid4().hex,
            )
            producer = TrackerProducerCore(context, self.journal)
            self.processors[camera] = EdgeTrackedObjectProcessor(
                self.config,
                context,
                producer,
                self.evidence[camera],
                self.ptz,
                lambda update: None,
                self.media,
            )

    def start(self) -> None:
        self._start_detectors()
        self.recording.start()
        self.output.start()
        self._start_processors()
        self.ptz.start()
        self.consumer.start()
        self.cameras.start()

    def _start_detectors(self) -> None:
        detector_runtime = start_detector_runtime(
            self.config,
            self.detection_queue,
            self.stop_event,
        )
        self.detectors = detector_runtime.processes
        self.detection_shms.extend(detector_runtime.shared_memory)

    def terminal_state(self) -> dict[str, int]:
        return {
            "pending_ack": self.journal.pending_count,
            "spool_bytes": self.journal.logical_bytes,
            "active": sum(
                processor.terminal_state()["active"]
                for processor in self.processors.values()
            ),
            "pinned_evidence": sum(
                ring.pinned_count for ring in self.evidence.values()
            ),
        }

    def camera_health(self) -> tuple[dict[str, object], ...]:
        snapshots = []
        for camera, metrics in self.camera_metrics.items():
            camera_fps = float(metrics.camera_fps.value)
            process_fps = float(metrics.process_fps.value)
            capture_pid = int(metrics.capture_process_pid.value)
            process_pid = int(metrics.process_pid.value)
            snapshots.append(
                {
                    "camera_id": camera,
                    "ready": (
                        camera_fps > 0
                        and process_fps > 0
                        and capture_pid > 0
                        and process_pid > 0
                    ),
                    "camera_fps": camera_fps,
                    "process_fps": process_fps,
                    "capture_pid": capture_pid,
                    "process_pid": process_pid,
                }
            )
        return tuple(snapshots)

    @property
    def degraded(self) -> bool:
        return self.consumer.degraded

    def stop(self, reason: str = "topology_drain") -> None:
        self.drain_active(reason)
        self.stop_event.set()
        self.cameras.join(timeout=self.node_config.shutdown_drain)
        self.consumer.join(timeout=self.node_config.shutdown_drain)
        self.ptz.join(timeout=self.node_config.shutdown_drain)
        self.onvif.close()
        self.output.terminate()
        self.output.join(timeout=self.node_config.shutdown_drain)
        self.recording.terminate()
        self.recording.join(timeout=self.node_config.shutdown_drain)
        for processor in self.processors.values():
            processor.close()
        self.detection_proxy.stop()
        for detector in self.detectors.values():
            detector.stop()
        empty_and_close_queue(self.detection_queue)
        empty_and_close_queue(self.frames_queue)
        for shm in self.detection_shms:
            shm.close()
        self.media.close()
        self.journal.close()
        if not self.edge_database.is_closed():
            self.edge_database.close()

    def drain_active(self, reason: str = "topology_drain") -> None:
        if self._drained:
            return
        self.consumer.finish_active(reason, "tracker node stopping")
        self._drained = True
