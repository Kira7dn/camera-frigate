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

from frigate.camera import CameraMetrics, PTZMetrics
from frigate.camera.maintainer import CameraMaintainer
from frigate.config import FrigateConfig
from frigate.object_detection.base import ObjectDetectProcess
from frigate.ptz.autotrack import PtzAutoTrackerThread
from frigate.ptz.onvif import OnvifController
from frigate.util.builtin import empty_and_close_queue
from frigate.util.image import UntrackedSharedMemory

from .edge_processor import EdgeTrackedObjectProcessor
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
        node = config.tracker.nodes.get(node_id)
        if node is None:
            raise ValueError(f"unknown tracker node: {node_id}")
        cameras = {
            name: config.cameras[name]
            for name in node.cameras
            if name in config.cameras
        }
        if set(cameras) != set(node.cameras):
            raise ValueError("tracker node references an unknown camera")
        self.config = config.model_copy(update={"cameras": cameras})
        self.node_id = node_id
        self.node_epoch = uuid.uuid4().hex
        self.node_config = node
        self.manager = manager
        self.stop_event = stop_event
        self.detection_queue: Queue = mp.Queue(maxsize=max(4, len(cameras) * 4))
        self.frames_queue: Queue = mp.Queue(maxsize=max(4, len(cameras) * 2))
        self.camera_metrics = manager.dict()
        self.ptz_metrics: dict[str, PTZMetrics] = {}
        self.detectors: dict[str, ObjectDetectProcess] = {}
        self.detection_shms: list[UntrackedSharedMemory] = []
        spool_root = Path(spool_dir)
        spool_root.mkdir(parents=True, exist_ok=True)
        self.journal = EdgeJournal(
            spool_root / "journal.db",
            max_bytes=node.spool.max_bytes,
            retention_seconds=node.spool.retention,
        )
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
        for camera in cameras:
            self.camera_metrics[camera] = CameraMetrics(manager)
            self.ptz_metrics[camera] = PTZMetrics(
                autotracker_enabled=cameras[camera].onvif.autotracking.enabled
            )
        self.onvif = OnvifController(self.config, self.ptz_metrics)
        self.dispatcher = _EdgeDispatcher()
        self.ptz = PtzAutoTrackerThread(
            self.config,
            self.onvif,
            self.ptz_metrics,
            self.dispatcher,  # type: ignore[arg-type]
            stop_event,
        )
        self.processors: dict[str, EdgeTrackedObjectProcessor] = {}
        for camera in cameras:
            context = ProducerContext(
                node_id,
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
            )
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
            edge_node_id=node_id,
        )

    def start(self) -> None:
        self._start_detectors()
        self.ptz.start()
        self.consumer.start()
        self.cameras.start()

    def _start_detectors(self) -> None:
        largest_frame = max(
            (
                detector.model.height * detector.model.width * 3
                if detector.model is not None
                else 320
                for detector in self.config.detectors.values()
            ),
            default=320,
        )
        camera_names = list(self.config.cameras)
        for name in camera_names:
            for shm_name, size in (
                (name, largest_frame),
                (f"out-{name}", 20 * 6 * 4),
            ):
                try:
                    shm = UntrackedSharedMemory(
                        name=shm_name, create=True, size=size
                    )
                except FileExistsError:
                    shm = UntrackedSharedMemory(name=shm_name)
                self.detection_shms.append(shm)
        for name, detector_config in self.config.detectors.items():
            detector = ObjectDetectProcess(
                name,
                self.detection_queue,
                camera_names,
                self.config,
                detector_config,
                self.stop_event,
            )
            self.detectors[name] = detector
            if not detector.ready_event.wait(timeout=60):
                raise RuntimeError(f"detector {name} did not become ready")

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
        self.consumer.finish_active(reason, "tracker node stopping")
        self.stop_event.set()
        self.cameras.join(timeout=self.node_config.shutdown_drain)
        self.consumer.join(timeout=self.node_config.shutdown_drain)
        self.ptz.join(timeout=self.node_config.shutdown_drain)
        self.onvif.close()
        for detector in self.detectors.values():
            detector.stop()
        empty_and_close_queue(self.detection_queue)
        empty_and_close_queue(self.frames_queue)
        for shm in self.detection_shms:
            shm.close()
        self.journal.close()
