"""Compose native Frigate camera components for an externally owned lane."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import queue
import signal
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

from peewee import SqliteDatabase

from extension.topology.fingerprint import canonical_json, fingerprint, model_value
from extension.topology.loader import PlatformConfigLoader
from frigate.domain.camera import CameraMetrics, PTZMetrics
from frigate.domain.camera.maintainer import CameraMaintainer
from frigate.domain.camera.runtime import (
    CAMERA_HISTORY_MODELS,
    CAMERA_RUNTIME_MODELS,
    ensure_runtime_dirs,
    start_detector_runtime,
)
from frigate.domain.camera.state import CameraState
from frigate.domain.ptz.autotrack import DispatcherProtocol, PtzAutoTrackerThread
from frigate.domain.ptz.onvif import OnvifController
from frigate.domain.record.record import RecordProcess
from frigate.domain.track.tracked_object import TrackedObject
from frigate.infrastructure.comms.detections_updater import (
    DetectionPublisher,
    DetectionTypeEnum,
)
from frigate.infrastructure.comms.object_detector_signaler import DetectorProxy
from frigate.infrastructure.comms.zmq_proxy import ZmqProxy
from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.output.output import OutputProcess
from frigate.log import setup_logging
from frigate.util.builtin import empty_and_close_queue
from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory

logger = logging.getLogger(__name__)


class TrackerOperation(StrEnum):
    START = "START"
    UPDATE = "UPDATE"
    END = "END"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    node_id: str
    node_epoch: str
    camera_id: str
    stream_epoch: str
    journal_sequence: int
    frame_seq: int
    source_pts: int
    frame_time: float
    event_id: str
    track_id: str
    operation: TrackerOperation
    label: str
    score_history: tuple[float, ...]
    score: float
    bbox: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    current_zones: tuple[str, ...] = ()
    entered_zones: tuple[str, ...] = ()
    path: tuple[tuple[float, float], ...] = ()
    speed: float | None = None
    motion: dict[str, Any] = field(default_factory=dict)
    region: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_id(self) -> str:
        """Return the producer-owned downstream correlation id."""
        return self.event_id

    def to_json(self) -> str:
        value = asdict(self)
        value["operation"] = self.operation.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> TrackerUpdate:
        data = json.loads(value)
        data["operation"] = TrackerOperation(data["operation"])
        data["bbox"] = BoundingBox(**data["bbox"])
        for key in ("score_history", "current_zones", "entered_zones"):
            data[key] = tuple(data.get(key, ()))
        data["path"] = tuple(tuple(item) for item in data.get("path", ()))
        return cls(**data)


def should_save_snapshot(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> bool:
    """Use Frigate's existing snapshot configuration without new heuristics."""
    if obj.false_positive or obj.obj_data["position_changes"] == 0:
        return False
    snapshot = config.cameras[camera].snapshots
    return bool(
        snapshot.enabled
        and (
            not snapshot.required_zones
            or set(obj.entered_zones) & set(snapshot.required_zones)
        )
    )


def should_retain_recording(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> bool:
    """Use Frigate's existing recording configuration without new heuristics."""
    return bool(
        not obj.false_positive
        and config.cameras[camera].record.enabled
        and obj.obj_data["position_changes"] > 0
        and obj.max_severity is not None
    )


def apply_media_policy(
    config: FrigateConfig, camera: str, obj: TrackedObject
) -> None:
    """Apply the same media decisions used by embedded object processing."""
    obj.has_snapshot = should_save_snapshot(config, camera, obj) or (
        obj.face_snapshot is not None
    )
    obj.has_clip = should_retain_recording(config, camera, obj)


def publish_video_detection(
    publisher: Any,
    camera: str,
    frame_name: str,
    frame_time: float,
    objects: list[dict[str, Any]],
    motion: list[tuple[int, int, int, int]],
    regions: list[tuple[int, int, int, int]],
) -> None:
    """Publish the native payload consumed by Frigate recorder and output."""
    publisher.publish(
        (camera, frame_name, frame_time, objects, motion, regions),
        DetectionTypeEnum.video.value,
    )


class TrackerJournal:
    """Persist producer updates until main acknowledges them."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            """create table if not exists updates(
                sequence integer primary key autoincrement,
                node_epoch text not null,
                event_id text not null,
                payload text not null,
                acknowledged integer not null default 0
            )"""
        )
        self._db.commit()

    def append(self, update: TrackerUpdate) -> TrackerUpdate:
        with self._lock:
            cursor = self._db.execute(
                "insert into updates(node_epoch,event_id,payload) values(?,?,?)",
                (update.node_epoch, update.event_id, update.to_json()),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("tracker journal did not allocate a sequence")
            sequence = int(cursor.lastrowid)
            value = TrackerUpdate.from_json(update.to_json())
            object.__setattr__(value, "journal_sequence", sequence)
            self._db.execute(
                "update updates set payload=? where sequence=?",
                (value.to_json(), sequence),
            )
            self._db.commit()
            return value

    def replay(self, after: int = 0) -> tuple[TrackerUpdate, ...]:
        with self._lock:
            rows = self._db.execute(
                "select payload from updates where sequence>? and acknowledged=0 order by sequence",
                (after,),
            ).fetchall()
        return tuple(TrackerUpdate.from_json(row[0]) for row in rows)

    def acknowledge(self, sequence: int, event_id: str, node_epoch: str) -> bool:
        with self._lock:
            cursor = self._db.execute(
                """update updates set acknowledged=1
                   where sequence=? and event_id=? and node_epoch=? and acknowledged=0""",
                (sequence, event_id, node_epoch),
            )
            self._db.commit()
            return cursor.rowcount == 1

    @property
    def pending_count(self) -> int:
        with self._lock:
            row = self._db.execute(
                "select count(*) from updates where acknowledged=0"
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def tracker_config_fingerprint(config: FrigateConfig, node_id: str) -> str:
    """Hash behavior while excluding deployment-specific TLS mount paths."""
    node = config.tracker[node_id]
    return fingerprint(
        canonical_json(
            {
                "node_id": node_id,
                "node": model_value(node, exclude={"tls"}),
                "tls_server_name": node.tls.server_name,
                "cameras": {
                    name: model_value(config.cameras[name])
                    for name in sorted(config.cameras)
                },
                "model": model_value(config.model),
                "detectors": {
                    name: model_value(value)
                    for name, value in sorted(config.detectors.items())
                },
            }
        )
    )


class _Dispatcher(DispatcherProtocol):
    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        logger.debug("tracker status topic=%s retain=%s", topic, retain)


class CameraTrackAdapter:
    """Translate native CameraState callbacks into durable transport updates."""

    def __init__(
        self,
        config: FrigateConfig,
        node_id: str,
        node_epoch: str,
        camera: str,
        ptz: PtzAutoTrackerThread,
        publish: Callable[[TrackerUpdate], TrackerUpdate],
    ) -> None:
        self.config = config
        self.node_id = node_id
        self.node_epoch = node_epoch
        self.camera = camera
        self.stream_epoch = uuid.uuid4().hex
        self.publish = publish
        self.frame_manager = SharedMemoryFrameManager()
        self.publisher = DetectionPublisher(DetectionTypeEnum.all.value)
        self.frame_seq = 0
        self.motion: list[tuple[int, int, int, int]] = []
        self.regions: list[tuple[int, int, int, int]] = []
        self.event_ids: dict[str, str] = {}
        self.state = CameraState(camera, config, self.frame_manager, ptz)
        self.state.on("start", self._start)
        self.state.on("update", self._update)
        self.state.on("end", self._end)
        self.state.on("autotrack", self._autotrack)
        self.ptz = ptz.ptz_autotracker

    def process(
        self,
        frame_name: str,
        frame_time: float,
        objects: dict[str, dict[str, Any]],
        motion: list[tuple[int, int, int, int]],
        regions: list[tuple[int, int, int, int]],
    ) -> None:
        self.frame_seq += 1
        self.motion = motion
        self.regions = regions
        self.state.update(frame_name, frame_time, objects, motion, regions)
        publish_video_detection(
            self.publisher,
            self.camera,
            frame_name,
            frame_time,
            [obj.to_dict() for obj in self.state.tracked_objects.values()],
            motion,
            regions,
        )

    def _start(self, camera: str, obj: TrackedObject, *_: object) -> None:
        track_id = str(obj.obj_data["id"])
        self.event_ids[track_id] = uuid.uuid4().hex[:30]
        self._emit(TrackerOperation.START, obj)

    def _update(self, camera: str, obj: TrackedObject, *_: object) -> None:
        apply_media_policy(self.config, camera, obj)
        self._emit(TrackerOperation.UPDATE, obj)

    def _end(self, camera: str, obj: TrackedObject, *_: object) -> None:
        apply_media_policy(self.config, camera, obj)
        self._emit(TrackerOperation.END, obj)
        self.event_ids.pop(str(obj.obj_data["id"]), None)
        if not obj.false_positive:
            self.ptz.end_object(camera, obj)

    def _autotrack(self, camera: str, obj: TrackedObject, *_: object) -> None:
        self.ptz.autotrack_object(camera, obj)

    def _emit(self, operation: TrackerOperation, obj: TrackedObject) -> None:
        data = obj.to_dict()
        track_id = str(obj.obj_data["id"])
        box = tuple(int(value) for value in data["box"])
        state = dict(data)
        state.update(
            {
                "frame_seq": self.frame_seq,
                "source_pts": int(float(data["frame_time"]) * 1_000_000),
                "track_id": track_id,
                "score_history": tuple(obj.score_history),
                "path": tuple(point for point, _ in obj.path_data),
                "speed": obj.current_estimated_speed,
                "motion": {"boxes": tuple(self.motion)},
                "region": {"boxes": tuple(self.regions)},
            }
        )
        self.publish(
            TrackerUpdate(
                self.node_id,
                self.node_epoch,
                self.camera,
                self.stream_epoch,
                0,
                self.frame_seq,
                state["source_pts"],
                float(data["frame_time"]),
                self.event_ids[track_id],
                track_id,
                operation,
                str(data["label"]),
                tuple(float(value) for value in obj.score_history),
                float(data["score"]),
                BoundingBox(*box),
                dict(data.get("attributes") or {}),
                state,
                tuple(data.get("current_zones") or ()),
                tuple(data.get("entered_zones") or ()),
                tuple(state["path"]),
                state["speed"],
                state["motion"],
                state["region"],
            )
        )

    def close(self) -> None:
        self.publisher.stop()


class TrackerRuntime:
    """Own only the composition of existing Frigate runtime components."""

    def __init__(
        self,
        config: FrigateConfig,
        node_id: str,
        manager: SyncManager,
        stop_event: MpEvent,
        spool_dir: str | Path,
    ) -> None:
        if (
            config.runtime.topology_role != "tracker"
            or config.runtime.topology_node_id != node_id
        ):
            raise ValueError("tracker requires its compiled isolated config")
        self.config = config
        self.node_id = node_id
        self.node_epoch = uuid.uuid4().hex
        self.manager = manager
        self.stop_event = stop_event
        self.node_config = config.tracker[node_id]
        self.detection_queue: Queue = mp.Queue(maxsize=max(4, len(config.cameras) * 4))
        self.frames_queue: Queue = mp.Queue(maxsize=max(4, len(config.cameras) * 2))
        self.camera_metrics = manager.dict()
        self.ptz_metrics: dict[str, PTZMetrics] = {}
        ensure_runtime_dirs(config)
        spool_root = Path(spool_dir)
        spool_root.mkdir(parents=True, exist_ok=True)
        config.database.path = str(spool_root / "camera.db")
        database = SqliteDatabase(config.database.path)
        models = CAMERA_RUNTIME_MODELS + CAMERA_HISTORY_MODELS
        database.bind(models)
        database.create_tables(models, safe=True)
        database.close()
        self.journal = TrackerJournal(spool_root / "journal.db")
        for camera, camera_config in config.cameras.items():
            self.camera_metrics[camera] = CameraMetrics(manager)
            self.ptz_metrics[camera] = PTZMetrics(
                autotracker_enabled=camera_config.onvif.autotracking.enabled
            )
        self.onvif = OnvifController(config, self.ptz_metrics)
        self.ptz = PtzAutoTrackerThread(
            config, self.onvif, self.ptz_metrics, _Dispatcher(), stop_event
        )
        self.zmq_proxy: ZmqProxy | None = None
        self.detector_proxy: DetectorProxy | None = None
        self.detectors = {}
        self.shms: list[UntrackedSharedMemory] = []
        self.recording = RecordProcess(config, stop_event)
        self.output = OutputProcess(config, stop_event)
        self.adapters: dict[str, CameraTrackAdapter] = {}
        self.degraded = False
        self.consumer = threading.Thread(
            target=self._consume, name="tracker_frames", daemon=True
        )
        self.cameras = CameraMaintainer(
            config,
            self.detection_queue,
            self.frames_queue,
            self.camera_metrics,
            self.ptz_metrics,
            stop_event,
            manager,
        )

    def start(self, publish: Callable[[TrackerUpdate], TrackerUpdate]) -> None:
        """Start every native dependency in the same order as FrigateApp."""
        self.zmq_proxy = ZmqProxy()
        self.detector_proxy = DetectorProxy()
        detectors = start_detector_runtime(
            self.config, self.detection_queue, self.stop_event
        )
        self.detectors = detectors.processes
        self.shms = detectors.shared_memory
        self.recording.start()
        self.output.start()
        for camera in self.config.cameras:
            self.adapters[camera] = CameraTrackAdapter(
                self.config,
                self.node_id,
                self.node_epoch,
                camera,
                self.ptz,
                publish,
            )
        self.ptz.start()
        self.consumer.start()
        self.cameras.start()

    def _consume(self) -> None:
        while not self.stop_event.is_set():
            try:
                camera, frame_name, frame_time, objects, motion, regions = (
                    self.frames_queue.get(timeout=1)
                )
            except queue.Empty:
                continue
            try:
                self.adapters[camera].process(
                    frame_name, frame_time, objects, motion, regions
                )
            except (KeyError, RuntimeError, ValueError):
                logger.exception("Tracker frame processing failed camera=%s", camera)
                self.degraded = True

    def camera_health(self) -> tuple[dict[str, object], ...]:
        output = []
        for camera, metrics in self.camera_metrics.items():
            camera_fps = float(metrics.camera_fps.value)
            process_fps = float(metrics.process_fps.value)
            capture_pid = int(metrics.capture_process_pid.value)
            process_pid = int(metrics.process_pid.value)
            output.append(
                {
                    "camera_id": camera,
                    "ready": camera_fps > 0
                    and process_fps > 0
                    and capture_pid > 0
                    and process_pid > 0,
                    "camera_fps": camera_fps,
                    "process_fps": process_fps,
                    "capture_pid": capture_pid,
                    "process_pid": process_pid,
                }
            )
        return tuple(output)

    def stop(self) -> None:
        """Stop native components and release tracker-owned IPC resources."""
        self.stop_event.set()
        timeout = self.node_config.shutdown_drain
        self.cameras.join(timeout=timeout)
        self.consumer.join(timeout=timeout)
        self.ptz.join(timeout=timeout)
        self.onvif.close()
        self.output.terminate()
        self.output.join(timeout=timeout)
        self.recording.terminate()
        self.recording.join(timeout=timeout)
        for adapter in self.adapters.values():
            adapter.close()
        for detector in self.detectors.values():
            detector.stop()
        if self.detector_proxy is not None:
            self.detector_proxy.stop()
        if self.zmq_proxy is not None:
            self.zmq_proxy.stop()
        empty_and_close_queue(self.detection_queue)
        empty_and_close_queue(self.frames_queue)
        for shm in self.shms:
            shm.close()
            shm.unlink()
        self.journal.close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="camera-tracker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--bind", default="0.0.0.0:50052")
    parser.add_argument("--spool-dir", default="/var/lib/camera-tracker/spool")
    parser.add_argument("--media-dir", default="/media/tracker")
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--client-ca", required=True)
    parser.add_argument("--allow-client", action="append", required=True)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    from extension.tracker.transport import TrackerService, TrackerTls, start_server

    config = await asyncio.to_thread(
        PlatformConfigLoader.load_runtime,
        args.config,
        expected_role="tracker",
        expected_node_id=args.node_id,
        install=True,
    )
    manager = mp.Manager()
    setup_logging(manager)
    stop_event = mp.Event()
    runtime = TrackerRuntime(
        config, args.node_id, manager, stop_event, args.spool_dir
    )
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop_event.set)
    service = TrackerService(
        runtime.node_id,
        runtime.node_epoch,
        runtime.journal,
        tracker_config_fingerprint(config, args.node_id),
        runtime.camera_health,
        frozenset(args.allow_client),
    )
    runtime.start(service.publish)
    server = await start_server(
        args.bind,
        service,
        TrackerTls(
            Path(args.certificate).read_bytes(),
            Path(args.key).read_bytes(),
            Path(args.client_ca).read_bytes(),
        ),
    )
    try:
        while not stop_event.is_set():
            service.degraded = runtime.degraded
            await asyncio.sleep(0.5)
    finally:
        await server.stop(runtime.node_config.shutdown_drain)
        runtime.stop()
        manager.shutdown()


def main() -> None:
    """Run the isolated tracker wrapper."""
    asyncio.run(_run(_arguments()))
