"""Compose native Frigate camera components for an externally owned lane."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

from peewee import SqliteDatabase

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
from frigate.domain.track.tracked_object import TrackedObject
from frigate.infrastructure.comms.object_detector_signaler import DetectorProxy
from frigate.infrastructure.comms.zmq_proxy import ZmqProxy
from frigate.infrastructure.config import FrigateConfig
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
class MediaManifest:
    """Describe edge-owned media without transferring bytes through events."""

    media_id: str
    event_id: str
    camera_id: str
    media_type: str
    codec: str
    start_time: float
    end_time: float
    byte_size: int
    sha256: str
    expiry_unix_ms: int


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
    media: tuple[MediaManifest, ...] = ()

    @property
    def trace_id(self) -> str:
        """Return the producer-owned downstream correlation id."""
        return self.event_id

    def to_json(self) -> str:
        # Avoid dataclasses.asdict(): native tracker state may contain defaultdicts.
        value = {item.name: getattr(self, item.name) for item in fields(self)}
        value["operation"] = self.operation.value
        value["bbox"] = {item.name: getattr(self.bbox, item.name) for item in fields(self.bbox)}
        value["media"] = [
            {item.name: getattr(manifest, item.name) for item in fields(manifest)}
            for manifest in self.media
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> TrackerUpdate:
        data = json.loads(value)
        data["operation"] = TrackerOperation(data["operation"])
        data["bbox"] = BoundingBox(**data["bbox"])
        data["media"] = tuple(
            MediaManifest(**manifest) for manifest in data.get("media", ())
        )
        for key in ("score_history", "current_zones", "entered_zones"):
            data[key] = tuple(data.get(key, ()))
        data["path"] = tuple(tuple(item) for item in data.get("path", ()))
        return cls(**data)


class EdgeMediaStore:
    """Materialize and serve tracker-owned evidence outside Frigate main."""

    def __init__(self, root: str | Path, config: FrigateConfig) -> None:
        self.root = Path(root) / "edge-media"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._paths: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tracker_media")
        self._futures: set[Future[MediaManifest | None]] = set()
        self._completed_events: set[str] = set()

    def _register(
        self,
        path: Path,
        update: TrackerUpdate,
        media_type: str,
        codec: str,
        start_time: float,
        end_time: float,
    ) -> MediaManifest:
        media_id = uuid.uuid4().hex
        content = path.read_bytes()
        manifest = MediaManifest(
            media_id=media_id,
            event_id=update.event_id,
            camera_id=update.camera_id,
            media_type=media_type,
            codec=codec,
            start_time=start_time,
            end_time=end_time,
            byte_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            expiry_unix_ms=int((time.time() + 3600) * 1000),
        )
        with self._lock:
            self._paths[media_id] = path
        return manifest

    def snapshot(self, update: TrackerUpdate, obj: TrackedObject) -> MediaManifest | None:
        if not should_save_snapshot(self.config, update.camera_id, obj):
            return None
        image, frame_time = obj.get_img_bytes(
            ext="jpg",
            timestamp=True,
            bounding_box=True,
            quality=self.config.cameras[update.camera_id].snapshots.quality,
        )
        if image is None:
            return None
        path = self.root / "snapshots" / f"{uuid.uuid4().hex}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image)
        timestamp = update.frame_time if frame_time is None else frame_time
        return self._register(path, update, "snapshot", "jpeg", timestamp, timestamp)

    def recognition_frame(
        self, update: TrackerUpdate, frame: Any
    ) -> MediaManifest:
        """Stage one exact I420 frame for recognition in Frigate main."""
        path = self.root / "recognition" / f"{uuid.uuid4().hex}.i420"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frame.tobytes())
        return self._register(
            path,
            update,
            "recognition_frame",
            "i420",
            update.frame_time,
            update.frame_time,
        )

    def publish_end(
        self,
        update: TrackerUpdate,
        obj: TrackedObject,
        publish: Callable[[TrackerUpdate], TrackerUpdate],
        source_epoch: float | None = None,
    ) -> None:
        manifests = tuple(
            manifest for manifest in (self.snapshot(update, obj),) if manifest is not None
        )
        start_time = float(obj.obj_data.get("start_time", update.frame_time))
        clip_path = self.root / "clips" / update.event_id / "clip.mp4"
        trace_path = self.root / "traces" / update.event_id / "trace.json"

        trace = {
            "event_id": update.event_id,
            "camera_id": update.camera_id,
            "track_id": update.track_id,
            "node_id": update.node_id,
            "node_epoch": update.node_epoch,
            "start_time": start_time,
            "end_time": update.frame_time,
            "source_epoch": source_epoch,
        }

        def materialize_trace() -> MediaManifest:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_payload = json.dumps(
                trace, sort_keys=True, separators=(",", ":")
            )
            trace_path.write_text(trace_payload, encoding="utf-8")
            return self._register(
                trace_path,
                update,
                "trace",
                "json",
                start_time,
                update.frame_time,
            )

        def materialize_direct_clip() -> MediaManifest | None:
            source: Path | None = None
            for ffmpeg_input in self.config.cameras[update.camera_id].ffmpeg.inputs:
                if "detect" in ffmpeg_input.roles:
                    candidate = Path(str(ffmpeg_input.path))
                    if candidate.is_file():
                        source = candidate
                        break
            if source is None:
                return None

            start_offset = (
                max(0.0, start_time - source_epoch)
                if source_epoch is not None
                else 0.0
            )
            end_offset = (
                max(start_offset + 1.0, update.frame_time - source_epoch)
                if source_epoch is not None
                else None
            )
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep .mp4 as the final suffix so ffmpeg selects the MP4 muxer.
            # The old clip.mp4.tmp name forced the full-source fallback.
            temporary = clip_path.with_name(f"{clip_path.stem}.tmp{clip_path.suffix}")
            command = [
                self.config.ffmpeg.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_offset:.3f}",
                "-i",
                str(source),
            ]
            if end_offset is not None:
                command.extend(["-t", f"{end_offset - start_offset:.3f}"])
            command.extend(
                [
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                    str(temporary),
                ]
            )
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError:
                completed = None

            if completed is None or completed.returncode != 0 or not temporary.is_file():
                # The tracker image may not expose ffmpeg in PATH. Preserve the
                # producer-owned media contract by copying the mounted replay
                # source rather than dropping the clip altogether.
                temporary.unlink(missing_ok=True)
                try:
                    shutil.copyfile(source, temporary)
                except OSError:
                    temporary.unlink(missing_ok=True)
                    return None
            os.replace(temporary, clip_path)
            return self._register(
                clip_path,
                update,
                "clip",
                "h264",
                start_time,
                update.frame_time,
            )

        def materialize() -> MediaManifest | None:
            return materialize_direct_clip()

        future = self._executor.submit(materialize)
        with self._lock:
            self._futures.add(future)

        def complete(result: Future[MediaManifest | None]) -> None:
            try:
                clip = result.result()
                media = manifests + (materialize_trace(),)
                if clip is not None:
                    media += (clip,)
                publish(replace(update, media=media))
                with self._lock:
                    self._completed_events.add(update.event_id)
            except (OSError, RuntimeError, ValueError):
                logger.exception("Tracker media materialization failed event_id=%s", update.event_id)
                publish(replace(update, media=manifests))
            finally:
                with self._lock:
                    self._futures.discard(result)

        future.add_done_callback(complete)

    def read(self, media_id: str, offset: int, length: int | None) -> bytes:
        if not media_id or any(character not in "0123456789abcdef" for character in media_id):
            raise ValueError("invalid_media_id")
        with self._lock:
            path = self._paths.get(media_id)
        if path is None:
            raise FileNotFoundError(media_id)
        size = path.stat().st_size
        if offset < 0 or offset > size or (length is not None and length < 0):
            raise ValueError("invalid_media_range")
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read() if length is None else handle.read(length)

    def pending_count(self) -> int:
        """Return media jobs whose publish callback has not completed."""
        with self._lock:
            return len(self._futures)

    def completed_event_ids(self) -> tuple[str, ...]:
        """Return events whose terminal media update was published."""
        with self._lock:
            return tuple(sorted(self._completed_events))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


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
    """Publish the native video payload used by Frigate main consumers."""
    publisher.publish(
        (camera, frame_name, frame_time, objects, motion, regions),
        "video",
    )


class TrackerJournal:
    """Persist one ordered outbox per tracker process epoch."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.execute(
            """create table if not exists updates_v2(
                sequence integer primary key autoincrement,
                epoch_sequence integer not null,
                node_epoch text not null,
                event_id text not null,
                payload text not null,
                acknowledged integer not null default 0,
                unique(node_epoch, epoch_sequence)
            )"""
        )
        self._db.commit()

    def append(self, update: TrackerUpdate) -> TrackerUpdate:
        with self._lock:
            cursor = self._db.execute(
                "select coalesce(max(epoch_sequence), 0) + 1 from updates_v2 where node_epoch=?",
                (update.node_epoch,),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("tracker journal did not allocate a sequence")
            sequence = int(row[0])
            value = TrackerUpdate.from_json(update.to_json())
            object.__setattr__(value, "journal_sequence", sequence)
            self._db.execute(
                "insert into updates_v2(epoch_sequence,node_epoch,event_id,payload) values(?,?,?,?)",
                (sequence, update.node_epoch, update.event_id, value.to_json()),
            )
            self._db.commit()
            return value

    def next_pending(self, node_epoch: str, after: int = 0) -> TrackerUpdate | None:
        with self._lock:
            row = self._db.execute(
                "select payload from updates_v2 where node_epoch=? and epoch_sequence>? and acknowledged=0 order by epoch_sequence limit 1",
                (node_epoch, after),
            ).fetchone()
        return None if row is None else TrackerUpdate.from_json(row[0])

    def acknowledge_through(self, node_epoch: str, sequence: int) -> int:
        with self._lock:
            cursor = self._db.execute(
                """update updates_v2 set acknowledged=1
                   where node_epoch=? and epoch_sequence<=? and acknowledged=0""",
                (node_epoch, sequence),
            )
            self._db.commit()
            return cursor.rowcount

    @property
    def pending_count(self) -> int:
        """Return unacknowledged updates across every retained epoch."""
        with self._lock:
            row = self._db.execute(
                "select count(*) from updates_v2 where acknowledged=0"
            ).fetchone()
        return int(row[0])

    def pending_count_for_epoch(self, node_epoch: str) -> int:
        """Return unacknowledged updates that can be sent by this runtime epoch."""
        with self._lock:
            row = self._db.execute(
                "select count(*) from updates_v2 where node_epoch=? and acknowledged=0",
                (node_epoch,),
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


def tracker_config_fingerprint(config: FrigateConfig, node_id: str) -> str:
    """Return the compiler-owned revision shared by every runtime view."""
    if node_id not in config.tracker:
        raise ValueError(f"unknown tracker node: {node_id}")
    revision = config.runtime.topology_revision
    if not revision:
        raise ValueError("compiled tracker topology revision is required")
    return revision


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
        media: EdgeMediaStore | None = None,
    ) -> None:
        self.config = config
        self.node_id = node_id
        self.node_epoch = node_epoch
        self.camera = camera
        self.stream_epoch = uuid.uuid4().hex
        self.publish = publish
        self.media = media
        self.frame_manager = SharedMemoryFrameManager()
        self.frame_seq = 0
        self.frame_name = ""
        self.source_epoch: float | None = None
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
        if self.source_epoch is None:
            self.source_epoch = frame_time
        self.frame_name = frame_name
        self.motion = motion
        self.regions = regions
        self.state.update(frame_name, frame_time, objects, motion, regions)

    def _start(self, camera: str, obj: TrackedObject, *_: object) -> None:
        track_id = str(obj.obj_data["id"])
        self.event_ids[track_id] = uuid.uuid4().hex[:30]
        self._emit(TrackerOperation.START, obj)

    def _update(self, camera: str, obj: TrackedObject, *_: object) -> None:
        apply_media_policy(self.config, camera, obj)
        self._emit(TrackerOperation.UPDATE, obj)

    def _end(self, camera: str, obj: TrackedObject, *_: object) -> None:
        apply_media_policy(self.config, camera, obj)
        update = self._update_value(TrackerOperation.END, obj)
        if self.media is None:
            self.publish(update)
        else:
            self.media.publish_end(update, obj, self.publish, self.source_epoch)
        self.event_ids.pop(str(obj.obj_data["id"]), None)
        if not obj.false_positive:
            self.ptz.end_object(camera, obj)

    def _autotrack(self, camera: str, obj: TrackedObject, *_: object) -> None:
        self.ptz.autotrack_object(camera, obj)

    def _emit(self, operation: TrackerOperation, obj: TrackedObject) -> None:
        self.publish(self._update_value(operation, obj))

    def _update_value(
        self, operation: TrackerOperation, obj: TrackedObject
    ) -> TrackerUpdate:
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
                "frame_name": self.frame_name,
                "motion_boxes": tuple(self.motion),
                "detection_regions": tuple(self.regions),
            }
        )
        update = TrackerUpdate(
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
            {"boxes": tuple(self.motion)},
            {"boxes": tuple(self.regions)},
        )
        camera_config = self.config.cameras[self.camera]
        needs_recognition = (
            operation is not TrackerOperation.END
            and (
                (update.label == "person" and camera_config.face_recognition.enabled)
                or (update.label in ("car", "motorcycle") and camera_config.lpr.enabled)
            )
        )
        if self.media is None or not needs_recognition:
            return update
        frame = self.frame_manager.get(
            self.frame_name, camera_config.frame_shape_yuv
        )
        if frame is None:
            return update
        manifest = self.media.recognition_frame(update, frame)
        return replace(update, media=(manifest,))

    def finalize(self) -> None:
        """End tracks that remain active when a finite source reaches EOF."""
        for track_id, obj in tuple(self.state.tracked_objects.items()):
            if str(track_id) in self.event_ids:
                self._end(self.camera, obj)

    def close(self) -> None:
        self.finalize()


class TrackerRuntime:
    """Own only the composition of existing Frigate runtime components."""

    def __init__(
        self,
        config: FrigateConfig,
        node_id: str,
        manager: SyncManager,
        stop_event: MpEvent,
        spool_dir: str | Path,
        media_dir: str | Path = "/media/tracker",
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
        # EdgeMediaStore creates bounded per-track clips directly from the
        # camera source. Continuous recording has no consumer in this service.
        for camera_config in config.cameras.values():
            camera_config.record.enabled = False
            camera_config.recreate_ffmpeg_cmds()
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
        self.media = EdgeMediaStore(media_dir, config)
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
        self.adapters: dict[str, CameraTrackAdapter] = {}
        self.finalized_sources: set[str] = set()
        self.source_idle_polls = {camera: 0 for camera in config.cameras}
        self.session_complete_written = False
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
        for camera in self.config.cameras:
            self.adapters[camera] = CameraTrackAdapter(
                self.config,
                self.node_id,
                self.node_epoch,
                camera,
                self.ptz,
                publish,
                self.media,
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
                self._finalize_ended_sources()
                continue
            self.source_idle_polls[camera] = 0
            try:
                self.adapters[camera].process(
                    frame_name, frame_time, objects, motion, regions
                )
            except (KeyError, RuntimeError, ValueError):
                logger.exception("Tracker frame processing failed camera=%s", camera)
                self.degraded = True

    def _finalize_ended_sources(self) -> None:
        """Close finite-source tracks and publish one explicit completion marker."""
        marker_root = os.environ.get("PASSAGE_SOURCE_START_DIR")
        if not marker_root:
            return
        root = Path(marker_root)
        for camera, adapter in self.adapters.items():
            if camera in self.finalized_sources:
                continue
            if not (root / f"{camera}.end").is_file():
                self.source_idle_polls[camera] = 0
                continue
            self.source_idle_polls[camera] += 1
            if self.source_idle_polls[camera] < 2:
                continue
            adapter.finalize()
            self.finalized_sources.add(camera)

        if (
            not self.session_complete_written
            and self.finalized_sources == set(self.adapters)
            and self.media.pending_count() == 0
        ):
            marker = root / "tracker-session-complete.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "node_id": self.node_id,
                        "node_epoch": self.node_epoch,
                        "cameras": sorted(self.finalized_sources),
                        "events": list(self.media.completed_event_ids()),
                        "completed_at": time.time(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, marker)
            self.session_complete_written = True

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

    def active_lifecycle_count(self) -> int:
        """Return tracks still owned by this tracker runtime."""
        return sum(len(adapter.event_ids) for adapter in self.adapters.values())

    def stop(self) -> None:
        """Stop native components and release tracker-owned IPC resources."""
        self.stop_event.set()
        timeout = self.node_config.shutdown_drain
        self.cameras.join(timeout=timeout)
        self.consumer.join(timeout=timeout)
        self.ptz.join(timeout=timeout)
        self.onvif.close()
        for adapter in self.adapters.values():
            adapter.close()
        self.media.close()
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
        config, args.node_id, manager, stop_event, args.spool_dir, args.media_dir
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
        runtime.media.read,
        runtime.active_lifecycle_count,
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
