"""Fast host integration for the external tracker boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from extension.tracker import runtime as tracker_runtime
from extension.tracker.runtime import (
    CameraTrackAdapter,
    MediaManifest,
    TrackerJournal,
    TrackerOperation,
)
from extension.tracker.transport import (
    SERVICE,
    TrackerHostIngest,
    TrackerMaintainer,
    TrackerService,
    start_server,
)
from grpc import aio

from frigate.application.events.types import EventStateEnum
from frigate.domain.camera import PTZMetrics
from frigate.domain.object_detection.base import LocalObjectDetector
from frigate.domain.track.norfair_tracker import NorfairTracker
from frigate.infrastructure.config import FrigateConfig
from frigate.timeline import TimelineProcessor
from frigate.util.image import SharedMemoryFrameManager

WORKSPACE = Path(__file__).resolve().parents[2]
FACE_VIDEO = WORKSPACE / "assets/fixtures/mock_videos/face-recognition/segments/01_P1E_S1_C1_5s-20s.mp4"
LPR_VIDEO = WORKSPACE / "assets/fixtures/mock_videos/car-number-plate-video/cam-in/pexels-casey-whalen-6571483 (1024p).mp4"
MODEL = WORKSPACE / "assets/models/yolov9-t-320.onnx"
LABELMAP = Path(__file__).resolve().parents[1] / "docker/main/rootfs/labelmap/coco-80.txt"


class _Frames:
    def __init__(self) -> None:
        self.frame: np.ndarray | None = None

    def get(self, _name: str, _shape: object) -> np.ndarray | None:
        return self.frame

    def close(self, _name: str) -> None:
        pass

    def delete(self, _name: str) -> None:
        pass

    def cleanup(self) -> None:
        pass


class _Publisher:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def publish(self, *_args: object, **_kwargs: object) -> None:
        pass

    def stop(self) -> None:
        pass


class _Ptz:
    autotracker_init: dict[str, object] = {}

    def autotrack_object(self, *_args: object) -> None:
        pass

    def end_object(self, *_args: object) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _config(tmp_path: Path) -> FrigateConfig:
    raw = yaml.safe_load((WORKSPACE / "deploy/config.yaml").read_text(encoding="utf-8"))
    raw["database"] = {"path": str(tmp_path / "tracker.db")}
    raw["recognition"] = {"runtime": "local"}
    raw["notifications"]["enabled"] = False
    raw["record"]["enabled"] = False
    raw["snapshots"]["enabled"] = False
    raw["model"]["path"] = str(MODEL)
    raw["model"]["labelmap_path"] = str(LABELMAP)
    raw["detectors"] = {
        "onnx": {
            "type": "onnx",
            "device": "CPU",
            "intra_op_num_threads": 2,
            "inter_op_num_threads": 1,
            "allow_spinning": False,
            "execution_mode": "sequential",
        }
    }
    raw["cameras"] = {
        camera: raw["cameras"][camera]
        for camera in ("face_camera", "car_camera")
    }
    raw["tracker"] = {
        "edge-local": {
            "endpoint": "127.0.0.1:50052",
            "cameras": ["face_camera", "car_camera"],
            "deadline": 5,
            "tls": {
                "ca": "test-ca",
                "certificate": "test-certificate",
                "key": "test-key",
                "server_name": "edge-local",
            },
        }
    }
    return FrigateConfig.parse_object(raw)


def _frames(video: Path, width: int, height: int, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "ffmpeg is required"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = width * height * 3 // 2
    try:
        index = 0
        while data := process.stdout.read(size):
            assert len(data) == size
            index += 1
            yield index, np.frombuffer(data, np.uint8).reshape((height * 3 // 2, width))
    finally:
        process.stdout.close()
        stderr = b"" if process.stderr is None else process.stderr.read()
        returncode = process.wait(timeout=5)
        assert returncode == 0, stderr.decode(errors="replace")


def _detections(detector: LocalObjectDetector, frame: np.ndarray, width: int, height: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
    tensor = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LINEAR)[None, ...]
    output = []
    for label, score, box in detector.detect(tensor):
        if label not in {"person", "car"}:
            continue
        top, left, bottom, right = box
        pixel_box = (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )
        area = max(0, pixel_box[2] - pixel_box[0]) * max(0, pixel_box[3] - pixel_box[1])
        if area == 0:
            continue
        output.append((label, score, pixel_box, area, (pixel_box[2] - pixel_box[0]) / max(1, pixel_box[3] - pixel_box[1]), (0, 0, width, height)))
    return output


def _track_video(
    config: FrigateConfig,
    detector: LocalObjectDetector,
    journal: TrackerJournal,
    camera: str,
    video: Path,
    fps: int,
) -> None:
    camera_config = config.cameras[camera]
    width, height = camera_config.detect.width, camera_config.detect.height
    ptz_metrics = PTZMetrics(autotracker_enabled=False)
    tracker = NorfairTracker(camera_config, ptz_metrics)
    adapter = CameraTrackAdapter(
        config,
        "edge-local",
        "node-epoch",
        camera,
        SimpleNamespace(ptz_autotracker=_Ptz()),
        journal.append,
    )
    frames = _Frames()
    adapter.frame_manager = frames  # type: ignore[assignment]
    adapter.state.frame_manager = frames  # type: ignore[assignment]
    try:
        last_index = 0
        for index, frame in _frames(video, width, height, fps):
            last_index = index
            frame_time = index / fps
            frame_name = f"{camera}-{index}"
            frames.frame = frame
            tracker.match_and_update(
                frame_name,
                frame_time,
                _detections(detector, frame, width, height),
            )
            objects = {
                object_id: {**value, "attributes": []}
                for object_id, value in tracker.tracked_objects.items()
            }
            adapter.process(frame_name, frame_time, objects, [], [])
        for offset in range(camera_config.detect.max_disappeared + 2):
            frame_time = (last_index + offset + 1) / fps
            frame_name = f"{camera}-flush-{offset}"
            tracker.match_and_update(frame_name, frame_time, [])
            objects = {
                object_id: {**value, "attributes": []}
                for object_id, value in tracker.tracked_objects.items()
            }
            adapter.process(frame_name, frame_time, objects, [], [])
    finally:
        adapter.close()


async def _grpc_roundtrip(journal: TrackerJournal) -> list:
    service = TrackerService(
        "edge-local", "node-epoch", journal, "integration", lambda: ()
    )
    port = _free_port()
    server = await start_server(f"127.0.0.1:{port}", service)
    channel = aio.insecure_channel(f"127.0.0.1:{port}")
    call = channel.stream_stream(
        f"/{SERVICE}/Connect",
        request_serializer=lambda value: value,
        response_deserializer=lambda value: value,
    )()
    committed = []
    ingest = TrackerHostIngest(
        {"face_camera": "edge-local", "car_camera": "edge-local"}, committed.append
    )
    try:
        await call.read()
        await call.write(
            b'{"type":"session_start","protocol_version":2,'
            b'"node_epoch":"node-epoch","ack_sequence":0}'
        )
        expected = journal.pending_count
        for _ in range(expected):
            message = json.loads(await asyncio.wait_for(call.read(), 5))
            update = tracker_runtime.TrackerUpdate.from_json(message["update"])
            ingest.accept(update)
            await call.write(
                json.dumps(
                    {
                        "type": "ack",
                        "ack_sequence": update.journal_sequence,
                    }
                ).encode()
            )
        deadline = time.monotonic() + 2
        while journal.pending_count and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert journal.pending_count == 0
        return committed
    finally:
        call.cancel()
        await channel.close()
        await server.stop(0)


async def _production_commit_roundtrip(
    config: FrigateConfig,
    journal: TrackerJournal,
    updates: list,
) -> None:
    """Run main commit, media RPC, SHM, receipt, and Timeline on Windows."""
    start = next(
        update
        for update in updates
        if update.camera_id == "face_camera"
        and update.operation is TrackerOperation.START
    )
    lifecycle = [update for update in updates if update.event_id == start.event_id]
    assert lifecycle[0].operation is TrackerOperation.START
    assert lifecycle[-1].operation is TrackerOperation.END

    shape = config.cameras[start.camera_id].frame_shape_yuv
    content = bytes([37]) * int(np.prod(shape))
    media_id = "a" * 32
    manifest = MediaManifest(
        media_id=media_id,
        event_id=start.event_id,
        camera_id=start.camera_id,
        media_type="recognition_frame",
        codec="i420",
        start_time=start.frame_time,
        end_time=start.frame_time,
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        expiry_unix_ms=int((time.time() + 60) * 1000),
    )
    lifecycle[0] = replace(start, media=(manifest,))

    def read_media(
        requested_media_id: str, offset: int, length: int | None
    ) -> bytes:
        assert requested_media_id == media_id
        end = None if length is None else offset + length
        return content[offset:end]

    service = TrackerService(
        "edge-local",
        "node-epoch",
        journal,
        "integration",
        lambda: (),
        media_reader=read_media,
    )
    port = _free_port()
    server = await start_server(f"127.0.0.1:{port}", service)
    channel = aio.insecure_channel(f"127.0.0.1:{port}")
    await channel.channel_ready()

    event_updates: queue.Queue = queue.Queue()
    event_commits: queue.Queue = queue.Queue()
    accepted = []
    maintainer = object.__new__(TrackerMaintainer)
    maintainer.config = config
    maintainer.event_update_queue = event_updates
    maintainer.event_commit_queue = event_commits
    maintainer.publisher = _Publisher()
    maintainer.frame_manager = SharedMemoryFrameManager()
    maintainer.store = SimpleNamespace(accept=accepted.append)
    maintainer.loop = asyncio.get_running_loop()
    maintainer.channels = {"edge-local": channel}
    consumer_frame_manager = SharedMemoryFrameManager()

    timeline = object.__new__(TimelineProcessor)
    timeline.config = config
    timeline.pre_event_cache = {}
    timeline.insert_or_save = lambda *_args, **_kwargs: None
    errors: list[BaseException] = []

    def consume() -> None:
        previous = None
        for index, expected in enumerate(lifecycle):
            receipt = ""
            try:
                update = event_updates.get(timeout=5)
                source_type, event_type, camera, frame_name, event_data, receipt = update
                assert camera == expected.camera_id
                assert source_type.value == "tracked_object"
                assert frame_name and "/" not in frame_name
                frame = consumer_frame_manager.get(frame_name, shape)
                if index == 0:
                    assert event_data["observed_in_frame"] is True
                    assert event_data["_recognition_evidence_owned"] is True
                    assert frame is not None
                    assert frame.tobytes() == content
                    consumer_frame_manager.delete(frame_name)
                else:
                    assert event_data["observed_in_frame"] is False
                    assert frame is None
                timeline.handle_object_detection(
                    camera,
                    event_type,
                    previous,
                    event_data,
                )
                previous = event_data
            except BaseException as error:
                errors.append(error)
                if receipt:
                    event_commits.put((receipt, False, str(error)))
                return
            event_commits.put((receipt, True, ""))

    consumer = threading.Thread(target=consume, name="tracker_commit_test")
    consumer.start()
    try:
        for update in lifecycle:
            await asyncio.to_thread(maintainer._commit, update)
        consumer.join(timeout=5)
        assert not consumer.is_alive()
        assert not errors, errors
        assert accepted == lifecycle
    finally:
        for frame_name in tuple(consumer_frame_manager.shm_store):
            consumer_frame_manager.delete(frame_name)
        for frame_name in tuple(maintainer.frame_manager.shm_store):
            maintainer.frame_manager.delete(frame_name)
        await channel.close()
        await server.stop(0)


@pytest.mark.integration
def test_mock_videos_produce_tracker_trace_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock videos must cross tracker transport and Frigate-main payload boundaries."""
    for asset in (FACE_VIDEO, LPR_VIDEO, MODEL, LABELMAP):
        assert asset.is_file(), asset
    monkeypatch.setattr(tracker_runtime, "DetectionPublisher", _Publisher)
    config = _config(tmp_path)
    journal = TrackerJournal(tmp_path / "journal.db")
    detector = LocalObjectDetector(
        config.detectors["onnx"], str(config.model.labelmap_path)
    )
    try:
        _track_video(config, detector, journal, "face_camera", FACE_VIDEO, 15)
        _track_video(config, detector, journal, "car_camera", LPR_VIDEO, 5)
        updates = asyncio.run(_grpc_roundtrip(journal))
        maintainer = object.__new__(TrackerMaintainer)
        timeline = object.__new__(TimelineProcessor)
        timeline.config = config
        timeline.pre_event_cache = {}
        timeline.insert_or_save = lambda *_args, **_kwargs: None
        frame_manager = SharedMemoryFrameManager()
        for update in updates:
            event_data = maintainer._event_data(update)
            region = event_data["region"]
            assert isinstance(region, list) and len(region) == 4, event_data
            timeline.handle_object_detection(
                update.camera_id,
                EventStateEnum.start,
                None,
                event_data,
            )
            frame_name = maintainer._unavailable_frame_name(update)
            assert frame_name and "/" not in frame_name
            assert (
                frame_manager.get(
                    frame_name,
                    config.cameras[update.camera_id].frame_shape_yuv,
                )
                is None
            )
        legacy = tracker_runtime.TrackerUpdate.from_json(updates[0].to_json())
        legacy.state["region"] = {"boxes": [[0, 0, 320, 320]]}
        legacy = replace(legacy, region={"boxes": ((0, 0, 320, 320),)})
        legacy_event_data = maintainer._event_data(legacy)
        assert legacy_event_data["region"] == [0, 0, 320, 320]
        timeline.handle_object_detection(
            legacy.camera_id,
            EventStateEnum.start,
            None,
            legacy_event_data,
        )
        asyncio.run(_production_commit_roundtrip(config, journal, updates))
        starts = {
            camera: {
                update.trace_id
                for update in updates
                if update.camera_id == camera and update.operation is TrackerOperation.START
            }
            for camera in ("face_camera", "car_camera")
        }
        assert len(starts["face_camera"]) == 4, starts
        assert 8 <= len(starts["car_camera"]) <= 11, starts
    finally:
        journal.close()
