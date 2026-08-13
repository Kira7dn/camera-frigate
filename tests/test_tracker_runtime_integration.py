"""Behavioral integration test for the thin tracker data plane."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from extension.tracker.runtime import (
    BoundingBox,
    TrackerJournal,
    TrackerOperation,
    TrackerUpdate,
)
from extension.tracker.transport import SERVICE, TrackerService, start_server
from grpc import aio

from frigate.domain.detectors.detector_config import ModelConfig
from frigate.domain.object_detection.base import RemoteObjectDetector
from frigate.infrastructure.comms import object_detector_signaler
from frigate.infrastructure.comms.object_detector_signaler import (
    DetectorProxy,
    ObjectDetectorPublisher,
)
from frigate.util.image import UntrackedSharedMemory


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _native_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, float, tuple[float, float, float, float]]:
    """Run a real RemoteObjectDetector SHM/queue/ZMQ completion cycle."""
    monkeypatch.setattr(
        object_detector_signaler,
        "SOCKET_PUB",
        f"tcp://127.0.0.1:{_free_port()}",
    )
    monkeypatch.setattr(
        object_detector_signaler,
        "SOCKET_SUB",
        f"tcp://127.0.0.1:{_free_port()}",
    )
    name = f"tracker-integration-{uuid.uuid4().hex}"
    input_shm = UntrackedSharedMemory(name=name, create=True, size=12)
    output_shm = UntrackedSharedMemory(name=f"out-{name}", create=True, size=20 * 6 * 4)
    output = np.ndarray((20, 6), dtype=np.float32, buffer=output_shm.buf)
    output[:] = 0
    detection_queue: mp.Queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()
    proxy = DetectorProxy()
    publisher = ObjectDetectorPublisher()
    detector = RemoteObjectDetector(
        name,
        {0: "person"},
        detection_queue,
        ModelConfig(width=2, height=2),
        stop_event,
    )

    def complete_detection() -> None:
        connection_id = detection_queue.get(timeout=2)
        assert connection_id == name
        output[:] = 0
        output[0] = (0, 0.9, 0.1, 0.2, 0.8, 0.9)
        publisher.publish(connection_id)

    worker = threading.Thread(target=complete_detection)
    try:
        # Allow PUB/SUB subscriptions to propagate before the single request.
        time.sleep(0.1)
        worker.start()
        result = detector.detect(np.ones((1, 2, 2, 3), dtype=np.uint8))
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(result) == 1
        return result[0]
    finally:
        stop_event.set()
        detector.cleanup()
        publisher.stop()
        proxy.stop()
        input_shm.close()
        output_shm.close()
        detection_queue.close()


async def _round_trip(tmp_path: Path, detection: tuple) -> None:
    journal = TrackerJournal(tmp_path / "journal.db")
    service = TrackerService(
        "edge-local",
        "node-epoch",
        journal,
        "config-hash",
        lambda: (
            {
                "camera_id": "face_camera",
                "ready": True,
                "camera_fps": 5.0,
                "process_fps": 5.0,
                "capture_pid": 10,
                "process_pid": 11,
            },
        ),
    )
    port = _free_port()
    server = await start_server(f"127.0.0.1:{port}", service)
    channel = aio.insecure_channel(f"127.0.0.1:{port}")
    connect = channel.stream_stream(
        f"/{SERVICE}/Connect",
        request_serializer=lambda value: value,
        response_deserializer=lambda value: value,
    )
    call = connect()
    try:
        hello = json.loads(await asyncio.wait_for(call.read(), 2))
        assert hello["node_id"] == "edge-local"
        await call.write(b'{"type":"hello","replay_after_sequence":0}')
        label, score, normalized_box = detection
        left, top, right, bottom = (
            int(value * 100) for value in normalized_box
        )
        trace_id = uuid.uuid4().hex[:30]
        persisted = service.publish(
            TrackerUpdate(
                "edge-local",
                "node-epoch",
                "face_camera",
                "stream-epoch",
                0,
                1,
                1_000_000,
                1.0,
                trace_id,
                "native-track-1",
                TrackerOperation.START,
                label,
                (score,),
                score,
                BoundingBox(left, top, right, bottom),
            )
        )
        envelope = json.loads(await asyncio.wait_for(call.read(), 2))
        received = TrackerUpdate.from_json(envelope["update"])
        assert received == persisted
        assert received.trace_id == trace_id
        assert received.label == "person"
        await call.write(
            json.dumps(
                {
                    "type": "ack",
                    "node_epoch": received.node_epoch,
                    "journal_sequence": received.journal_sequence,
                    "event_id": received.event_id,
                }
            ).encode()
        )
        deadline = time.monotonic() + 2
        while journal.pending_count and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert journal.pending_count == 0
    finally:
        await call.done_writing()
        await channel.close()
        await server.stop(0)
        journal.close()


def test_detector_to_tracker_journal_grpc_trace_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detector completion must become a durable streamed producer trace."""
    detection = _native_detection(monkeypatch)
    asyncio.run(_round_trip(tmp_path, detection))
