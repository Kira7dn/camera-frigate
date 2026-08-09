from __future__ import annotations

import threading
import time
from collections import deque
from types import MethodType, SimpleNamespace

import numpy as np

from frigate.data_processing.common.evidence import (
    EvidenceBufferPolicy,
    EvidenceCandidate,
    EvidenceRingBuffer,
)
from frigate.data_processing.common.license_plate.pipeline import (
    LatestLprTaskQueue,
    LprExpireTask,
    LprFrameTask,
    LprTrackKey,
    PlateObservation,
)
from frigate.data_processing.common.quality import QualitySelector
from frigate.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)

TASK_RING = EvidenceRingBuffer(
    {"cam": EvidenceBufferPolicy(10.0, 8 * 1024 * 1024, 100.0)}
)


def frame_ref(track: str, frame_time: float):
    ref = TASK_RING.ingest(
        "cam", "detect", f"{track}-{frame_time}", frame_time, np.zeros((6, 4), np.uint8)
    )
    assert ref is not None
    return ref


def frame_task(track: str, frame_time: float) -> LprFrameTask:
    return LprFrameTask(
        key=LprTrackKey("cam", track, 0),
        obj_data={"id": track, "camera": "cam", "frame_time": frame_time},
        frame_ref=frame_ref(track, frame_time),
        dedicated_lpr=False,
        frame_time=frame_time,
    )


def test_same_track_replaces_older_frame_and_queue_is_bounded() -> None:
    tasks = LatestLprTaskQueue(max_tracks=2)
    assert tasks.submit(frame_task("one", 1.0))
    assert tasks.submit(frame_task("one", 2.0))
    assert tasks.replaced == 1
    assert tasks.depth == 1
    assert tasks.submit(frame_task("two", 1.0))
    assert not tasks.submit(frame_task("three", 1.0))
    assert tasks.depth == 2
    assert tasks.full_drops == 1
    item = tasks.get()
    assert isinstance(item, LprFrameTask)
    assert item.frame_time == 2.0


def test_expire_is_priority_and_invalidates_pending_generation() -> None:
    tasks = LatestLprTaskQueue(max_tracks=2)
    old = frame_task("one", 1.0)
    assert tasks.submit(old)
    assert tasks.advance_generation("cam", "one") == 1
    item = tasks.get()
    assert isinstance(item, LprExpireTask)
    assert not tasks.is_current(old.key)
    assert tasks.depth == 0


def test_expire_controls_remain_bounded_without_losing_invalidations() -> None:
    tasks = LatestLprTaskQueue(max_tracks=2)
    old_keys = []
    for index in range(10):
        old_keys.append(LprTrackKey("cam", f"track-{index}", 0))
        tasks.advance_generation("cam", f"track-{index}")
    assert tasks.control_depth <= 2
    assert all(not tasks.is_current(key) for key in old_keys)


def make_worker(infer) -> LicensePlateRealTimeProcessor:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.metrics = SimpleNamespace()
    processor._tasks = LatestLprTaskQueue(8)
    processor._states = {}
    processor._results = deque()
    processor._results_lock = threading.Lock()
    processor._stop_event = threading.Event()
    processor.evidence_ring = EvidenceRingBuffer(
        {"cam": EvidenceBufferPolicy(10.0, 8 * 1024 * 1024, 100.0)}
    )
    processor.quality_selector = QualitySelector(processor.evidence_ring)
    processor._is_eligible = MethodType(
        lambda _self, obj_data, dedicated: (
            "cam",
            str(obj_data["id"]),
            float(obj_data["frame_time"]),
        ),
        processor,
    )
    processor.lpr_process = MethodType(infer, processor)
    processor._reduce = MethodType(lambda _self, value: value, processor)
    processor._worker = threading.Thread(target=processor._worker_loop, daemon=True)
    processor._worker.start()
    return processor


def test_slow_inference_does_not_block_process_frame() -> None:
    finished = threading.Event()

    def slow_infer(_self, obj_data, frame, dedicated, key, frame_ref):
        time.sleep(0.2)
        finished.set()

    processor = make_worker(slow_infer)
    started = time.monotonic()
    processor.process_frame(
        {"id": "one", "camera": "cam", "frame_time": 1.0},
        processor.evidence_ring.ingest(
            "cam", "detect", "one", 1.0, np.zeros((6, 4), dtype=np.uint8)
        ),
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.05
    assert finished.wait(1.0)
    processor.shutdown()
    assert processor.pending_count == 0


def test_expire_blocks_stale_inflight_result() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_infer(_self, obj_data, frame, dedicated, key, frame_ref):
        started.set()
        assert release.wait(1.0)
        lease = _self.evidence_ring.acquire(frame_ref)
        assert lease is not None
        return PlateObservation(
            key=key,
            frame_time=1.0,
            plate="ABC1234",
            char_confidences=(0.9,) * 7,
            text_area=100,
            plate_box=(0, 0, 2, 2),
            object_box=(0, 0, 4, 4),
            obj_data=obj_data,
            dedicated_lpr=False,
            evidence=EvidenceCandidate(
                "candidate",
                "lpr",
                "cam",
                "one",
                key.generation,
                frame_ref,
                (0, 0, 4, 4),
                (0, 0, 2, 2),
                1.0,
                {"dimensions": 1.0},
                (),
                (),
                frame_ref.source_role,
                lease,
            ),
        )

    processor = make_worker(blocked_infer)
    processor.process_frame(
        {"id": "one", "camera": "cam", "frame_time": 1.0},
        processor.evidence_ring.ingest(
            "cam", "detect", "one", 1.0, np.zeros((6, 4), dtype=np.uint8)
        ),
    )
    assert started.wait(1.0)
    processor.expire_object("one", "cam")
    release.set()
    time.sleep(0.05)
    assert processor.drain_results() == []
    processor.shutdown()
