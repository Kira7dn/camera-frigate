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
    PlateTrackState,
    PreparedPlateCandidate,
)
from frigate.data_processing.common.quality import QualitySelector
from frigate.data_processing.common.recognition import RecognitionLifecycle
from frigate.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)
from frigate.embeddings.maintainer import EmbeddingMaintainer

TASK_RING = EvidenceRingBuffer(
    {"cam": EvidenceBufferPolicy(10.0, 8 * 1024 * 1024, 100.0)}
)


def frame_ref(track: str, frame_time: float):
    ref = TASK_RING.ingest(
        "cam", "detect", f"{track}-{frame_time}", frame_time, np.zeros((6, 4), np.uint8)
    )
    assert ref is not None
    return ref


def frame_task(track: str, frame_time: float, priority: float = 0.0) -> LprFrameTask:
    return LprFrameTask(
        key=LprTrackKey("cam", track, 0),
        obj_data={"id": track, "camera": "cam", "frame_time": frame_time},
        frame_ref=frame_ref(track, frame_time),
        dedicated_lpr=False,
        frame_time=frame_time,
        priority=priority,
    )


def test_same_track_replaces_older_frame_and_queue_is_bounded() -> None:
    tasks = LatestLprTaskQueue(max_tracks=2)
    assert tasks.submit(frame_task("one", 1.0))
    assert tasks.submit(frame_task("one", 2.0))
    assert tasks.replaced == 1
    assert tasks.depth == 1
    assert tasks.submit(frame_task("two", 2.1))
    assert not tasks.submit(frame_task("three", 2.2))
    assert tasks.depth == 2
    assert tasks.full_drops == 1
    item = tasks.get()
    assert isinstance(item, LprFrameTask)
    assert item.frame_time == 2.0


def test_higher_quality_roi_replaces_and_runs_before_low_priority_work() -> None:
    tasks = LatestLprTaskQueue(max_tracks=2)
    assert tasks.submit(frame_task("low", 1.0, 1.0))
    assert tasks.submit(frame_task("medium", 1.1, 2.0))
    assert tasks.submit(frame_task("high", 1.2, 3.0))
    assert tasks.replaced == 1
    first = tasks.get()
    second = tasks.get()
    assert isinstance(first, LprFrameTask)
    assert isinstance(second, LprFrameTask)
    assert [first.key.track_id, second.key.track_id] == ["high", "medium"]


def test_scheduler_waits_for_collection_window_then_selects_best_quality() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    key = LprTrackKey("cam", "passage", 0)
    low = SimpleNamespace(
        key=key,
        prepared_monotonic=1.0,
        evidence=SimpleNamespace(quality_score=0.4, candidate_id="low"),
    )
    high = SimpleNamespace(
        key=key,
        prepared_monotonic=2.0,
        evidence=SimpleNamespace(quality_score=0.9, candidate_id="high"),
    )
    processor._prepared = {key: [low, high]}
    processor._collection_started = {key: time.monotonic()}
    processor.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(
                recognition_lifecycle=SimpleNamespace(
                    candidate_collection_seconds=0.4
                )
            )
        }
    )
    selected = []
    processor._recognize_prepared = MethodType(
        lambda _self, candidate: selected.append(candidate.evidence.candidate_id),
        processor,
    )

    processor._run_ready_candidate()
    assert selected == []
    processor._collection_started[key] -= 0.4
    processor._run_ready_candidate()
    assert selected == ["high"]


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
    processor.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(
                recognition_lifecycle=SimpleNamespace(
                    candidate_collection_seconds=0.0
                )
            )
        }
    )
    processor._tasks = LatestLprTaskQueue(8)
    processor._states = {}
    processor._prepared = {}
    processor._collection_started = {}
    processor._terminal_keys = set()
    processor.recognition_lifecycle = RecognitionLifecycle()
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
        return PreparedPlateCandidate(
            key=key,
            frame_time=1.0,
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
            plate_frame=np.zeros((10, 20, 3), dtype=np.uint8),
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


def test_shutdown_releases_valid_state_without_best_effort_publish() -> None:
    processor = make_worker(lambda *_args: None)
    released = []
    key = LprTrackKey("cam", "one", 0)
    state = PlateTrackState(key)
    state.variants.append(
        {
            "plate": "ABC1234",
            "conf": 0.99,
            "observation": SimpleNamespace(
                evidence=SimpleNamespace(release=lambda: released.append(True))
            ),
        }
    )
    processor._states[key] = state
    processor.shutdown()
    assert released == [True]
    assert processor.drain_results() == []


def test_tracked_lpr_uses_conflated_detection_frames() -> None:
    submitted = []
    lpr = object.__new__(LicensePlateRealTimeProcessor)
    lpr.lp_objects = ["car", "motorcycle"]
    lpr._active_detection_ids = {}
    lpr.expire_missing_objects = MethodType(lambda *_args: None, lpr)
    lpr.associate_frame_objects = MethodType(
        lambda _self, _camera, objects: (
            [
                SimpleNamespace(passage_id=str(obj["id"]), obj_data=obj)
                for obj in objects
            ],
            [],
        ),
        lpr,
    )
    lpr.process_frame = MethodType(
        lambda _self, obj, frame_ref: submitted.append((obj["id"], frame_ref)),
        lpr,
    )
    maintainer = object.__new__(EmbeddingMaintainer)
    maintainer.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(
                type="generic",
                objects=SimpleNamespace(track=["car"]),
                lpr=SimpleNamespace(enabled=True),
                face_recognition=SimpleNamespace(enabled=False),
                frame_shape_yuv=(6, 4),
            )
        },
        classification=SimpleNamespace(custom={}),
    )
    maintainer.realtime_processors = [lpr]
    maintainer.frame_manager = SimpleNamespace(
        get=lambda *_args: np.zeros((6, 4), dtype=np.uint8),
        close=lambda *_args: None,
    )
    maintainer.evidence_ring = SimpleNamespace(
        ingest=lambda *_args: "frame-ref",
        last_reject_reason=lambda *_args: None,
    )
    maintainer.quality_selector = SimpleNamespace(record_reject=lambda *_args: None)

    maintainer._process_latest_frame(
        (
            "cam",
            "frame",
            1.0,
            [
                {"id": "small", "label": "car", "box": [0, 0, 2, 2], "area": 4},
                {"id": "large", "label": "car", "box": [0, 0, 4, 4], "area": 16},
                {"id": "person", "label": "person", "box": [0, 0, 4, 4], "area": 16},
            ],
            [],
            None,
        )
    )

    assert submitted == [("small", "frame-ref"), ("large", "frame-ref")]


def test_empty_event_metadata_poll_does_not_crash_maintainer() -> None:
    maintainer = object.__new__(EmbeddingMaintainer)
    maintainer.event_metadata_subscriber = SimpleNamespace(
        check_for_update=lambda: (None, None)
    )
    maintainer.post_processors = []

    maintainer._process_event_metadata()


def test_lpr_detection_reconciliation_expires_missing_track_once() -> None:
    expired = []
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor._active_detection_ids = {}
    processor._tasks = SimpleNamespace(
        advance_generation=lambda camera, track_id: expired.append(
            (camera, track_id)
        )
    )
    processor._update_queue_metrics = lambda: None

    processor.expire_missing_objects("cam", {"one", "two"})
    processor.expire_missing_objects("cam", {"two"})
    processor.expire_missing_objects("cam", {"two"})

    assert expired == [("cam", "one")]
