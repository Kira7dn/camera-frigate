from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import numpy as np

from frigate.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)
from frigate.embeddings.maintainer import EmbeddingMaintainer
from frigate.events.types import EventStateEnum, EventTypeEnum


def test_realtime_processor_does_not_own_passage_identity() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)

    assert not hasattr(processor, "_with_passage_identity")
    assert not hasattr(processor, "_passage_registry")


def test_realtime_processor_delegates_frame_to_upstream_lpr_process() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    calls = []
    processor.lpr_process = MethodType(
        lambda _self, obj, frame, dedicated=False: calls.append(
            (obj, frame, dedicated)
        ),
        processor,
    )
    frame = np.zeros((6, 4), dtype=np.uint8)
    obj = {"id": "car-1", "camera": "cam"}

    processor.process_frame(obj, frame)

    assert calls == [(obj, frame, False)]


def test_realtime_processor_delegates_expiry_to_upstream_lpr_expire() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    calls = []
    processor.lpr_expire = MethodType(
        lambda _self, object_id, camera: calls.append((object_id, camera)),
        processor,
    )

    processor.expire_object("car-1", "cam")

    assert calls == [("car-1", "cam")]


def test_maintainer_passes_canonical_event_yuv_frame_directly_to_lpr() -> None:
    submitted = []
    lpr = object.__new__(LicensePlateRealTimeProcessor)
    lpr.process_frame = MethodType(
        lambda _self, obj, frame, dedicated=False: submitted.append(
            (obj["id"], frame, dedicated)
        ),
        lpr,
    )
    frame = np.zeros((6, 4), dtype=np.uint8)
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
        semantic_search=SimpleNamespace(enabled=False),
    )
    maintainer.realtime_processors = [lpr]
    maintainer.post_processors = []
    maintainer.event_subscriber = SimpleNamespace(
        check_for_update=lambda: (
            EventTypeEnum.tracked_object,
            EventStateEnum.update,
            "cam",
            "frame",
            {"id": "car-1", "camera": "cam", "label": "car"},
        )
    )
    maintainer.frame_manager = SimpleNamespace(
        get=lambda *_args: frame,
        close=lambda *_args: None,
    )

    maintainer._process_updates()

    assert len(submitted) == 1
    assert submitted[0][0] == "car-1"
    assert submitted[0][1] is frame
    assert submitted[0][2] is False


def test_maintainer_does_not_run_lpr_on_end_event_with_stale_bbox() -> None:
    submitted = []
    lpr = object.__new__(LicensePlateRealTimeProcessor)
    lpr.process_frame = MethodType(
        lambda _self, obj, frame, dedicated=False: submitted.append(
            (obj, frame, dedicated)
        ),
        lpr,
    )
    frame = np.zeros((6, 4), dtype=np.uint8)
    maintainer = object.__new__(EmbeddingMaintainer)
    maintainer.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(frame_shape_yuv=(6, 4)),
        },
        semantic_search=SimpleNamespace(enabled=False),
    )
    maintainer.realtime_processors = [lpr]
    maintainer.post_processors = []
    maintainer.event_subscriber = SimpleNamespace(
        check_for_update=lambda: (
            EventTypeEnum.tracked_object,
            EventStateEnum.end,
            "cam",
            "newer-frame-after-car-disappeared",
            {
                "id": "car-1",
                "camera": "cam",
                "label": "car",
                "frame_time": 1.0,
                "box": [0, 0, 4, 4],
            },
        )
    )
    maintainer.frame_manager = SimpleNamespace(
        get=lambda *_args: frame,
        close=lambda *_args: None,
    )

    maintainer._process_updates()

    assert submitted == []


def test_maintainer_does_not_feed_tracked_lpr_from_detection_frame() -> None:
    submitted = []
    lpr = object.__new__(LicensePlateRealTimeProcessor)
    lpr.process_frame = MethodType(
        lambda _self, obj, frame, dedicated=False: submitted.append(
            (obj, frame, dedicated)
        ),
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

    maintainer._process_latest_frame(
        (
            "cam",
            "frame",
            1.0,
            [{"id": "raw-car-1", "label": "car", "box": [0, 0, 4, 4]}],
            [],
            None,
        )
    )

    assert submitted == []


def test_pending_canonical_lpr_retries_only_same_track_with_synchronized_frame() -> None:
    calls = []
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.lp_objects = ["car"]
    processor.config = SimpleNamespace(
        cameras={"cam": SimpleNamespace(lpr=SimpleNamespace(enabled=True))}
    )
    processor.lpr_process = MethodType(
        lambda _self, obj, frame, dedicated=False: calls.append(
            (dict(obj), frame, dedicated)
        ),
        processor,
    )
    canonical_frame = np.zeros((6, 4), dtype=np.uint8)
    retry_frame = np.ones((6, 4), dtype=np.uint8)
    canonical = {
        "id": "car-1",
        "camera": "cam",
        "frame_time": 1.0,
        "label": "car",
        "box": [0, 0, 2, 4],
        "position_changes": 0,
        "stationary": False,
    }

    processor.process_frame(canonical, canonical_frame)
    assert processor.has_pending_retry("cam") is True

    assert (
        processor.retry_pending_frame(
            "cam",
            [
                {
                    "id": "other-car",
                    "label": "car",
                    "box": [2, 0, 4, 4],
                    "position_changes": 2,
                    "stationary": False,
                }
            ],
            retry_frame,
            1.2,
        )
        == 0
    )
    assert len(calls) == 1

    retry_obj = {
        "id": "car-1",
        "label": "car",
        "box": [1, 0, 4, 4],
        "position_changes": 1,
        "stationary": False,
    }
    assert (
        processor.retry_pending_frame("cam", [retry_obj], retry_frame, 1.4) == 1
    )
    assert len(calls) == 2
    submitted, submitted_frame, dedicated = calls[-1]
    assert submitted["id"] == "car-1"
    assert submitted["camera"] == "cam"
    assert submitted["frame_time"] == 1.4
    assert submitted["box"] == retry_obj["box"]
    assert submitted_frame is retry_frame
    assert dedicated is False
    assert processor.has_pending_retry("cam") is False

    # Neither the same detection frame nor a stale canonical update can revive
    # or duplicate a resolved retry lifecycle.
    assert (
        processor.retry_pending_frame("cam", [retry_obj], retry_frame, 1.4) == 0
    )
    processor.process_frame(canonical, canonical_frame)
    assert processor.has_pending_retry("cam") is False
    assert len(calls) == 3


def test_pending_lpr_retry_is_cancelled_on_end_and_shutdown() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.lp_objects = ["car"]
    processor.config = SimpleNamespace(
        cameras={"cam": SimpleNamespace(lpr=SimpleNamespace(enabled=True))}
    )
    processor.lpr_process = MethodType(lambda *_args: None, processor)
    expired = []
    processor.lpr_expire = MethodType(
        lambda _self, object_id, camera: expired.append((object_id, camera)), processor
    )
    frame = np.zeros((6, 4), dtype=np.uint8)

    for object_id in ("car-1", "car-2"):
        processor.process_frame(
            {
                "id": object_id,
                "camera": "cam",
                "frame_time": 1.0,
                "label": "car",
                "box": [0, 0, 4, 4],
                "position_changes": 0,
                "stationary": False,
            },
            frame,
        )

    processor.expire_object("car-1", "cam")
    assert expired == [("car-1", "cam")]
    assert ("cam", "car-1") not in processor._pending()
    assert ("cam", "car-2") in processor._pending()

    processor.shutdown()
    assert processor._pending() == {}
    assert processor._closed_retries() == set()


def test_pending_lpr_retry_is_cancelled_on_stream_epoch_reset() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.lp_objects = ["car"]
    processor.config = SimpleNamespace(
        cameras={"cam": SimpleNamespace(lpr=SimpleNamespace(enabled=True))}
    )
    processor.lpr_process = MethodType(lambda *_args: None, processor)
    frame = np.zeros((6, 4), dtype=np.uint8)
    processor.process_frame(
        {
            "id": "car-1",
            "camera": "cam",
            "frame_time": 10.0,
            "label": "car",
            "box": [0, 0, 4, 4],
            "position_changes": 0,
            "stationary": False,
        },
        frame,
    )

    assert processor.retry_pending_frame("cam", [], frame, 1.0) == 0
    assert processor.has_pending_retry("cam") is False
    assert processor._closed_retries() == set()


def test_maintainer_feeds_detection_frame_only_to_pending_canonical_lpr() -> None:
    retries = []
    lpr = object.__new__(LicensePlateRealTimeProcessor)
    lpr.has_pending_retry = MethodType(lambda _self, camera: camera == "cam", lpr)
    lpr.retry_pending_frame = MethodType(
        lambda _self, camera, objects, frame, frame_time: retries.append(
            (camera, objects, frame, frame_time)
        )
        or 1,
        lpr,
    )
    frame = np.zeros((6, 4), dtype=np.uint8)
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
        get=lambda *_args: frame,
        close=lambda *_args: None,
    )
    objects = [
        {
            "id": "car-1",
            "label": "car",
            "box": [0, 0, 4, 4],
            "position_changes": 1,
        }
    ]

    maintainer._process_latest_frame(("cam", "frame", 1.2, objects, [], None))

    assert retries == [("cam", objects, frame, 1.2)]


def test_synchronous_lpr_has_no_deferred_results() -> None:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    assert processor.drain_results() == []


def test_pre_gate_lpr_invocation_persists_runtime_evidence(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.metrics = SimpleNamespace(
        alpr_pps=SimpleNamespace(value=0.0),
        yolov9_lpr_pps=SimpleNamespace(value=0.0),
    )
    processor.plates_rec_second = SimpleNamespace(eps=lambda: 0.0)
    processor.plates_det_second = SimpleNamespace(eps=lambda: 0.0)
    processor.lp_objects = ["car"]
    processor.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(
                lpr=SimpleNamespace(enabled=True),
                detect=SimpleNamespace(min_initialized=1),
            )
        }
    )
    frame = np.zeros((6, 4), dtype=np.uint8)

    processor.lpr_process(
        {
            "id": "car-1",
            "camera": "cam",
            "frame_time": 1.25,
            "label": "car",
            "score": 0.8,
            "box": [0, 0, 4, 4],
            "area": 16,
            "position_changes": 0,
            "stationary": False,
            "motionless_count": 0,
        },
        frame,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "lpr" / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {record["stage"] for record in records} == {
        "invocation",
        "runtime_frame",
        "runtime_frame_object_box",
        "eligibility_decision",
    }
    decision = next(
        record for record in records if record["stage"] == "eligibility_decision"
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "no_position_changes"
    raw_record = next(record for record in records if record["stage"] == "runtime_frame")
    assert "artifact_path" not in raw_record
    annotated_record = next(
        record for record in records if record["stage"] == "runtime_frame_object_box"
    )
    assert (tmp_path / annotated_record["artifact_path"]).is_file()
    assert annotated_record["artifact_sha256"]
