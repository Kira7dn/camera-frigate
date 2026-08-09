from __future__ import annotations

from collections import OrderedDict
from types import MethodType, SimpleNamespace

import numpy as np

from frigate.data_processing.common.evidence import (
    EvidenceBufferPolicy,
    EvidenceCandidate,
    EvidenceRingBuffer,
)
from frigate.data_processing.common.license_plate.pipeline import (
    LatestLprTaskQueue,
    LprTrackKey,
    PlateCommit,
    PlateObservation,
)
from frigate.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)


def make_processor(threshold: float = 0.5) -> LicensePlateRealTimeProcessor:
    processor = object.__new__(LicensePlateRealTimeProcessor)
    processor.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(
                detect=SimpleNamespace(fps=5),
                lpr=SimpleNamespace(expire_time=2.0),
            )
        }
    )
    processor.lpr_config = SimpleNamespace(
        recognition_threshold=threshold,
        min_plate_length=4,
        format=None,
        known_plates={},
        match_distance=1,
    )
    processor.cluster_threshold = 0.85
    processor._tasks = LatestLprTaskQueue(8)
    processor._states = OrderedDict()
    processor.quality_selector = None
    processor.evidence_ring = EvidenceRingBuffer(
        {"cam": EvidenceBufferPolicy(10.0, 8 * 1024 * 1024, 100.0)}
    )

    def cluster(_self, variants):
        best = max(variants, key=lambda value: value["conf"])
        return (
            best["plate"],
            best["conf"],
            best["char_confidences"],
            best["area"],
        )

    def materialize(_self, observation, event_id, normalized, plate, score, sub_label):
        return PlateCommit(
            commit_id=f"{observation.key}:{normalized}",
            key=observation.key,
            event_id=event_id,
            camera="cam",
            plate=plate,
            score=score,
            sub_label=sub_label,
            timestamp=observation.frame_time,
            frame_time=observation.frame_time,
            plate_box=observation.plate_box,
            object_box=observation.object_box,
            evidence_id="evidence",
            frame_ref="frame.jpg",
            frame_width=4,
            frame_height=4,
            obj_data=observation.obj_data,
            dedicated_lpr=observation.dedicated_lpr,
        )

    processor._get_cluster_rep = MethodType(cluster, processor)
    processor._materialize_commit = MethodType(materialize, processor)
    return processor


def observation(
    processor: LicensePlateRealTimeProcessor,
    key: LprTrackKey,
    frame_time: float,
    plate: str = "ABC1234",
    confidence: float = 0.9,
    box: tuple[int, int, int, int] = (0, 0, 100, 100),
) -> PlateObservation:
    frame = np.zeros((6, 4), dtype=np.uint8)
    ref = processor.evidence_ring.ingest(
        key.camera, "detect", f"{key.track_id}-{frame_time}-{plate}", frame_time, frame
    )
    assert ref is not None
    lease = processor.evidence_ring.acquire(ref)
    assert lease is not None
    candidate = EvidenceCandidate(
        f"{key.track_id}-{frame_time}-{plate}",
        "lpr",
        key.camera,
        key.track_id,
        key.generation,
        ref,
        box,
        (10, 10, 40, 30),
        1.0,
        {"dimensions": 1.0},
        (),
        (),
        ref.source_role,
        lease,
    )
    return PlateObservation(
        key=key,
        frame_time=frame_time,
        plate=plate,
        char_confidences=(confidence,) * len(plate),
        text_area=100,
        plate_box=(10, 10, 40, 30),
        object_box=box,
        obj_data={"id": key.track_id, "camera": key.camera, "box": box},
        dedicated_lpr=False,
        evidence=candidate,
    )


def test_consensus_counts_each_frame_once_and_commits_decision_once() -> None:
    processor = make_processor()
    key = LprTrackKey("cam", "track", 0)
    first = observation(processor, key, 1.0)

    assert processor._reduce(first) is not None
    duplicate = observation(processor, key, 1.0)
    assert processor._reduce(duplicate) is None
    assert processor._reduce(observation(processor, key, 2.0)) is None
    assert len(processor._states[key].variants) == 2
    assert len(processor._states[key].seen_frames) == 2


def test_confirmed_close_follow_starts_a_new_generation() -> None:
    processor = make_processor()
    old_key = LprTrackKey("cam", "track", 0)
    assert (
        processor._reduce(observation(processor, old_key, 1.0, "ABC1234")) is not None
    )

    assert (
        processor._reduce(
            observation(processor, old_key, 2.0, "ZZZ9999", box=(500, 500, 700, 700))
        )
        is None
    )
    commit = processor._reduce(
        observation(processor, old_key, 3.0, "ZZZ9999", box=(510, 510, 710, 710))
    )

    assert commit is not None
    assert commit.key.generation == 1
    assert not processor._tasks.is_current(old_key)
    assert processor._states[commit.key].variants[0]["plate"] == "ZZZ9999"


def test_state_is_bounded() -> None:
    processor = make_processor(threshold=2.0)
    for index in range(processor.MAX_STATES + 10):
        key = LprTrackKey("cam", f"track-{index}", 0)
        processor._reduce(observation(processor, key, float(index + 1)))
    assert len(processor._states) == processor.MAX_STATES


def test_commit_uses_representative_candidate_frame_and_bbox() -> None:
    processor = make_processor(threshold=1.0)
    key = LprTrackKey("cam", "track", 0)
    representative = observation(
        processor,
        key,
        1.0,
        "ABC1234",
        confidence=0.95,
        box=(10, 10, 110, 110),
    )
    assert processor._reduce(representative) is None

    processor.lpr_config.recognition_threshold = 0.5
    current = observation(
        processor,
        key,
        2.0,
        "ABC1234",
        confidence=0.80,
        box=(200, 200, 300, 300),
    )
    commit = processor._reduce(current)

    assert commit is not None
    assert commit.frame_time == representative.frame_time
    assert commit.object_box == representative.object_box
