"""Frozen differential vectors for master 50a2b672 recognition semantics."""

from __future__ import annotations

import importlib
import json
import queue
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest

from frigate.recognition import (
    FacePolicy,
    LprPolicy,
    RawRecognition,
    RecognitionCore,
    RecognitionTask,
    TrackedObservation,
    TrackKey,
)
from frigate.recognition.adapters.frigate import (
    BorrowedEvidenceResolver,
    FrigateEventAdapter,
    FrigateRecognitionAdapter,
)
from frigate.recognition.face import FaceEngine
from frigate.recognition.lpr import LprEngine
from frigate.recognition.writer import BoundedTraceWriter

MASTER_COMMIT = "50a2b6729eb152d9512b100c78c55fa84dffa430"


def observation(
    task: RecognitionTask,
    index: int,
    *,
    track_id: str = "raw-track-1",
    sub_label: str | None = None,
) -> TrackedObservation:
    return TrackedObservation(
        task=task,
        key=TrackKey("front", "process-epoch", track_id),
        frame_time=float(index),
        object_bbox=(1, 2, 101, 202),
        detail_bbox=(11, 12, 61, 42),
        observed_in_frame=None,
        evidence_ref=f"frame-{index}",
        attributes={"sub_label": sub_label},
    )


@pytest.mark.parametrize(
    ("variants", "winner"),
    [
        ([('ABC123', 0.91)], "ABC123"),
        ([('ABC123', 0.91), ('ABC128', 0.92), ('ZZ9999', 0.99)], "ABC128"),
        ([('ABC123', 0.91), ('ZZ9999', 0.95)], "ZZ9999"),
        ([('ABC123', 0.95), ('ZZ9999', 0.95)], "ABC123"),
    ],
)
def test_lpr_frozen_master_decision_sequence(variants, winner):
    engine = LprEngine(LprPolicy(detect_fps=5, recognition_threshold=0.9))
    updates = []
    for index, (plate, score) in enumerate(variants):
        updates.extend(
            engine.observe(
                observation(RecognitionTask.LPR, index),
                RawRecognition(plate, score, area=1000),
            )
        )
    assert updates[-1].aggregate_value == winner


def test_lpr_master_window_pruning_and_threshold():
    engine = LprEngine(LprPolicy(detect_fps=1, recognition_threshold=0.9))
    assert not engine.observe(
        observation(RecognitionTask.LPR, 0), RawRecognition("LOW", 0.89)
    )
    for index in range(6):
        engine.observe(
            observation(RecognitionTask.LPR, index + 1),
            RawRecognition(f"P{index}", 0.91 + index / 100),
        )
    last = engine.observe(
        observation(RecognitionTask.LPR, 7), RawRecognition("P5", 0.99)
    )[0]
    assert last.metadata["history_size"] == 5


def test_face_frozen_master_unknown_min_faces_tie_and_area_cap():
    engine = FaceEngine(
        FacePolicy(unknown_score=0.8, recognition_threshold=0.9, min_faces=2)
    )
    key_obs = observation(RecognitionTask.FACE, 0)
    assert engine.observe(key_obs, RawRecognition("unknown", 0.99, area=9999))[0].aggregate_value is None
    assert engine.observe(key_obs, RawRecognition("alice", 0.91, area=1000))[0].aggregate_value is None
    assert engine.observe(key_obs, RawRecognition("bob", 0.81, area=100))[0].aggregate_value is None
    update = engine.observe(key_obs, RawRecognition("alice", 0.95, area=8000))[0]
    assert update.aggregate_value == "alice"
    assert update.aggregate_score == pytest.approx((0.91 * 1100 + 0.95 * 6000) / 7100)

    tie_engine = FaceEngine(
        FacePolicy(unknown_score=0.8, recognition_threshold=0.9, min_faces=2)
    )
    for name, score in (("alice", 0.99), ("alice", 0.98), ("bob", 0.91)):
        tie_engine.observe(key_obs, RawRecognition(name, score, area=4000))
    tie = tie_engine.observe(key_obs, RawRecognition("bob", 0.90, area=4000))[0]
    assert tie.aggregate_value is None


def test_face_master_attempt_limits_12_and_6():
    policy = FacePolicy(unknown_score=0.8, recognition_threshold=0.9)
    engine = FaceEngine(policy)
    no_rec = observation(RecognitionTask.FACE, 0)
    for _ in range(12):
        assert engine.should_attempt(no_rec)
        engine.observe(no_rec, RawRecognition("unknown", 0.1, area=100))
    assert not engine.should_attempt(no_rec)

    first = observation(RecognitionTask.FACE, 0, track_id="recognized")
    assert engine.should_attempt(first)
    engine.observe(first, RawRecognition("alice", 0.95, area=1000))
    recognized = observation(
        RecognitionTask.FACE, 1, track_id="recognized", sub_label="alice"
    )
    for _ in range(5):
        assert engine.should_attempt(recognized)
        engine.observe(recognized, RawRecognition("alice", 0.95, area=1000))
    assert not engine.should_attempt(recognized)


class FakeEvidence:
    def resolve(self, obs):
        return nullcontext(obs.evidence_ref)


class FakeModel:
    def recognize(self, task, observation, evidence):
        assert evidence == observation.evidence_ref
        return RawRecognition("ABC123", 0.95, detail_bbox=observation.detail_bbox, area=1000)


def test_core_lifecycle_is_synchronous_and_idempotent():
    core = RecognitionCore(
        FakeModel(),
        FakeEvidence(),
        LprPolicy(5, 0.9),
        FacePolicy(0.8, 0.9),
    )
    obs = observation(RecognitionTask.LPR, 1)
    assert core.observe(obs)[0].evidence_ref == "frame-1"
    assert core.stats == {"sessions": 1, "in_flight": 0, "evidence_pinned": 0}
    core.end_track(obs.key, "ended")
    core.end_track(obs.key, "ended-again")
    assert core.stats == {"sessions": 0, "in_flight": 0, "evidence_pinned": 0}
    core.shutdown()
    core.shutdown()
    assert not core.observe(obs)


def test_core_does_not_resolve_evidence_for_explicitly_unobserved_bbox():
    model = FakeModel()
    core = RecognitionCore(
        model,
        FakeEvidence(),
        LprPolicy(5, 0.9),
        FacePolicy(0.8, 0.9),
    )
    base = observation(RecognitionTask.FACE, 1)
    stale = TrackedObservation(
        task=base.task,
        key=base.key,
        frame_time=base.frame_time,
        object_bbox=base.object_bbox,
        detail_bbox=base.detail_bbox,
        observed_in_frame=False,
        evidence_ref=base.evidence_ref,
        attributes=base.attributes,
    )

    assert core.observe(stale) == ()
    assert core.stats == {"sessions": 0, "in_flight": 0, "evidence_pinned": 0}


def test_adapter_preserves_raw_lineage_and_publishes_once():
    tracked = []
    metadata = []
    event_adapter = FrigateEventAdapter(
        tracked.append,
        lambda kind, payload: metadata.append((kind, payload)),
    )
    evidence = BorrowedEvidenceResolver()
    core = RecognitionCore(
        FakeModel(),
        evidence,
        LprPolicy(5, 0.9),
        FacePolicy(0.8, 0.9),
        event_adapter,
    )
    adapter = FrigateRecognitionAdapter(core, "process-epoch", evidence)
    updates = adapter.observe(
        RecognitionTask.LPR,
        {"camera": "front", "id": "raw-track-1", "box": [1, 2, 101, 202]},
        12.5,
        "frame-ref",
        evidence="frame-ref",
        detail_bbox=(11, 12, 61, 42),
    )
    assert len(updates) == len(tracked) == 1
    assert tracked[0]["id"] == "raw-track-1"
    assert tracked[0]["timestamp"] == 12.5
    assert tracked[0]["bbox"] == (1, 2, 101, 202)
    assert tracked[0]["plate_box"] == (11, 12, 61, 42)
    assert tracked[0]["evidence_ref"] == "frame-ref"
    assert metadata == [
        ("attribute", ("raw-track-1", "recognized_license_plate", "ABC123", 0.95))
    ]
    assert not adapter.observe(
        RecognitionTask.LPR,
        {"camera": "front", "id": "raw-track-1", "box": [1, 2, 101, 202]},
        12.5,
        "frame-ref",
        evidence="frame-ref",
        detail_bbox=(11, 12, 61, 42),
    )
    assert len(tracked) == 1
    assert adapter.stats["evidence_pinned"] == 0
    adapter.end_track("front", "raw-track-1", "event_end")
    assert adapter.stats["sessions"] == 0
    assert not adapter.observe(
        RecognitionTask.LPR,
        {"camera": "front", "id": "raw-track-1", "box": [1, 2, 101, 202]},
        13.0,
        "late-frame",
        evidence="late-frame",
    )


def test_import_isolation():
    forbidden = (
        "frigate.comms",
        "frigate.events",
        "frigate.object_detection",
        "frigate.embeddings.maintainer",
    )
    before = set(sys.modules)
    module = importlib.reload(importlib.import_module("frigate.recognition"))
    assert module.RecognitionCore
    loaded = set(sys.modules) - before
    assert not any(name.startswith(forbidden) for name in loaded)


def test_import_isolation_in_fresh_interpreter():
    script = """
import json, sys
import frigate.recognition
print(json.dumps(sorted(sys.modules)))
"""
    loaded = set(
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parents[2],
            ).stdout
        )
    )
    forbidden = (
        "frigate.comms",
        "frigate.events",
        "frigate.object_detection",
        "frigate.embeddings.maintainer",
    )
    assert not any(name.startswith(forbidden) for name in loaded)


def test_production_has_no_phase5_decision_owner():
    package = Path(__file__).parents[1]
    production = (
        package / "embeddings" / "maintainer.py",
        package / "data_processing" / "real_time" / "face.py",
        package / "data_processing" / "real_time" / "license_plate.py",
        package / "data_processing" / "common" / "license_plate" / "mixin.py",
    )
    forbidden = (
        "BestResultReducer",
        "RecognitionLifecycle",
        "FaceRecognitionPipeline",
        "PreparedPlateCandidate",
        "PendingLprEligibility",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in production)
    assert "FrigateRecognitionAdapter" in combined
    assert not any(name in combined for name in forbidden)
    for removed in (
        package / "data_processing" / "common" / "recognition.py",
        package / "data_processing" / "common" / "face" / "pipeline.py",
        package / "data_processing" / "common" / "license_plate" / "pipeline.py",
    ):
        assert not removed.exists()


def test_trace_writer_overflow_error_and_flush(tmp_path: Path):
    blocker = queue.Queue()

    def encode(_image):
        blocker.get(timeout=1)
        raise ValueError("encode failed")

    writer = BoundedTraceWriter(tmp_path, capacity=1, encode_jpeg=encode)
    assert writer.submit({"one": 1}, image_name="one.jpg", image=b"frame")
    deadline = time.monotonic() + 1
    while writer.depth and time.monotonic() < deadline:
        time.sleep(0.005)
    assert writer.submit({"two": 2})
    assert not writer.submit({"three": 3})
    assert writer.dropped == 1
    blocker.put(None)
    assert writer.close(1)
    assert writer.errors == 1
    assert '"two": 2' in (tmp_path / "recognition.jsonl").read_text(encoding="utf-8")


def test_trace_writer_byte_limit_uses_encoded_artifact_size(tmp_path: Path):
    writer = BoundedTraceWriter(
        tmp_path,
        capacity=4,
        encode_jpeg=lambda image: bytes(image),
        max_artifact_bytes=5,
    )

    assert writer.submit({"id": 1}, image_name="one.jpg", image=b"1234")
    assert writer.submit({"id": 2}, image_name="two.jpg", image=b"5678")
    assert writer.close(1)

    records = [
        json.loads(line)
        for line in (tmp_path / "recognition.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["artifact_bytes"] == 4
    assert records[1]["artifact_rejected"] == "byte_limit"
    assert "artifact_path" not in records[1]
    assert (tmp_path / "one.jpg").read_bytes() == b"1234"
    assert not (tmp_path / "two.jpg").exists()
    assert writer.artifact_bytes == 4
