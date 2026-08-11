"""Face media tests; decision ownership is covered by recognition tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import frigate.data_processing.real_time.face as face_module
from frigate.data_processing.real_time.face import FaceRealTimeProcessor
from frigate.recognition.adapters.frigate import FrigateEventAdapter
from frigate.recognition.contracts import RecognitionTask, RecognitionUpdate, TrackKey
from frigate.util.face_snapshot import FaceSnapshotJob, write_face_snapshot_artifact


def test_face_snapshot_is_encoded_after_decision(tmp_path):
    frame = np.zeros((6, 4), dtype=np.uint8)
    result = write_face_snapshot_artifact(
        FaceSnapshotJob(
            camera="front",
            event_id="raw-track",
            frame_time=1.25,
            person_box=(0, 0, 4, 4),
            face_box=(1, 1, 3, 3),
            sub_label="alice",
            face_score=0.95,
            frame=frame,
        ),
        str(tmp_path),
    )
    assert result is not None
    assert result.event_id == "raw-track"
    assert result.artifact_path.startswith(str(tmp_path))


def test_face_attempt_writer_only_receives_unknown():
    processor = object.__new__(FaceRealTimeProcessor)
    processor.config = SimpleNamespace(
        face_recognition=SimpleNamespace(save_attempts=4)
    )
    processor.face_attempt_worker = Mock()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    processor.queue_face_attempt("front", frame, "raw-track", 1.0, "alice", 0.95)
    processor.face_attempt_worker.submit.assert_not_called()
    processor.queue_face_attempt("front", frame, "raw-track", 2.0, "unknown", 0.1)
    processor.face_attempt_worker.submit.assert_called_once()


def test_face_startup_library_stats_remains_available(tmp_path, monkeypatch):
    identity = tmp_path / "alice"
    identity.mkdir()
    (identity / "one.jpg").write_bytes(b"fixture")
    monkeypatch.setattr(face_module, "FACE_DIR", str(tmp_path))

    processor = object.__new__(FaceRealTimeProcessor)
    stats = processor._FaceRealTimeProcessor__face_library_stats()

    assert stats == (1, 1)


def test_media_failure_does_not_cancel_recognition_publication():
    tracked = []
    metadata = []
    adapter = FrigateEventAdapter(
        tracked.append,
        lambda kind, payload: metadata.append((kind, payload)),
        snapshot_after_decision=Mock(side_effect=OSError("disk unavailable")),
    )
    adapter.on_update(
        RecognitionUpdate(
            task=RecognitionTask.FACE,
            key=TrackKey("front", "epoch", "raw-track"),
            frame_time=1.0,
            evidence_ref="frame-ref",
            raw_value="alice",
            raw_score=0.95,
            aggregate_value="alice",
            aggregate_score=0.95,
            object_bbox=(0, 0, 10, 10),
            detail_bbox=(2, 2, 8, 8),
            publish=True,
            reason="master_weighted_vote",
        )
    )
    assert tracked[0]["id"] == "raw-track"
    assert metadata == [("sub_label", ("raw-track", "alice", 0.95))]
