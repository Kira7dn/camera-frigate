"""Frigate-side parity checks against master 50a2b672."""

from types import SimpleNamespace
from unittest.mock import ANY, Mock

import numpy as np

from frigate.infrastructure.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)
from frigate.infrastructure.data_processing.real_time.face import FaceRealTimeProcessor
from frigate.infrastructure.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)
from frigate.application.embeddings.maintainer import EmbeddingMaintainer
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.application.recognition.contracts import RecognitionTask
from frigate.application.recognition.ports import RawRecognition


def plate(value: str, confidence: float, char: float, area: int) -> dict:
    return {
        "plate": value,
        "conf": confidence,
        "char_confidences": [char],
        "area": area,
    }


def test_production_lpr_uses_master_cluster_score_and_representative():
    processor = object.__new__(LicensePlateProcessingMixin)
    processor.cluster_threshold = 0.85
    variants = [
        plate("ABC123", 0.91, 0.99, 9999),
        plate("ABC128", 0.95, 0.10, 10),
        plate("ZZ9999", 0.94, 1.00, 99999),
        plate("ZZ9998", 0.93, 1.00, 99999),
    ]
    representative = processor._get_cluster_rep(variants)
    assert representative == ("ABC128", 0.95, [0.1], 10)


def test_production_face_uses_master_weighted_vote_and_min_faces():
    processor = object.__new__(FaceRealTimeProcessor)
    processor.face_config = SimpleNamespace(
        unknown_score=0.8, recognition_threshold=0.9, min_faces=2
    )
    history = [
        ("unknown", 0.99, 99999),
        ("alice", 0.91, 1000),
        ("bob", 0.81, 100),
        ("alice", 0.95, 8000),
    ]
    assert processor.weighted_average(history) == (
        "alice",
        (0.91 * 1100 + 0.95 * 6000) / 7100,
    )


def test_realtime_lpr_calls_adapter_once_without_retry_owner():
    processor = object.__new__(LicensePlateRealTimeProcessor)
    adapter = Mock()
    adapter.observe.return_value = ()
    processor._recognition_adapter = Mock(return_value=adapter)
    obj = {
        "camera": "front",
        "id": "raw-id",
        "box": [1, 2, 3, 4],
        "frame_time": 1.25,
    }
    frame = object()
    processor.process_frame(obj, frame)
    adapter.observe.assert_called_once_with(
        ANY,
        obj,
        1.25,
        "lpr:front:raw-id:1.250000",
        evidence=(obj, frame),
        observed_in_frame=None,
        attributes={"current_attributes": ()},
    )
    assert not hasattr(processor, "_pending_eligibility")


def test_realtime_lpr_model_port_uses_mixin_signature():
    processor = object.__new__(LicensePlateRealTimeProcessor)
    expected = RawRecognition("ABC123", 0.91)
    processor.lpr_process = Mock(return_value=expected)
    obj = {"camera": "front", "id": "raw-id"}
    frame = object()

    result = processor.recognize(RecognitionTask.LPR, object(), (obj, frame))

    assert result is expected
    processor.lpr_process.assert_called_once_with(obj, frame, False)


def test_realtime_face_calls_adapter_once_from_tracked_object():
    processor = object.__new__(FaceRealTimeProcessor)
    processor.config = SimpleNamespace(
        cameras={
            "front": SimpleNamespace(face_recognition=SimpleNamespace(enabled=True))
        }
    )
    adapter = Mock()
    adapter.observe.return_value = ()
    processor._recognition_adapter = Mock(return_value=adapter)
    obj = {
        "camera": "front",
        "id": "raw-face-id",
        "label": "person",
        "box": [1, 2, 101, 202],
        "frame_time": 2.5,
        "current_attributes": [],
    }
    frame = np.zeros((6, 4), dtype=np.uint8)
    processor.process_frame(obj, frame)
    adapter.observe.assert_called_once_with(
        ANY,
        obj,
        2.5,
        "face:front:raw-face-id:2.500000",
        evidence=(frame, None),
        observed_in_frame=None,
        attributes={"current_attributes": []},
    )


def test_tracked_object_end_cleans_recognition_before_frame_lookup():
    maintainer = object.__new__(EmbeddingMaintainer)
    maintainer.event_subscriber = Mock()
    maintainer.event_subscriber.check_for_update.return_value = (
        EventTypeEnum.tracked_object,
        EventStateEnum.end,
        "front",
        "missing-frame",
        {"id": "raw-id", "label": "person"},
    )
    maintainer.config = SimpleNamespace(
        semantic_search=SimpleNamespace(enabled=False),
        cameras={"front": SimpleNamespace(frame_shape_yuv=(6, 4))},
    )
    processor = object.__new__(FaceRealTimeProcessor)
    processor.expire_object = Mock()
    maintainer.realtime_processors = [processor]
    maintainer.post_processors = []
    maintainer.frame_manager = Mock()
    maintainer.frame_manager.get.side_effect = FileNotFoundError

    maintainer._process_updates()

    processor.expire_object.assert_called_once_with("raw-id", "front")


def test_owned_recognition_frame_is_deleted_after_synchronous_observe():
    maintainer = object.__new__(EmbeddingMaintainer)
    maintainer.event_subscriber = Mock()
    maintainer.event_subscriber.check_for_update.return_value = (
        EventTypeEnum.tracked_object,
        EventStateEnum.update,
        "front",
        "recognition_front_unique",
        {
            "id": "raw-id",
            "camera": "front",
            "label": "person",
            "box": [1, 2, 3, 4],
            "frame_time": 1.0,
            "_recognition_evidence_owned": True,
        },
    )
    maintainer.config = SimpleNamespace(
        semantic_search=SimpleNamespace(enabled=False),
        cameras={"front": SimpleNamespace(frame_shape_yuv=(6, 4))},
    )
    processor = Mock()
    maintainer.realtime_processors = [processor]
    maintainer.post_processors = []
    maintainer.frame_manager = Mock()
    frame = np.zeros((6, 4), dtype=np.uint8)
    maintainer.frame_manager.get.return_value = frame

    maintainer._process_updates()

    processor.process_frame.assert_called_once()
    maintainer.frame_manager.delete.assert_called_once_with(
        "recognition_front_unique"
    )
    maintainer.frame_manager.close.assert_not_called()
