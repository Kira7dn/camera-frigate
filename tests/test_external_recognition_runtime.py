from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from frigate.infrastructure.config.recognition import RecognitionRuntimeConfig
from frigate.infrastructure.data_processing.common.face_pipeline import render_recognition_boxes
from frigate.infrastructure.data_processing.real_time import external_recognition as runtime
from frigate.infrastructure.data_processing.real_time.face import FaceRealTimeProcessor
from frigate.application.recognition.contracts import (
    JobReceipt,
    RecognitionOperation,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
)
from extension.recognition.models import FaceRecognitionModel
from extension.recognition.threaded_client import ClientResult


class FakeClient:
    instance = None
    reject_reason = None

    def __init__(self, *args, **kwargs) -> None:
        self.service_epoch = "service"
        self.jobs = []
        self.results = []
        self.stats = {"queue_depth": 0, "outcome_depth": 0, "healthy": 1}
        FakeClient.instance = self

    def submit_nowait(self, job):
        self.jobs.append(job)
        if self.reject_reason:
            return JobReceipt(
                job.job_id, self.service_epoch, False, self.reject_reason, False
            )
        return JobReceipt(job.job_id, self.service_epoch, True)

    def drain_results(self):
        values = tuple(self.results)
        self.results.clear()
        return values

    def close(self, timeout):
        return True


class FakeConfig(SimpleNamespace):
    def model_dump_json(self, **kwargs) -> str:
        return "{}"


def config() -> FakeConfig:
    recognition = SimpleNamespace(
        endpoint="recognition:50051",
        deadline=5.0,
        job_deadline=30.0,
        observation_capacity=8,
        control_capacity=4,
        outcome_capacity=8,
        shutdown_drain=1.0,
        tls=SimpleNamespace(
            ca="ca", certificate="certificate", key="key", server_name=None
        ),
    )
    camera = SimpleNamespace(
        detect=SimpleNamespace(fps=5),
        face_recognition=SimpleNamespace(enabled=True),
        lpr=SimpleNamespace(enabled=True),
    )
    return FakeConfig(
        recognition=recognition,
        face_recognition=SimpleNamespace(enabled=True),
        lpr=SimpleNamespace(enabled=True, known_plates={}, match_distance=1),
        cameras={"front": camera},
    )


def test_external_runtime_requires_complete_tls_configuration():
    with pytest.raises(ValueError, match="requires TLS fields"):
        RecognitionRuntimeConfig(runtime="external", endpoint="recognition:50051")


def test_shared_bbox_renderer_draws_person_and_face_boxes() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    rendered = render_recognition_boxes(
        image,
        object_box=(1, 1, 18, 18),
        detail_box=(5, 5, 10, 10),
    )

    assert rendered[1, 1].tolist() == [0, 0, 255]
    assert rendered[5, 5].tolist() == [0, 255, 0]
    assert not image.any()


def test_external_face_detector_bbox_matches_synchronous_pipeline() -> None:
    class Detector:
        def setInputSize(self, size) -> None:
            self.size = size

        def detect(self, image):
            return None, np.array(
                [[1.9, 2.9, 10.9, 20.9, 0, 0, 0, 0, 0.91]], dtype=np.float32
            )

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    synchronous = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
    synchronous.face_detector = Detector()
    external = FaceRecognitionModel.__new__(FaceRecognitionModel)
    external.face_detector = Detector()

    assert synchronous._FaceRealTimeProcessor__detect_face(image, 0.5) == (  # type: ignore[attr-defined]
        1,
        2,
        11,
        22,
    )
    assert external._detect_face(
        image, 0.5
    ) == synchronous._FaceRealTimeProcessor__detect_face(  # type: ignore[attr-defined]
        image, 0.5
    )


def test_external_face_model_matches_synchronous_crop_and_result() -> None:
    class Recognizer:
        def __init__(self) -> None:
            self.shapes = []

        def classify(self, image):
            self.shapes.append(tuple(image.shape))
            return "Joe", 0.97

    class Metric:
        def update(self, *args) -> None:
            return None

    camera = SimpleNamespace(face_recognition=SimpleNamespace(min_area=1))
    shared_config = SimpleNamespace(cameras={"front": camera})
    face_config = SimpleNamespace(unknown_score=0.8, detection_threshold=0.7)
    observation = TrackedObservation(
        RecognitionTask.FACE,
        TrackKey("front", "stream", "track"),
        1.0,
        (0, 0, 4, 4),
        evidence_ref="evidence",
        attributes={
            "current_attributes": (
                {"label": "face", "score": 0.9, "box": [1, 1, 4, 5]},
            )
        },
    )
    yuv = np.zeros((6, 4), dtype=np.uint8)

    local_recognizer = Recognizer()
    synchronous = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
    synchronous.config = shared_config
    synchronous.face_config = face_config
    synchronous.requires_face_detection = False
    synchronous.recognizer = local_recognizer
    synchronous.face_counters = Counter()
    synchronous.faces_per_second = Metric()
    synchronous.inference_speed = Metric()

    remote_recognizer = Recognizer()
    external = FaceRecognitionModel.__new__(FaceRecognitionModel)
    external.config = shared_config
    external.face_config = face_config
    external.requires_face_detection = False
    external.recognizer = remote_recognizer
    external.face_counters = Counter()

    local_result = synchronous.recognize(RecognitionTask.FACE, observation, (yuv, None))
    remote_result = external.recognize(observation, yuv)

    assert remote_result == local_result
    assert remote_recognizer.shapes == local_recognizer.shapes == [(3, 3, 3)]


def test_external_processor_submits_copied_evidence_and_ordered_end(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runtime, "ThreadedRecognitionClient", FakeClient)
    monkeypatch.setattr(runtime, "_read_bytes", lambda path: path.encode())
    monkeypatch.setattr(runtime, "canonical_config_json", lambda config: "{}")
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    requestor = SimpleNamespace(send_data=lambda *args: None)
    publisher = SimpleNamespace(publish=lambda *args: None)
    processor = runtime.ExternalRecognitionProcessor(
        config(), requestor, publisher, SimpleNamespace(), "stream"
    )
    frame = np.arange(24, dtype=np.uint8).reshape((6, 4))
    processor.process_frame(
        {
            "camera": "front",
            "id": "track",
            "frame_time": 1.0,
            "box": [0, 0, 2, 2],
            "label": "person",
            "observed_in_frame": True,
        },
        frame,
    )
    client = FakeClient.instance
    assert client is not None
    assert [job.operation for job in client.jobs] == [
        RecognitionOperation.OBSERVE,
        RecognitionOperation.OBSERVE,
    ]
    assert [job.sequence for job in client.jobs] == [0, 1]
    face_capture = client.jobs[0].observation.evidence_capture
    lpr_capture = client.jobs[1].observation.evidence_capture
    assert face_capture is not None
    assert face_capture.trace_id == "face:front:track"
    assert lpr_capture is not None
    assert lpr_capture.trace_id == "lpr:front:track"
    assert face_capture.evidence_id.startswith("track-")
    assert lpr_capture.evidence_id.startswith("track-")
    evidence = client.jobs[0].observation.evidence_ref
    original = evidence.data
    frame[:] = 0
    assert evidence.data == original

    processor.expire_object("track", "front")
    end_job = client.jobs[-1]
    assert end_job.operation is RecognitionOperation.END_TRACK
    assert end_job.sequence == 2
    assert end_job.key in processor._ending
    assert end_job.key not in processor._ended
    processor.process_frame(
        {
            "camera": "front",
            "id": "track",
            "frame_time": 2.0,
            "box": [0, 0, 2, 2],
            "label": "person",
        },
        frame,
    )
    assert len(client.jobs) == 3

    # Accepted observations stay applicable until the ordered END outcome arrives.
    face_job = client.jobs[0]
    client.results.extend(
        (
            ClientResult(
                outcome=RecognitionOutcome(
                    face_job.job_id,
                    face_job.client_id,
                    client.service_epoch,
                    face_job.key,
                    face_job.sequence,
                    RecognitionOutcomeStatus.SUCCEEDED,
                )
            ),
            ClientResult(
                outcome=RecognitionOutcome(
                    end_job.job_id,
                    end_job.client_id,
                    client.service_epoch,
                    end_job.key,
                    end_job.sequence,
                    RecognitionOutcomeStatus.ENDED,
                )
            ),
        )
    )
    processor.drain_results()
    assert end_job.key not in processor._ending
    assert end_job.key in processor._ended
    assert end_job.key not in processor._sequence
    processor.shutdown()


def test_external_processor_traces_rejected_observation(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ThreadedRecognitionClient", FakeClient)
    monkeypatch.setattr(runtime, "_read_bytes", lambda path: path.encode())
    monkeypatch.setattr(runtime, "canonical_config_json", lambda config: "{}")
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    FakeClient.reject_reason = "service_unavailable"
    traces = []
    monkeypatch.setattr(runtime, "passage_trace", lambda stage, **fields: traces.append((stage, fields)))
    processor = runtime.ExternalRecognitionProcessor(
        config(), SimpleNamespace(send_data=lambda *args: None),
        SimpleNamespace(publish=lambda *args: None), SimpleNamespace(), "stream"
    )
    processor.process_frame(
        {"camera": "front", "id": "rejected", "frame_time": 1.0,
         "box": [0, 0, 2, 2], "label": "car"},
        np.zeros((6, 4), dtype=np.uint8),
    )
    assert any(
        stage == "recognition_failed" and fields["reason"] == "service_unavailable"
        for stage, fields in traces
    )
    processor.shutdown()
    FakeClient.reject_reason = None


def test_face_only_camera_does_not_enqueue_lpr_or_duplicate_lineage(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runtime, "ThreadedRecognitionClient", FakeClient)
    monkeypatch.setattr(runtime, "_read_bytes", lambda path: path.encode())
    monkeypatch.setattr(runtime, "canonical_config_json", lambda config: "{}")
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    value = config()
    value.cameras["face_camera"] = value.cameras.pop("front")
    processor = runtime.ExternalRecognitionProcessor(
        value,
        SimpleNamespace(send_data=lambda *args: None),
        SimpleNamespace(publish=lambda *args: None),
        SimpleNamespace(),
        "stream",
    )
    processor.process_frame(
        {
            "camera": "face_camera",
            "id": "track",
            "frame_time": 1.0,
            "box": [0, 0, 2, 2],
            "label": "person",
            "observed_in_frame": True,
        },
        np.arange(24, dtype=np.uint8).reshape((6, 4)),
    )

    client = FakeClient.instance
    assert client is not None
    assert len(client.jobs) == 1
    assert client.jobs[0].observation.task is RecognitionTask.FACE
    processor.shutdown()


def test_evidence_side_effect_cannot_suppress_recognition_update(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(runtime, "ThreadedRecognitionClient", FakeClient)
    monkeypatch.setattr(runtime, "_read_bytes", lambda path: path.encode())
    monkeypatch.setattr(runtime, "canonical_config_json", lambda config: "{}")
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    processor = runtime.ExternalRecognitionProcessor(
        config(),
        SimpleNamespace(send_data=lambda *args: None),
        SimpleNamespace(publish=lambda *args: None),
        SimpleNamespace(),
        "stream",
    )
    frame = np.arange(24, dtype=np.uint8).reshape((6, 4))
    processor.process_frame(
        {
            "camera": "front",
            "id": "track",
            "frame_time": 1.0,
            "box": [0, 0, 2, 2],
            "label": "person",
            "observed_in_frame": True,
        },
        frame,
    )
    client = FakeClient.instance
    assert client is not None
    lpr_job = client.jobs[1]
    update = RecognitionUpdate(
        task=RecognitionTask.LPR,
        key=lpr_job.key,
        frame_time=1.0,
        evidence_ref="evidence",
        raw_value="ABC123",
        raw_score=0.99,
        aggregate_value="ABC123",
        aggregate_score=0.99,
        object_bbox=(0, 0, 2, 2),
        detail_bbox=(0, 0, 1, 1),
        publish=True,
        reason="master_variant_representative",
    )
    applied = []
    processor._events = SimpleNamespace(on_update=applied.append)
    client.results.append(
        ClientResult(
            outcome=RecognitionOutcome(
                lpr_job.job_id,
                lpr_job.client_id,
                client.service_epoch,
                lpr_job.key,
                lpr_job.sequence,
                RecognitionOutcomeStatus.SUCCEEDED,
                updates=(update,),
                reason="",
                artifacts=(),
            )
        )
    )

    processor.drain_results()

    assert applied == [update]
    processor.shutdown()
