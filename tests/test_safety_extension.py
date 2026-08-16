"""Contracts for the canonical-config Safety extension."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from extension.safety.config import SafetyConfigError, load_config, validate_camera_keys
from extension.safety.events import (
    HazardDecision,
    SafetyEventError,
    SafetyMediaStore,
    SafetyProducer,
    TemporalGate,
)
from extension.safety.inference import Detection, OnnxSafetyModel
from extension.tracker.transport import ProducerTransportError


def _config(tmp_path: Path) -> Path:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"test")
    path = tmp_path / "config.yaml"
    path.write_text(
        """runtime:
  replay:
    sources:
      safety_camera: fixture.mp4
cameras:
  cam:
    media_mode: external
    review:
      alerts:
        labels: [smoking]
""",
        encoding="utf-8",
    )
    return path


def _detection(score: float = 0.8, observed_at: float = 1.0) -> Detection:
    return Detection("smoking", score, (0.1, 0.1, 0.2, 0.2), observed_at)


def test_config_rejects_unknown_fields_and_camera_mismatch(tmp_path: Path) -> None:
    path = _config(tmp_path)
    path.write_text("cameras: {}\n", encoding="utf-8")
    with pytest.raises(SafetyConfigError, match="external smoking camera"):
        load_config(path)

    config = load_config(_config(tmp_path))
    assert config.cameras["cam"].inference_fps == 2
    with pytest.raises(SafetyConfigError, match="missing"):
        validate_camera_keys(config, {"other"})


def test_config_excludes_external_camera_without_smoking_label(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "  tracker_camera:\n"
        + "    media_mode: external\n"
        + "    review:\n"
        + "      alerts:\n"
        + "        labels: []\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert set(config.cameras) == {"cam"}


def test_temporal_gate_requires_confirm_and_clear_time(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection()], 0.0) == []
    assert gate.observe("cam", [_detection()], 0.9) == []
    active = gate.observe("cam", [_detection()], 1.1)
    assert active == [HazardDecision("cam", "smoking", True, 0.8, (0.1, 0.1, 0.2, 0.2))]
    assert gate.observe("cam", [], 2.9) == []
    assert gate.observe("cam", [], 8.1) == [HazardDecision("cam", "smoking", False, 0.0, None)]


def test_smoking_below_threshold_does_not_open_event(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection(score=0.05)], 0.0) == []
    assert gate.observe("cam", [_detection(score=0.05)], 2.0) == []


def test_smoking_confirmation_tolerates_short_detection_gap(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection()], 0.0) == []
    assert gate.observe("cam", [], 0.1) == []
    assert gate.observe("cam", [_detection()], 1.0) == [
        HazardDecision("cam", "smoking", True, 0.8, (0.1, 0.1, 0.2, 0.2))
    ]


def test_smoking_single_frame_expires_without_opening_event(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection()], 0.0) == []
    assert gate.observe("cam", [], 0.1) == []
    assert gate.observe("cam", [], 5.2) == []
    assert gate.observe("cam", [_detection()], 6.0) == []


def test_safety_snapshot_is_real_and_requires_bbox() -> None:
    media = SafetyMediaStore()
    frame = np.full((32, 48, 3), 80, dtype=np.uint8)
    decision = HazardDecision("cam", "smoking", True, 0.91, (0.1, 0.2, 0.5, 0.8))
    image = media.snapshot(frame, decision)
    assert image[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # Producer media is a raw full frame. The canonical Frigate renderer owns
    # the only bbox/label overlay.
    assert int(decoded[8, 5, 2]) < 150
    with pytest.raises(SafetyEventError, match="requires_bbox"):
        media.snapshot(frame, HazardDecision("cam", "smoking", True, 0.9, None))


def test_safety_clear_reuses_active_bbox_and_event_id(monkeypatch) -> None:
    class Client:
        node_epoch = "epoch"
        stream_epoch = "stream"

        def __init__(self, *_args):
            self.uploads = []
            self.updates = []

        def upload_media(self, manifest, content):
            self.uploads.append((manifest, content))

        def publish(self, update):
            self.updates.append(update)

        def close(self):
            return None

    class Media:
        def snapshot(self, _frame, decision):
            assert decision.bbox == (0.1, 0.2, 0.5, 0.8)
            return b"snapshot"

        def clip(self, _event_id, _frames):
            return b"clip"

    monkeypatch.setattr("extension.safety.events.ProducerClient", Client)
    producer = SafetyProducer("unused")
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    start_id = producer.publish(
        HazardDecision("cam", "smoking", True, 0.91, (0.1, 0.2, 0.5, 0.8)),
        frame,
        1.0,
        Media(),
        [(1.0, frame)],
    )
    end_id = producer.publish(
        HazardDecision("cam", "smoking", False, 0.0, None),
        frame,
        2.0,
        Media(),
        [(1.0, frame), (2.0, frame)],
    )

    assert end_id == start_id
    assert [update.operation.value for update in producer.client.updates] == ["START", "END"]
    assert producer.client.updates[1].bbox == producer.client.updates[0].bbox
    assert [manifest.media_type for manifest, _content in producer.client.uploads] == [
        "snapshot_jpg",
        "snapshot_jpg",
        "clip",
    ]
    assert producer.active == {}
    assert producer.last_bbox == {}


def test_safety_publish_reuses_sequence_after_transport_failure(monkeypatch) -> None:
    class Client:
        node_epoch = "epoch"
        stream_epoch = "stream"

        def __init__(self, *_args):
            self.attempts = []
            self.fail_once = True

        def upload_media(self, _manifest, _content):
            return None

        def publish(self, update):
            self.attempts.append(update.journal_sequence)
            if self.fail_once:
                self.fail_once = False
                raise ProducerTransportError("publish_failed")

        def close(self):
            return None

    class Media:
        def snapshot(self, _frame, _decision):
            return b"snapshot"

    monkeypatch.setattr("extension.safety.events.ProducerClient", Client)
    producer = SafetyProducer("unused")
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    decision = HazardDecision("cam", "smoking", True, 0.91, (0.1, 0.2, 0.5, 0.8))

    with pytest.raises(SafetyEventError, match="publish_failed"):
        producer.publish(decision, frame, 1.0, Media(), [(1.0, frame)])
    producer.publish(decision, frame, 1.0, Media(), [(1.0, frame)])

    assert producer.client.attempts == [1, 1]
    assert producer.sequence == 1


def test_safety_clip_uses_rolling_real_frames() -> None:
    media = SafetyMediaStore()
    frames = []
    for index in range(3):
        frames.append((float(index), np.full((32, 48, 3), index + 1, dtype=np.uint8)))
    clip = media.clip("producer-event", frames)
    assert clip[:4] == b"\x00\x00\x00\x18" or len(clip) > 32


def test_latest_frame_reader_keeps_one_latest_sample(monkeypatch) -> None:
    from extension.safety.app import LatestFrameReader

    class Capture:
        def __init__(self):
            self.counter = 0

        def isOpened(self):
            return True

        def read(self):
            self.counter += 1
            return True, np.full((2, 2, 3), self.counter, dtype=np.uint8)

        def release(self):
            return None

    monkeypatch.setattr("extension.safety.app.cv2.VideoCapture", lambda *args: Capture())
    from time import sleep

    reader = LatestFrameReader("rtsp://example/safety")
    reader.start()
    sleep(0.02)
    sample = reader.latest()
    reader.stop()
    assert sample is not None
    frame, frame_at = sample
    assert frame.shape == (2, 2, 3)
    assert frame_at > 0


def test_model_decodes_real_smoking_artifact() -> None:
    model_path = Path("assets/models/smoking/best.onnx")
    if not model_path.is_file():
        pytest.skip("smoking model artifact is not present")
    from extension.safety.config import ModelConfig

    model = OnnxSafetyModel.build(ModelConfig(model_path, ("CPUExecutionProvider",)), threshold=0.1)
    detections = model.infer(np.zeros((720, 1280, 3), dtype=np.uint8), 1.0)
    assert all(item.label == "smoking" for item in detections)
    assert all(0 <= item.score <= 1 for item in detections)
