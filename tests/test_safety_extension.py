"""Isolated contracts for the standalone Safety extension."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import requests
from extension.safety.config import SafetyConfigError, load_config, validate_camera_keys
from extension.safety.events import (
    FrigateEventClient,
    HazardDecision,
    SafetyEventError,
    TemporalGate,
)
from extension.safety.inference import Detection, OnnxSafetyModel


def _config(tmp_path: Path, *, threshold: float = 0.1) -> Path:
    model = tmp_path / "best.onnx"
    model.write_bytes(b"test")
    path = tmp_path / "safety.yaml"
    path.write_text(
        f"""frigate_url: http://frigate:5000
restream_url: rtsp://frigate:8554
model:
  path: {model.as_posix()}
  providers: [CPUExecutionProvider]
cameras:
  cam:
    stream: cam
    inference_fps: 1
    labels:
      smoking: {{enabled: true, threshold: {threshold}}}
    confirm_seconds: 1
    clear_seconds: 2
""",
        encoding="utf-8",
    )
    return path


def _detection(score: float = 0.8, observed_at: float = 1.0) -> Detection:
    return Detection("smoking", score, (0.1, 0.1, 0.2, 0.2), observed_at)


def test_config_rejects_unknown_fields_and_camera_mismatch(tmp_path: Path) -> None:
    path = _config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "unexpected: true\n", encoding="utf-8")
    with pytest.raises(SafetyConfigError, match="unknown field"):
        load_config(path)

    config = load_config(_config(tmp_path))
    with pytest.raises(SafetyConfigError, match="missing"):
        validate_camera_keys(config, {"other"})


def test_temporal_gate_requires_confirm_and_clear_time(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection()], 0.0) == []
    assert gate.observe("cam", [_detection()], 0.9) == []
    active = gate.observe("cam", [_detection()], 1.1)
    assert active == [HazardDecision("cam", "smoking", True, 0.8, (0.1, 0.1, 0.2, 0.2))]
    assert gate.observe("cam", [], 2.9) == []
    assert gate.observe("cam", [], 4.9) == [HazardDecision("cam", "smoking", False, 0.0, None)]


def test_smoking_below_threshold_does_not_open_event(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, threshold=0.5))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection(score=0.49)], 0.0) == []
    assert gate.observe("cam", [_detection(score=0.49)], 2.0) == []


def test_smoking_single_frame_does_not_open_event(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    gate = TemporalGate(dict(config.cameras))
    assert gate.observe("cam", [_detection()], 0.0) == []
    assert gate.observe("cam", [], 0.1) == []
    assert gate.observe("cam", [_detection()], 1.0) == []


class _Response:
    def __init__(self, payload, status: int = 200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("latest.jpg"):
            return _Response(b"jpg", headers={"X-Frame-Time": "10"})
        return _Response([])

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _Response({"success": True, "event_id": "server-event"})

    def put(self, url, **kwargs):
        self.calls.append(("PUT", url, kwargs))
        return _Response({"success": True})


def test_event_client_uses_server_id_and_safety_sub_label() -> None:
    session = _Session()
    client = FrigateEventClient("http://frigate:5000", session=session)
    decision = HazardDecision("cam", "smoking", True, 0.91, (0.1, 0.2, 0.4, 0.5))
    assert client.probe_camera("cam")
    assert client.create_event(decision) == "server-event"
    client.apply(HazardDecision("cam", "smoking", False, 0, None))
    post = next(call for call in session.calls if call[0] == "POST")
    assert post[1].endswith("/api/events/cam/smoking/create")
    assert post[2]["json"]["sub_label"] == "camera-safety"
    assert post[2]["json"]["draw"]["boxes"][0]["box"] == [0.1, 0.2, 0.4, 0.5]
    put = next(call for call in session.calls if call[0] == "PUT")
    assert put[1].endswith("/api/events/server-event/end")


class _TimeoutSession(_Session):
    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        raise requests.Timeout("create response lost")

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("latest.jpg"):
            return _Response(b"jpg", headers={"X-Frame-Time": "10"})
        return _Response([{"id": "reconciled-event", "sub_label": "camera-safety"}])


def test_event_create_timeout_reconciles_existing_smoking_event() -> None:
    session = _TimeoutSession()
    client = FrigateEventClient("http://frigate:5000", session=session)
    decision = HazardDecision("cam", "smoking", True, 0.91, None)
    assert client.create_event(decision) == "reconciled-event"
    assert client.active[("cam", "smoking")] == "reconciled-event"
    assert len([call for call in session.calls if call[0] == "POST"]) == 1


def test_event_api_error_fails_closed_without_active_event() -> None:
    class ErrorSession(_Session):
        def post(self, url, **kwargs):
            raise requests.ConnectionError("Frigate unavailable")

    client = FrigateEventClient("http://frigate:5000", session=ErrorSession())
    with pytest.raises(SafetyEventError):
        client.create_event(HazardDecision("cam", "smoking", True, 0.9, None))
    assert client.active == {}


def test_reconcile_ends_only_camera_safety_smoking_events() -> None:
    class ReconcileSession(_Session):
        def get(self, url, **kwargs):
            self.calls.append(("GET", url, kwargs))
            return _Response(
                [
                    {"id": "safety-event", "sub_label": "camera-safety"},
                    {"id": "other-event", "sub_label": "other-producer"},
                ]
            )

    session = ReconcileSession()
    client = FrigateEventClient("http://frigate:5000", session=session)
    client.reconcile([("cam", "smoking")])
    puts = [call for call in session.calls if call[0] == "PUT"]
    assert [call[1] for call in puts] == ["http://frigate:5000/api/events/safety-event/end"]


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
