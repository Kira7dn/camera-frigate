"""Executable process for the optional camera-safety service."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .config import SafetyConfigError, load_config, resolve_stream_url
from .events import FrigateEventClient, HazardDecision, SafetyEventError, TemporalGate
from .inference import OnnxSafetyModel

logger = logging.getLogger(__name__)
HEALTH_PATH = Path("/tmp/camera-safety-health.json")


@dataclass
class HealthState:
    model: bool = False
    source: bool = False
    frigate_api: bool = False
    last_frame_at: float = 0.0
    last_inference_at: float = 0.0
    error: str = ""
    inference_count: int = 0
    detection_count: int = 0
    last_detection_label: str = ""
    last_detection_score: float = 0.0
    last_detection_bbox: tuple[float, float, float, float] | None = None
    last_detection_at: float = 0.0
    active_decision_count: int = 0
    clear_decision_count: int = 0
    event_create_attempts: int = 0
    event_create_successes: int = 0
    event_end_attempts: int = 0
    event_end_successes: int = 0
    event_sync_failures: int = 0
    last_event_id: str = ""

    @property
    def ready(self) -> bool:
        now = time.time()
        return self.model and self.source and self.frigate_api and now - self.last_frame_at < 15


class LatestFrameReader:
    """A single latest-frame slot; old frames are dropped instead of queued."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._capture: cv2.VideoCapture | None = None
        self._frame: object = None
        self._frame_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="safety-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._capture = None
                self._stop.wait(1.0)
                continue
            try:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    with self._lock:
                        self._frame = frame
                        self._frame_at = time.time()
            finally:
                capture.release()
                self._capture = None
            self._stop.wait(0.5)

    def latest(self) -> tuple[object, float] | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._frame_at

    def stop(self) -> None:
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
        if self._thread is not None:
            self._thread.join(timeout=3)


class CameraWorker:
    def __init__(
        self,
        camera: str,
        config,
        model: OnnxSafetyModel,
        events: FrigateEventClient,
        health: HealthState,
        stop_event: threading.Event,
    ) -> None:
        self.camera = camera
        self.config = config
        self.model = model
        self.events = events
        self.health = health
        self.stop_event = stop_event
        self.reader = LatestFrameReader(resolve_stream_url(config, camera))
        self.gate = TemporalGate({camera: config.cameras[camera]})
        self.pending: dict[tuple[str, str], HazardDecision] = {}

    def run(self) -> None:
        self.reader.start()
        interval = 1.0 / self.config.cameras[self.camera].inference_fps
        try:
            while not self.stop_event.is_set():
                sample = self.reader.latest()
                if sample is None:
                    self.health.source = False
                    self.stop_event.wait(0.1)
                    continue
                frame, frame_at = sample
                self.health.source = True
                self.health.last_frame_at = frame_at
                try:
                    detections = self.model.infer(frame, frame_at)
                    self.health.last_inference_at = time.time()
                    self.health.inference_count += 1
                    self.health.detection_count += len(detections)
                    if detections:
                        detection = max(detections, key=lambda item: item.score)
                        self.health.last_detection_label = detection.label
                        self.health.last_detection_score = detection.score
                        self.health.last_detection_bbox = detection.bbox
                        self.health.last_detection_at = detection.observed_at
                    decisions = self.gate.observe(self.camera, detections, time.monotonic())
                    for decision in decisions:
                        self.pending[(decision.camera, decision.label)] = decision
                    for key, decision in list(self.pending.items()):
                        try:
                            if decision.active:
                                self.health.active_decision_count += 1
                                self.health.event_create_attempts += 1
                            else:
                                self.health.clear_decision_count += 1
                                self.health.event_end_attempts += 1
                            self.events.apply(decision)
                            event_id = self.events.active.get(key, "")
                            if event_id:
                                self.health.last_event_id = event_id
                            if decision.active:
                                self.health.event_create_successes += 1
                            else:
                                self.health.event_end_successes += 1
                            self.pending.pop(key, None)
                        except SafetyEventError as exc:
                            self.health.event_sync_failures += 1
                            self.health.error = str(exc)
                            logger.warning("Safety Event sync failed: %s", exc)
                except Exception as exc:  # model/runtime errors are degraded, not negative safety decisions
                    self.health.error = str(exc)
                    logger.exception("Safety inference failed")
                self.stop_event.wait(interval)
        finally:
            self.reader.stop()


class SafetyApplication:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        self.health = HealthState()
        self.stop_event = threading.Event()
        self.events = FrigateEventClient(self.config.frigate_url)
        self.model = None
        self.workers: list[CameraWorker] = []

    def start(self) -> None:
        self.model = OnnxSafetyModel.build(self.config.model)
        self.health.model = True
        labels = [(camera, label) for camera, policy in self.config.cameras.items() for label in policy.labels]
        last_error: SafetyEventError | None = None
        for _ in range(30):
            try:
                self.events.reconcile(labels)
                last_error = None
                break
            except SafetyEventError as exc:
                last_error = exc
                time.sleep(1)
        if last_error is not None:
            raise last_error
        self.health.frigate_api = True
        for camera in self.config.cameras:
            worker = CameraWorker(camera, self.config, self.model, self.events, self.health, self.stop_event)
            self.workers.append(worker)
            thread = threading.Thread(target=worker.run, name=f"safety-{camera}", daemon=True)
            thread.start()
        self._write_health()

    def _write_health(self) -> None:
        HEALTH_PATH.write_text(json.dumps({**asdict(self.health), "ready": self.health.ready}), encoding="utf-8")

    def run(self) -> None:
        self.start()
        while not self.stop_event.wait(1):
            self._write_health()

    def stop(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.reader.stop()
        self._write_health()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera Safety runtime")
    parser.add_argument("command", choices=("run", "validate", "healthcheck"), nargs="?", default="run")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _arguments()
    try:
        if args.command == "validate":
            config = load_config(args.config)
            logger.info("Safety config valid for cameras: %s", ", ".join(config.cameras))
            return 0
        if args.command == "healthcheck":
            if not HEALTH_PATH.is_file():
                return 1
            health = json.loads(HEALTH_PATH.read_text(encoding="utf-8"))
            return 0 if health.get("ready") else 1
        app = SafetyApplication(args.config)
        for name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), lambda *_: app.stop())
        app.run()
        return 0
    except (SafetyConfigError, SafetyEventError, OSError, ValueError) as exc:
        logger.error("Safety startup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
