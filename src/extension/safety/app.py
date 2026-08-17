"""Executable process for the optional camera-safety service."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import yaml

from .config import SafetyConfigError, load_config, resolve_stream_url
from .events import (
    HazardDecision,
    SafetyEventError,
    SafetyMediaStore,
    SafetyProducer,
    TemporalGate,
)
from .inference import OnnxSafetyModel

logger = logging.getLogger(__name__)
HEALTH_PATH = Path("/tmp/camera-safety-health.json")
CONFIG_PATH = Path("/config/config.yml")


def _runtime_config() -> dict:
    try:
        value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


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
    source_mode: str = "rtsp"

    @property
    def ready(self) -> bool:
        now = time.time()
        return self.model and self.source and self.frigate_api and now - self.last_frame_at < 15


class LatestFrameReader:
    """Latest frame plus a bounded producer-owned rolling evidence buffer."""

    def __init__(self, url: str, mock_url: str | None = None) -> None:
        self.url = url
        self.mock_url = mock_url
        self._capture: cv2.VideoCapture | None = None
        self._frame: object = None
        self._frame_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: list[tuple[float, object]] = []
        self._last_history_at = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="safety-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            mode = (_runtime_config().get("runtime") or {}).get("input_mode", "rtsp")
            url = self.mock_url if mode == "mock" else self.url
            if not url:
                self._stop.wait(1.0)
                continue
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self._capture = capture
            if not capture.isOpened():
                capture.release()
                self._capture = None
                self._stop.wait(1.0)
                continue
            next_mode_check = 0.0
            try:
                while not self._stop.is_set():
                    now = time.monotonic()
                    if now >= next_mode_check:
                        current_mode = (
                            (_runtime_config().get("runtime") or {}).get(
                                "input_mode", "rtsp"
                            )
                        )
                        if current_mode != mode:
                            break
                        next_mode_check = now + 0.25
                    ok, frame = capture.read()
                    if not ok:
                        break
                    with self._lock:
                        self._frame = frame
                        self._frame_at = time.time()
                        if self._frame_at - self._last_history_at >= 0.2:
                            self._history.append((self._frame_at, frame.copy()))
                            self._last_history_at = self._frame_at
                        if len(self._history) > 120:
                            del self._history[:-120]
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

    def history(self) -> list[tuple[float, object]]:
        with self._lock:
            return [(timestamp, frame.copy()) for timestamp, frame in self._history]


class CameraWorker:
    def __init__(
        self,
        camera: str,
        config,
        model: OnnxSafetyModel,
        events: SafetyProducer,
        health: HealthState,
        stop_event: threading.Event,
    ) -> None:
        self.camera = camera
        self.config = config
        self.model = model
        self.events = events
        self.health = health
        self.stop_event = stop_event
        mock_url = (
            (_runtime_config().get("runtime") or {}).get("mock_sources") or {}
        ).get(camera)
        self.reader = LatestFrameReader(resolve_stream_url(config, camera), mock_url)
        self.media = SafetyMediaStore()
        self.gate = TemporalGate({camera: config.cameras[camera]})
        self.pending: dict[
            tuple[str, str],
            tuple[HazardDecision, object, float, list[tuple[float, object]]],
        ] = {}
        self.queued: set[tuple[str, str]] = set()
        self.publish_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=32)
        self.publish_stop = threading.Event()
        self.publish_thread: threading.Thread | None = None
        self.live_pending: tuple[HazardDecision, tuple[int, int], float] | None = None
        self.live_queued = False

    def _queue_pending(self) -> None:
        for key in tuple(self.pending):
            if key in self.queued:
                continue
            try:
                self.publish_queue.put_nowait(("event", key))
            except queue.Full:
                return
            self.queued.add(key)

    def _queue_live(self) -> None:
        if self.live_pending is None or self.live_queued:
            return
        try:
            self.publish_queue.put_nowait(("live", None))
        except queue.Full:
            return
        self.live_queued = True

    def _publish_loop(self) -> None:
        while not self.publish_stop.is_set() or not self.publish_queue.empty():
            try:
                kind, value = self.publish_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "live":
                    job = self.live_pending
                    if job is not None:
                        decision, frame_shape, frame_at = job
                        self.events.publish_live(decision, frame_shape, frame_at)
                        if self.live_pending is job:
                            self.live_pending = None
                    continue
                key = value
                job = self.pending.get(key)
                if job is None:
                    continue
                decision, frame, frame_at, frames = job
                if decision.active:
                    self.health.active_decision_count += 1
                    self.health.event_create_attempts += 1
                else:
                    self.health.clear_decision_count += 1
                    self.health.event_end_attempts += 1
                event_id = self.events.publish(
                    decision, frame, frame_at, self.media, frames
                )
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
            finally:
                if kind == "live":
                    self.live_queued = False
                else:
                    self.queued.discard(value)
                self.publish_queue.task_done()

    def run(self) -> None:
        self.reader.start()
        self.publish_thread = threading.Thread(
            target=self._publish_loop,
            name=f"safety-publisher-{self.camera}",
            daemon=True,
        )
        self.publish_thread.start()
        interval = 1.0 / self.config.cameras[self.camera].inference_fps
        try:
            while not self.stop_event.is_set():
                self.health.source_mode = (
                    (_runtime_config().get("runtime") or {}).get("input_mode", "rtsp")
                )
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
                    live_detection = max(
                        (
                            item
                            for item in detections
                            if item.label in self.config.cameras[self.camera].labels
                            and self.config.cameras[self.camera].labels[item.label].enabled
                            and item.score
                            >= self.config.cameras[self.camera].labels[item.label].threshold
                        ),
                        key=lambda item: item.score,
                        default=None,
                    )
                    if live_detection is not None:
                        self.live_pending = (
                            HazardDecision(
                                self.camera,
                                live_detection.label,
                                True,
                                live_detection.score,
                                live_detection.bbox,
                            ),
                            tuple(frame.shape[:2]),
                            frame_at,
                        )
                        self._queue_live()
                    decisions = self.gate.observe(self.camera, detections, time.monotonic())
                    for decision in decisions:
                        key = (decision.camera, decision.label)
                        self.pending[key] = (
                            decision,
                            frame.copy(),
                            frame_at,
                            self.reader.history(),
                        )
                    self._queue_pending()
                except Exception as exc:  # model/runtime errors are degraded, not negative safety decisions
                    self.health.error = str(exc)
                    logger.exception("Safety inference failed")
                self._queue_pending()
                self.stop_event.wait(interval)
        finally:
            self.reader.stop()
            self.publish_stop.set()
            if self.publish_thread is not None:
                self.publish_thread.join(timeout=12)


class SafetyApplication:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        self.health = HealthState()
        self.stop_event = threading.Event()
        self.events = SafetyProducer(self.config.grpc_url)
        self.model = None
        self.workers: list[CameraWorker] = []
        self.control_server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        self.model = OnnxSafetyModel.build(self.config.model)
        self.health.model = True
        last_error: SafetyEventError | None = None
        for _ in range(30):
            try:
                if not self.events.ready():
                    raise SafetyEventError("producer ingress is not ready")
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
        self.events.close()
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
