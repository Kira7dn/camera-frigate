"""Switch camera inputs directly through the canonical Frigate config."""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import requests
from fastapi import APIRouter, HTTPException, Request
from ruamel.yaml import YAML

from frigate.api.config_util import swap_runtime_config
from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.config.env import substitute_frigate_vars
from frigate.util.config import find_config_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runtime/input", tags=["Runtime input"])
CAMERAS = ["face_camera", "car_camera", "safety_camera"]
_runtime_lock = threading.Lock()
_recovery_timer: threading.Timer | None = None
_mock_monitor_thread: threading.Thread | None = None
_mock_monitor_stop = threading.Event()
_last_stop_reason: str | None = None
_MOCK_EOF_DIR = "/config/runtime/mock-eof"


def _write_mode(app, mode: str) -> None:
    config_file = find_config_file()
    yaml = YAML()
    with open(config_file, encoding="utf-8") as stream:
        raw = yaml.load(stream)
    raw.setdefault("runtime", {})["input_mode"] = mode
    output = StringIO()
    yaml.dump(raw, output)
    content = output.getvalue()
    new_config = FrigateConfig.parse(content)
    # config_file is a writable bind mount. Replacing the mountpoint itself
    # is not portable across Docker/Windows, so update the canonical file in
    # place and flush it before publishing the parsed config object.
    with open(config_file, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    swap_runtime_config(app, new_config)


def _source(app, camera: str, mode: str) -> str:
    if mode == "mock":
        path = app.frigate_config.runtime.mock_sources.get(camera)
        if not path:
            raise RuntimeError(f"runtime.mock_sources.{camera} is missing")
        return f"ffmpeg:{path.replace(' ', '%20')}#video=h264"

    source = app.frigate_config.go2rtc.streams.get(camera)
    if isinstance(source, list):
        source = source[0] if source else None
    if not source:
        raise RuntimeError(f"go2rtc.streams.{camera} is missing")
    return substitute_frigate_vars(str(source))


def _set_go2rtc(app, camera: str, mode: str) -> None:
    response = requests.put(
        "http://127.0.0.1:1984/api/streams",
        params={"name": camera, "src": _source(app, camera, mode)},
        timeout=10,
    )
    response.raise_for_status()


def _set_all_go2rtc(app, mode: str) -> None:
    """Switch all camera sources concurrently and fail as one operation."""
    with ThreadPoolExecutor(max_workers=len(CAMERAS)) as executor:
        futures = [executor.submit(_set_go2rtc, app, camera, mode) for camera in CAMERAS]
        for future in futures:
            future.result()


def _mock_eof_cameras() -> list[str]:
    ended: list[str] = []
    for camera in CAMERAS:
        if os.path.isfile(os.path.join(_MOCK_EOF_DIR, f"{camera}.end")):
            ended.append(camera)
    return ended


def _clear_mock_eof() -> None:
    for camera in CAMERAS:
        try:
            os.unlink(os.path.join(_MOCK_EOF_DIR, f"{camera}.end"))
        except FileNotFoundError:
            continue


def _cancel_mock_monitor() -> None:
    global _mock_monitor_thread
    _mock_monitor_stop.set()
    _mock_monitor_thread = None


def _monitor_mock_sources(app, stop_event: threading.Event) -> None:
    global _last_stop_reason
    while not stop_event.wait(0.5):
        with _runtime_lock:
            if str(app.frigate_config.runtime.input_mode) != "mock":
                return
            ended = _mock_eof_cameras()
            if not ended:
                continue
            _last_stop_reason = f"mock_source_eof:{','.join(ended)}"
            _cancel_recovery()
            _cancel_mock_monitor()
            try:
                _switch_mode(app, "rtsp")
                logger.info(
                    "Mock source EOF on %s; restored all runtime inputs to rtsp",
                    ",".join(ended),
                )
            except Exception:
                logger.exception("Automatic mock EOF recovery failed")
            return


def _start_mock_monitor(app) -> None:
    global _mock_monitor_thread, _mock_monitor_stop
    _cancel_mock_monitor()
    _mock_monitor_stop = threading.Event()
    _mock_monitor_thread = threading.Thread(
        target=_monitor_mock_sources,
        args=(app, _mock_monitor_stop),
        name="runtime-mock-eof-monitor",
        daemon=True,
    )
    _mock_monitor_thread.start()


def _switch_mode(app, mode: str) -> None:
    previous_mode = str(app.frigate_config.runtime.input_mode)
    try:
        _set_all_go2rtc(app, mode)
        _write_mode(app, mode)
    except Exception:
        logger.exception("Runtime input switch to %s failed; restoring %s", mode, previous_mode)
        try:
            _set_all_go2rtc(app, previous_mode)
        except Exception:
            logger.exception("Could not restore all go2rtc streams")
        raise


def _cancel_recovery() -> None:
    global _recovery_timer
    if _recovery_timer is not None:
        _recovery_timer.cancel()
        _recovery_timer = None


def _restore_after_timeout(app) -> None:
    global _recovery_timer
    with _runtime_lock:
        _recovery_timer = None
        try:
            _switch_mode(app, "rtsp")
        except Exception:
            logger.exception("Automatic runtime input recovery failed")


def _arm_recovery(app) -> None:
    global _recovery_timer
    _cancel_recovery()
    seconds = max(30, int(getattr(app.frigate_config.runtime, "test_timeout_seconds", 300)))
    _recovery_timer = threading.Timer(seconds, _restore_after_timeout, args=(app,))
    _recovery_timer.daemon = True
    _recovery_timer.start()


def _state(app) -> dict[str, object]:
    mode = app.frigate_config.runtime.input_mode
    return {
        "inputs": {camera: mode for camera in CAMERAS},
        "reason": _last_stop_reason,
    }


def recover_stale_runtime_input(config: FrigateConfig) -> FrigateConfig:
    """Never boot Frigate in an abandoned test-input mode."""
    if config.runtime.input_mode != "mock":
        return config
    config_file = find_config_file()
    yaml = YAML()
    with open(config_file, encoding="utf-8") as stream:
        raw = yaml.load(stream)
    raw.setdefault("runtime", {})["input_mode"] = "rtsp"
    output = StringIO()
    yaml.dump(raw, output)
    content = output.getvalue()
    with open(config_file, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    logger.warning("Recovered stale mock input mode to rtsp during startup")
    return FrigateConfig.parse(content)


@router.get("")
def get_runtime_input(request: Request) -> dict[str, object]:
    return _state(request.app)


@router.post("/start")
def start_runtime_input(request: Request) -> dict[str, object]:
    global _last_stop_reason
    try:
        with _runtime_lock:
            _clear_mock_eof()
            _last_stop_reason = None
            _switch_mode(request.app, "mock")
            _arm_recovery(request.app)
            _start_mock_monitor(request.app)
            return {"success": True, **_state(request.app)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop")
def stop_runtime_input(request: Request) -> dict[str, object]:
    global _last_stop_reason
    try:
        with _runtime_lock:
            _cancel_recovery()
            _cancel_mock_monitor()
            _switch_mode(request.app, "rtsp")
            _last_stop_reason = "manual"
            return {"success": True, **_state(request.app)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
