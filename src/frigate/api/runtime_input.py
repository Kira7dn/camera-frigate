"""Switch camera inputs directly through the canonical Frigate config."""

from __future__ import annotations

import logging
import os
import threading
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


def _switch_mode(app, mode: str) -> None:
    previous_mode = str(app.frigate_config.runtime.input_mode)
    if previous_mode == mode:
        return
    try:
        for camera in CAMERAS:
            _set_go2rtc(app, camera, mode)
        _write_mode(app, mode)
    except Exception:
        logger.exception("Runtime input switch to %s failed; restoring %s", mode, previous_mode)
        for camera in CAMERAS:
            try:
                _set_go2rtc(app, camera, previous_mode)
            except Exception:
                logger.exception("Could not restore go2rtc stream %s", camera)
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
    return {"inputs": {camera: mode for camera in CAMERAS}}


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
    try:
        with _runtime_lock:
            _switch_mode(request.app, "mock")
            _arm_recovery(request.app)
            return {"success": True, **_state(request.app)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop")
def stop_runtime_input(request: Request) -> dict[str, object]:
    try:
        with _runtime_lock:
            _cancel_recovery()
            _switch_mode(request.app, "rtsp")
            return {"success": True, **_state(request.app)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
