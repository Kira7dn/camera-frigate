"""Local development runtime input selector."""

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(prefix="/runtime/input", tags=["Runtime input"])
CAMERAS = ["face_camera", "car_camera", "safety_camera"]
STATE_PATH = Path("/config/runtime-input.json")


def _persist(mock: bool) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({"mode": "mock" if mock else "rtsp"}), encoding="utf-8")
    temporary.replace(STATE_PATH)


def _state(request: Request) -> dict[str, object]:
    maintainer = request.app.camera_maintainer
    return {
        "inputs": {
            camera: "mock" if maintainer.camera_metrics[camera].runtime_input.value else "rtsp"
            for camera in CAMERAS
            if camera in maintainer.camera_metrics
        }
    }


@router.get("")
def get_runtime_input(request: Request) -> dict[str, object]:
    return _state(request)


@router.post("/start")
def start_runtime_input(request: Request) -> dict[str, object]:
    request.app.camera_maintainer.set_runtime_input(CAMERAS, True)
    _persist(True)
    return {"success": True, **_state(request)}


@router.post("/stop")
def stop_runtime_input(request: Request) -> dict[str, object]:
    request.app.camera_maintainer.set_runtime_input(CAMERAS, False)
    _persist(False)
    return {"success": True, **_state(request)}
