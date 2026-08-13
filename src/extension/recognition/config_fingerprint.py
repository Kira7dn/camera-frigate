"""Canonical recognition configuration fingerprint."""

from __future__ import annotations

from typing import Any

from extension.topology.fingerprint import canonical_json, fingerprint, model_value
from frigate.infrastructure.config import FrigateConfig


def canonical_config_json(config: FrigateConfig) -> str:
    """Serialize only settings that affect recognition decisions or capacity."""

    def value(model: Any, *, exclude: Any = None) -> Any:
        return model_value(model, exclude=exclude)

    recognition = value(config.recognition, exclude={"tls": {"key"}})
    payload = {
        "recognition": recognition,
        "face_recognition": value(config.face_recognition),
        "lpr": value(config.lpr),
        "objects": {
            "all_objects": sorted(config.objects.all_objects),
        },
        "cameras": {
            name: {
                "face_recognition": value(camera.face_recognition),
                "lpr": value(camera.lpr),
                "detect": value(camera.detect),
            }
            for name, camera in sorted(config.cameras.items())
        },
    }
    return canonical_json(payload)


def config_fingerprint(config_json: str) -> str:
    return fingerprint(config_json)
