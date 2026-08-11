"""Canonical recognition configuration fingerprint."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from frigate.config import FrigateConfig


def canonical_config_json(config: FrigateConfig) -> str:
    """Serialize only settings that affect recognition decisions or capacity."""

    def value(model: Any, *, exclude: Any = None) -> Any:
        return json.loads(model.model_dump_json(exclude=exclude))

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
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_fingerprint(config_json: str) -> str:
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()
