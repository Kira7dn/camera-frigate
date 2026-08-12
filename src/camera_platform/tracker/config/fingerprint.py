"""Canonical configuration view for one tracker node."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from frigate.infrastructure.config import FrigateConfig


def _value(model: Any, *, exclude: Any = None) -> Any:
    return json.loads(model.model_dump_json(exclude=exclude))


def canonical_tracker_config_json(config: FrigateConfig, node_id: str) -> str:
    """Serialize only settings consumed by one edge-owned camera runtime."""
    node = config.tracker[node_id]
    cameras = {
        name: _value(config.cameras[name])
        for name in sorted(node.cameras)
    }
    payload = {
        "node_id": node_id,
        "node": _value(node, exclude={"tls": {"key"}}),
        "model": _value(config.model),
        "detectors": {
            name: _value(detector)
            for name, detector in sorted(config.detectors.items())
        },
        "ffmpeg": _value(config.ffmpeg),
        "objects": _value(config.objects),
        "cameras": cameras,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tracker_config_fingerprint(config: FrigateConfig, node_id: str) -> str:
    return hashlib.sha256(
        canonical_tracker_config_json(config, node_id).encode("utf-8")
    ).hexdigest()
