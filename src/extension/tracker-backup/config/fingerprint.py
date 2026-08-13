"""Canonical configuration view for one tracker node."""

from __future__ import annotations

from typing import Any

from extension.topology.fingerprint import canonical_json, fingerprint, model_value
from frigate.infrastructure.config import FrigateConfig


def _value(model: Any, *, exclude: Any = None) -> Any:
    return model_value(model, exclude=exclude)


def canonical_tracker_config_json(config: FrigateConfig, node_id: str) -> str:
    """Serialize only settings consumed by one edge-owned camera runtime."""
    node = config.tracker[node_id]
    cameras = {
        name: _value(config.cameras[name])
        for name in sorted(node.cameras)
    }
    payload = {
        "node_id": node_id,
        # Credential mount paths intentionally differ between Frigate main and
        # the isolated tracker container. They are deployment details, not a
        # behavioral topology revision. Keep the peer identity in the contract
        # while excluding all credential locations (including the private key).
        "node": _value(node, exclude={"tls"}),
        "tls_server_name": node.tls.server_name,
        "model": _value(config.model),
        "detectors": {
            name: _value(detector)
            for name, detector in sorted(config.detectors.items())
        },
        "ffmpeg": _value(config.ffmpeg),
        "objects": _value(config.objects),
        "cameras": cameras,
    }
    return canonical_json(payload)


def tracker_config_fingerprint(config: FrigateConfig, node_id: str) -> str:
    return fingerprint(canonical_tracker_config_json(config, node_id))
