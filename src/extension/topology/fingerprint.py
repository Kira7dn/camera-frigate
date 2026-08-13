"""Shared canonical serialization and fingerprint primitives for runtimes."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def model_value(model: Any, *, exclude: Any = None) -> Any:
    """Convert a Frigate config model to JSON-compatible canonical data."""
    return json.loads(model.model_dump_json(exclude=exclude))


def canonical_json(payload: Any) -> str:
    """Serialize runtime contract data deterministically as UTF-8 JSON."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(canonical_payload: str) -> str:
    """Hash one canonical runtime contract payload."""
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
