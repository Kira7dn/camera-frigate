"""Opt-in passage funnel trace, disabled unless explicitly configured."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()


def passage_trace(stage: str, *, camera: str, frame_time: float | None = None, track_id: str | None = None, generation: int | None = None, **fields: Any) -> None:
    path = os.environ.get("PASSAGE_TRACE_PATH")
    if not path:
        return
    record = {"stage": stage, "camera": camera, "frame_time": frame_time, "trace_time": time.time(), "track_id": track_id, "generation": generation, **fields}
    with _LOCK, open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
