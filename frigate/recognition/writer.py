"""Bounded non-blocking JSONL and image evidence writer."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceJob:
    record: Mapping[str, Any]
    image_name: str | None = None
    image: object | None = None


class BoundedTraceWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        capacity: int = 64,
        manifest_name: str = "recognition.jsonl",
        copy_image: Callable[[object], object] | None = None,
        encode_jpeg: Callable[[object], bytes] | None = None,
        max_artifact_bytes: int | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._manifest_name = manifest_name
        self._queue: queue.Queue[TraceJob | None] = queue.Queue(maxsize=capacity)
        self._copy_image = copy_image or (lambda image: image)
        self._encode_jpeg = encode_jpeg
        self._max_artifact_bytes = max_artifact_bytes
        self._thread = threading.Thread(target=self._run, name="recognition-trace", daemon=True)
        self._started = False
        self._closed = False
        self.dropped = 0
        self.errors = 0
        self.artifact_bytes = 0

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._started:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._thread.start()

    def submit(
        self,
        record: Mapping[str, Any],
        *,
        image_name: str | None = None,
        image: object | None = None,
    ) -> bool:
        if self._closed:
            return False
        if not self._started:
            self.start()
        copied = self._copy_image(image) if image is not None else None
        try:
            self._queue.put_nowait(TraceJob(dict(record), image_name, copied))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def flush(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout: float = 2.0) -> bool:
        if self._closed:
            return self.flush(timeout)
        self._closed = True
        if not self._started:
            return True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            if not self.flush(timeout):
                return False
            self._queue.put_nowait(None)
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        manifest = self._output_dir / self._manifest_name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as output:
            while True:
                job = self._queue.get()
                try:
                    if job is None:
                        output.flush()
                        return
                    record = dict(job.record)
                    if job.image is not None and job.image_name and self._encode_jpeg:
                        encoded = self._encode_jpeg(job.image)
                        if (
                            self._max_artifact_bytes is not None
                            and self.artifact_bytes + len(encoded)
                            > self._max_artifact_bytes
                        ):
                            record.pop("artifact_path", None)
                            record["artifact_rejected"] = "byte_limit"
                        else:
                            target = self._output_dir / job.image_name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(encoded)
                            self.artifact_bytes += len(encoded)
                            record.update(
                                {
                                    "artifact_path": Path(job.image_name).as_posix(),
                                    "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
                                    "artifact_bytes": len(encoded),
                                    "image_shape": [
                                        int(value)
                                        for value in getattr(job.image, "shape", ())
                                    ],
                                }
                            )
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                except (OSError, TypeError, ValueError):
                    self.errors += 1
                finally:
                    self._queue.task_done()
