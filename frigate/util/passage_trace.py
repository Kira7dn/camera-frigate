"""Opt-in passage funnel trace, disabled unless explicitly configured."""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from frigate.recognition.writer import BoundedTraceWriter

_LOCK = threading.Lock()
_EVIDENCE_LOCK = threading.Lock()
_EVIDENCE_SEQUENCE = 0
_EVIDENCE_LAST_CAPTURE: dict[tuple[str, str], float] = {}
_WRITERS: dict[str, BoundedTraceWriter] = {}


def _encode_jpeg(image: Any) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    )
    if not ok:
        raise ValueError("jpeg_encode_failed")
    return encoded.tobytes()


def _writer(
    path: Path, *, images: bool = False, output_dir: Path | None = None
) -> BoundedTraceWriter:
    key = str(path.resolve())
    with _LOCK:
        writer = _WRITERS.get(key)
        if writer is None:
            base = output_dir or path.parent
            writer = BoundedTraceWriter(
                base,
                capacity=int(os.environ.get("PASSAGE_WRITER_QUEUE_CAPACITY", "64")),
                manifest_name=path.relative_to(base).as_posix(),
                copy_image=lambda image: image.copy(),
                encode_jpeg=_encode_jpeg if images else None,
                max_artifact_bytes=(
                    int(
                        os.environ.get(
                            "PASSAGE_EVIDENCE_MAX_BYTES", "134217728"
                        )
                    )
                    if images
                    else None
                ),
            )
            _WRITERS[key] = writer
        return writer


def passage_writer_stats() -> dict[str, int]:
    return {
        "depth": sum(writer.depth for writer in _WRITERS.values()),
        "drops": sum(writer.dropped for writer in _WRITERS.values()),
        "errors": sum(writer.errors for writer in _WRITERS.values()),
    }


def shutdown_passage_writers(timeout: float = 2.0) -> bool:
    writers = list(_WRITERS.values())
    if not writers:
        return True
    deadline = time.monotonic() + timeout
    complete = True
    for writer in writers:
        complete = writer.close(max(0.0, deadline - time.monotonic())) and complete
    _WRITERS.clear()
    return complete


atexit.register(shutdown_passage_writers)


def _capture_started(frame_time: float | None) -> bool:
    """Return whether the test harness has opened the active capture window."""
    start_path = os.environ.get("PASSAGE_CAPTURE_START_PATH")
    if not start_path:
        return True
    try:
        start = float(Path(start_path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return frame_time is None or float(frame_time) + 1e-9 >= start


def _past_capture_cutoff(frame_time: float | None) -> bool:
    """Stop acceptance capture after a validator-owned frame-time boundary."""
    cutoff_path = os.environ.get("PASSAGE_CAPTURE_CUTOFF_PATH")
    if not cutoff_path or frame_time is None:
        return False
    try:
        cutoff = float(Path(cutoff_path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return float(frame_time) > cutoff + 1e-9


def canonical_trace_id(
    pipeline: str,
    camera: str,
    identity: str | None,
    generation: int | None = None,
) -> str:
    """Return the producer-owned lifecycle identity used by runtime reports."""
    safe_pipeline = re.sub(r"[^A-Za-z0-9_.-]+", "_", pipeline or "unknown")
    safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera or "unknown")
    safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity or "unassigned")
    suffix = f":g{int(generation)}" if generation is not None else ""
    return f"{safe_pipeline}:{safe_camera}:{safe_identity}{suffix}"


def _derived_trace_id(
    stage: str,
    camera: str,
    frame_time: float | None,
    track_id: str | None,
    generation: int | None,
    fields: dict[str, Any],
) -> str:
    face_stages = {
        "first_qualified_face",
        "candidate_submitted",
        "recognition_candidate",
        "first_attempt",
        "confirmed_result",
    }
    pipeline = str(
        fields.get("task")
        or ("detector" if stage == "detector_hit" else "face" if stage in face_stages else "lpr")
    )
    identity = fields.get("recognition_passage_id") or fields.get("passage_id") or track_id
    if identity is None and stage == "detector_hit":
        raw = json.dumps(
            {"camera": camera, "frame_time": frame_time, "box": fields.get("object_box")},
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = f"observation-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"
    return canonical_trace_id(
        pipeline,
        camera,
        None if identity is None else str(identity),
        generation if pipeline == "face" else None,
    )


def passage_trace(
    stage: str,
    *,
    camera: str,
    frame_time: float | None = None,
    track_id: str | None = None,
    generation: int | None = None,
    trace_id: str | None = None,
    **fields: Any,
) -> None:
    path = os.environ.get("PASSAGE_TRACE_PATH")
    if not path or not _capture_started(frame_time) or _past_capture_cutoff(frame_time):
        return
    pipeline = str(
        fields.get("task")
        or ("detector" if stage == "detector_hit" else "face" if stage in {"first_qualified_face", "candidate_submitted", "recognition_candidate", "first_attempt", "confirmed_result"} else "lpr")
    )
    # The shared LPR mixin can observe every enabled camera.  Camera ownership
    # is authoritative for runtime artifact placement; face-camera records
    # must never become LPR traces merely because the shared stage defaulted to
    # lpr.
    if camera == "face_camera" and pipeline == "lpr":
        pipeline = "face"
    resolved_trace_id = trace_id
    if pipeline == "face" and resolved_trace_id and resolved_trace_id.startswith("lpr:"):
        resolved_trace_id = "face:" + resolved_trace_id[len("lpr:"):]
    record = {
        "stage": stage,
        "pipeline": pipeline,
        "trace_id": resolved_trace_id or _derived_trace_id(stage, camera, frame_time, track_id, generation, {**fields, "task": pipeline}),
        "camera": camera,
        "frame_time": frame_time,
        "source_pts": frame_time,
        "trace_time": time.time(),
        "track_id": track_id,
        "generation": generation,
        "run_id": os.environ.get("PASSAGE_RUN_ID") or None,
        **fields,
    }
    _writer(Path(path)).submit(record)


def passage_evidence_enabled() -> bool:
    """Return whether opt-in runtime image evidence is enabled."""
    return bool(os.environ.get("PASSAGE_EVIDENCE_DIR"))


def passage_evidence_should_capture(
    camera: str,
    track_id: str | None,
    frame_time: float,
) -> bool:
    """Sample acceptance evidence at the candidate-diversity cadence per track."""
    if (
        not passage_evidence_enabled()
        or not _capture_started(frame_time)
        or _past_capture_cutoff(frame_time)
    ):
        return False
    minimum_interval = float(
        os.environ.get("PASSAGE_EVIDENCE_MIN_INTERVAL_SECONDS", "0.4")
    )
    key = (camera, track_id or "none")
    with _EVIDENCE_LOCK:
        previous = _EVIDENCE_LAST_CAPTURE.get(key)
        if (
            previous is not None
            and frame_time >= previous
            and frame_time - previous + 1e-9 < minimum_interval
        ):
            return False
        _EVIDENCE_LAST_CAPTURE[key] = frame_time
        return True


def passage_evidence_id(
    camera: str, track_id: str | None, frame_time: float | None, nonce: int
) -> str:
    """Build a filesystem-safe, deterministic ID for one processor invocation."""
    raw = f"{camera}|{track_id or 'none'}|{frame_time}|{nonce}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    safe_track = re.sub(r"[^A-Za-z0-9_.-]+", "_", track_id or "none")[-48:]
    return f"{safe_track}-{digest}"


def passage_evidence(
    stage: str,
    *,
    evidence_id: str,
    camera: str,
    frame_time: float | None,
    track_id: str | None,
    trace_id: str | None = None,
    pipeline: str = "lpr",
    image: Any | None = None,
    image_index: int | None = None,
    **fields: Any,
) -> dict[str, Any] | None:
    """Persist bounded acceptance-only LPR evidence and its integrity metadata."""
    root_value = os.environ.get("PASSAGE_EVIDENCE_DIR")
    if (
        not root_value
        or not _capture_started(frame_time)
        or _past_capture_cutoff(frame_time)
    ):
        return None

    global _EVIDENCE_SEQUENCE
    root = Path(root_value)
    max_records = int(os.environ.get("PASSAGE_EVIDENCE_MAX_RECORDS", "4096"))

    with _EVIDENCE_LOCK:
        sequence = _EVIDENCE_SEQUENCE
        _EVIDENCE_SEQUENCE += 1
        if camera == "face_camera" and pipeline == "lpr":
            pipeline = "face"
        resolved_trace_id = trace_id
        if pipeline == "face" and resolved_trace_id and resolved_trace_id.startswith("lpr:"):
            resolved_trace_id = "face:" + resolved_trace_id[len("lpr:"):]
        record: dict[str, Any] = {
            "sequence": sequence,
            "stage": stage,
            "pipeline": pipeline,
            "trace_id": resolved_trace_id or canonical_trace_id(pipeline, camera, track_id),
            "evidence_id": evidence_id,
            "camera": camera,
            "frame_time": frame_time,
            "source_pts": frame_time,
            "track_id": track_id,
            "run_id": os.environ.get("PASSAGE_RUN_ID") or None,
            **fields,
        }
        if image_index is not None:
            record["image_index"] = image_index

        image_name: str | None = None
        if sequence >= max_records:
            record["artifact_rejected"] = "record_limit"
        elif image is not None:
            suffix = f"-{image_index:02d}" if image_index is not None else ""
            safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
            safe_trace = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_id or canonical_trace_id(pipeline, camera, track_id))
            relative = Path(pipeline) / safe_trace / evidence_id / f"{sequence:05d}-{safe_stage}{suffix}.jpg"
            image_name = relative.as_posix()
            record["artifact_path"] = image_name

        manifest = root / pipeline / "evidence.jsonl"
        writer = _writer(manifest, images=True, output_dir=root)
        accepted = writer.submit(
            record,
            image_name=image_name,
            image=image if image_name is not None else None,
        )
        if not accepted:
            record["artifact_rejected"] = "writer_queue_full"
        return record
