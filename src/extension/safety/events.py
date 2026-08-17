"""Temporal Safety decisions and producer-owned media lifecycle."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from extension.tracker.runtime import (
    BoundingBox,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
)
from extension.tracker.transport import ProducerClient, ProducerTransportError

from .config import CameraSafetyConfig
from .inference import Detection


@dataclass(frozen=True)
class HazardDecision:
    camera: str
    label: str
    active: bool
    score: float
    bbox: tuple[float, float, float, float] | None


class _State(Enum):
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"


@dataclass
class _Observation:
    state: _State = _State.IDLE
    candidate_since: float | None = None
    clear_since: float | None = None
    last_score: float = 0.0
    last_bbox: tuple[float, float, float, float] | None = None


class TemporalGate:
    def __init__(self, policies: dict[str, CameraSafetyConfig]) -> None:
        self._policies = policies
        self._states: dict[tuple[str, str], _Observation] = {}

    def observe(self, camera: str, detections: Iterable[Detection], now: float) -> list[HazardDecision]:
        policy = self._policies[camera]
        by_label = {d.label: d for d in detections if d.label in policy.labels and policy.labels[d.label].enabled}
        decisions: list[HazardDecision] = []
        for label, label_policy in policy.labels.items():
            if not label_policy.enabled:
                continue
            state = self._states.setdefault((camera, label), _Observation())
            candidate = by_label.get(label)
            if candidate is not None and candidate.score >= label_policy.threshold:
                state.last_score, state.last_bbox, state.clear_since = candidate.score, candidate.bbox, None
                if state.state is _State.IDLE:
                    state.state, state.candidate_since = _State.PENDING, now
                if state.state is _State.PENDING and state.candidate_since is not None and now - state.candidate_since >= policy.confirm_seconds:
                    state.state = _State.ACTIVE
                    decisions.append(HazardDecision(camera, label, True, state.last_score, state.last_bbox))
            elif state.state is _State.PENDING:
                state.clear_since = now if state.clear_since is None else state.clear_since
                if now - state.clear_since >= policy.clear_seconds:
                    state.state, state.candidate_since, state.clear_since = (
                        _State.IDLE,
                        None,
                        None,
                    )
            elif state.state is _State.ACTIVE:
                state.clear_since = now if state.clear_since is None else state.clear_since
                if now - state.clear_since >= policy.clear_seconds:
                    state.state, state.clear_since = _State.IDLE, None
                    decisions.append(HazardDecision(camera, label, False, 0.0, None))
        return decisions

    def reset(self) -> None:
        self._states.clear()


class SafetyEventError(RuntimeError):
    """Safety producer evidence or gRPC synchronization failed."""


class SafetyMediaStore:
    """Encode real producer frames and a bounded MP4 terminal clip."""

    def __init__(self, max_frames: int = 120) -> None:
        self.max_frames = max_frames

    @staticmethod
    def _snapshot(frame: np.ndarray, decision: HazardDecision) -> bytes:
        image = frame.copy()
        if decision.bbox is None:
            raise SafetyEventError("safety_snapshot_requires_bbox")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = decision.bbox
        pixels = (
            max(0, min(width - 1, round(x1 * width))),
            max(0, min(height - 1, round(y1 * height))),
            max(0, min(width - 1, round(x2 * width))),
            max(0, min(height - 1, round(y2 * height))),
        )
        if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
            raise SafetyEventError("safety_snapshot_invalid_bbox")
        # The producer contract stores a raw full frame. Frigate's canonical
        # media renderer owns the single bbox/label overlay; drawing here too
        # would make every canonical notification contain two overlays.
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok or not encoded.tobytes():
            raise SafetyEventError("safety_snapshot_encode_failed")
        return encoded.tobytes()

    def snapshot(self, frame: np.ndarray, decision: HazardDecision) -> bytes:
        return self._snapshot(frame, decision)

    def clip(self, event_id: str, frames: list[tuple[float, np.ndarray]]) -> bytes:
        if not frames:
            raise SafetyEventError("safety_clip_requires_real_frames")
        frames = frames[-self.max_frames :]
        first = frames[0][1]
        height, width = first.shape[:2]
        path = Path("/tmp") / f"safety-{event_id}-{uuid.uuid4().hex}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height)
        )
        if not writer.isOpened():
            raise SafetyEventError("safety_clip_encoder_unavailable")
        try:
            for _, frame in frames:
                if frame.shape[:2] == (height, width):
                    writer.write(frame)
        finally:
            writer.release()
        try:
            content = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        if not content:
            raise SafetyEventError("safety_clip_encode_failed")
        return content


class SafetyProducer:
    """Publish Safety lifecycle and media through the shared producer contract."""

    def __init__(self, endpoint: str, node_id: str = "safety") -> None:
        self.client = ProducerClient(endpoint, node_id)
        self.node_id = node_id
        self.active: dict[tuple[str, str], str] = {}
        self.last_bbox: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        self.last_score: dict[tuple[str, str], float] = {}
        self.pending_event_ids: dict[tuple[str, str], str] = {}
        self.sequence = 0
        self.live_sequence = 0
        self._lock = threading.Lock()

    def ready(self) -> bool:
        return self.client.ready()

    @staticmethod
    def _manifest(event_id: str, camera: str, media_type: str, codec: str, content: bytes, frame_time: float) -> MediaManifest:
        return MediaManifest(
            media_id=uuid.uuid4().hex,
            event_id=event_id,
            camera_id=camera,
            media_type=media_type,
            codec=codec,
            start_time=frame_time,
            end_time=frame_time,
            byte_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            expiry_unix_ms=int((time.time() + 3600) * 1000),
        )

    def publish(
        self,
        decision: HazardDecision,
        frame: np.ndarray,
        frame_time: float,
        media: SafetyMediaStore,
        frames: list[tuple[float, np.ndarray]],
    ) -> str:
        with self._lock:
            key = (decision.camera, decision.label)
            if decision.active:
                if decision.bbox is None:
                    raise SafetyEventError("safety_event_requires_bbox")
                existing_event_id = self.active.get(key)
                event_id = (
                    existing_event_id
                    or self.pending_event_ids.get(key)
                    or "safety" + uuid.uuid4().hex[:24]
                )
                operation = (
                    TrackerOperation.START
                    if existing_event_id is None
                    else TrackerOperation.UPDATE
                )
            else:
                event_id = self.active.get(key)
                if event_id is None:
                    raise SafetyEventError("safety_clear_without_active_event")
                operation = TrackerOperation.END

            evidence_bbox = decision.bbox or self.last_bbox.get(key)
            if evidence_bbox is None:
                raise SafetyEventError("safety_event_requires_bbox")
            evidence_score = decision.score
            if not decision.active:
                evidence_score = self.last_score.get(key, evidence_score)
            evidence_decision = HazardDecision(
                camera=decision.camera,
                label=decision.label,
                active=decision.active,
                score=evidence_score,
                bbox=evidence_bbox,
            )
            height, width = frame.shape[:2]
            bbox = BoundingBox(
                round(evidence_bbox[0] * width),
                round(evidence_bbox[1] * height),
                round(evidence_bbox[2] * width),
                round(evidence_bbox[3] * height),
            )
            snapshot = media.snapshot(frame, evidence_decision)
            manifests = [
                self._manifest(
                    event_id,
                    decision.camera,
                    "snapshot_jpg",
                    "jpeg",
                    snapshot,
                    frame_time,
                )
            ]
            content_by_id = {manifests[0].media_id: snapshot}
            if operation is TrackerOperation.END:
                clip = media.clip(event_id, frames)
                clip_manifest = self._manifest(
                    event_id, decision.camera, "clip", "mp4", clip, frame_time
                )
                manifests.append(clip_manifest)
                content_by_id[clip_manifest.media_id] = clip
            next_sequence = self.sequence + 1
            update = TrackerUpdate(
                node_id=self.node_id,
                node_epoch=self.client.node_epoch,
                camera_id=decision.camera,
                stream_epoch=self.client.stream_epoch,
                journal_sequence=next_sequence,
                frame_seq=next_sequence,
                source_pts=round(frame_time * 1_000_000),
                frame_time=frame_time,
                event_id=event_id,
                track_id=f"{decision.camera}:{decision.label}",
                operation=operation,
                label=decision.label,
                score_history=(evidence_score,),
                score=evidence_score,
                bbox=bbox,
                state={"source": "safety"},
                media=tuple(manifests),
                source_type="safety",
            )
            if decision.active:
                self.pending_event_ids[key] = event_id
            try:
                for manifest in manifests:
                    self.client.upload_media(manifest, content_by_id[manifest.media_id])
                self.client.publish(update)
            except ProducerTransportError as error:
                raise SafetyEventError(str(error)) from error
            self.sequence = next_sequence
            if operation is TrackerOperation.END:
                self.active.pop(key, None)
                self.pending_event_ids.pop(key, None)
                self.last_bbox.pop(key, None)
                self.last_score.pop(key, None)
            else:
                self.active[key] = event_id
                self.pending_event_ids.pop(key, None)
                self.last_bbox[key] = decision.bbox
                self.last_score[key] = decision.score
            return event_id

    def publish_live(
        self,
        decision: HazardDecision,
        frame_shape: tuple[int, int],
        frame_time: float,
    ) -> None:
        if decision.bbox is None:
            return
        with self._lock:
            self.live_sequence += 1
            key = (decision.camera, decision.label)
            event_id = self.active.get(
                key, f"live-{decision.camera}-{decision.label}"
            )
            height, width = frame_shape
            bbox = BoundingBox(
                round(decision.bbox[0] * width),
                round(decision.bbox[1] * height),
                round(decision.bbox[2] * width),
                round(decision.bbox[3] * height),
            )
            update = TrackerUpdate(
                node_id=self.node_id,
                node_epoch=self.client.node_epoch,
                camera_id=decision.camera,
                stream_epoch=self.client.stream_epoch,
                journal_sequence=0,
                frame_seq=self.live_sequence,
                source_pts=round(frame_time * 1_000_000),
                frame_time=frame_time,
                event_id=event_id,
                track_id=f"{decision.camera}:{decision.label}:live",
                operation=TrackerOperation.UPDATE,
                label=decision.label,
                score_history=(decision.score,),
                score=decision.score,
                bbox=bbox,
                state={"source": "safety", "live": True},
                media=(),
                source_type="safety",
            )
            try:
                self.client.publish_live(update)
            except ProducerTransportError as error:
                raise SafetyEventError(str(error)) from error

    def close(self) -> None:
        self.client.close()
