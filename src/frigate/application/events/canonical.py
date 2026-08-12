"""Canonical event projection and immutable media materialization.

This module intentionally owns only the new projection fields on ``Event``.
Tracking continues to populate the legacy columns during the shadow rollout.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
from peewee import IntegrityError

from frigate.const import CLIPS_DIR
from frigate.models import (
    Event,
    EventEvidence,
    EventObservation,
    MediaArtifact,
    NotificationDelivery,
)

RENDER_VERSION = 1
OBSERVATION_RETENTION_DAYS = 2
DEFAULT_ARTIFACT_RETENTION_DAYS = 30

logger = logging.getLogger(__name__)


class EvidenceMismatch(ValueError):
    """Raised when an overlay did not originate from the selected frame."""


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def as_utc(value: datetime.datetime | str) -> datetime.datetime:
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def display_label(
    object_label: str | None,
    sub_label: str | None = None,
    license_plate: str | None = None,
) -> str:
    """Return the single user-facing label using the canonical precedence."""
    return str(sub_label or license_plate or object_label or "object")


def normalized_xyxy(
    box: Iterable[float], width: int, height: int
) -> list[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or width <= 0 or height <= 0:
        raise ValueError("bbox requires four coordinates and positive dimensions")
    x1, y1, x2, y2 = values
    if max(abs(value) for value in values) > 1:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    result = [
        min(1.0, max(0.0, x1)),
        min(1.0, max(0.0, y1)),
        min(1.0, max(0.0, x2)),
        min(1.0, max(0.0, y2)),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("bbox must have positive normalized area")
    return result


@dataclass(frozen=True)
class RenderSpec:
    event_id: str
    revision: int
    evidence_id: str
    profile: str = "canonical"
    render_version: int = RENDER_VERSION

    @property
    def key(self) -> str:
        return ":".join(
            (
                self.event_id,
                str(self.revision),
                self.evidence_id,
                self.profile,
                str(self.render_version),
            )
        )

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(self.key.encode("utf-8")).hexdigest()


class CanonicalMediaStore:
    """Lazy, deterministic, single-flight artifact renderer and quota owner."""

    _locks_guard = threading.Lock()
    _locks: ClassVar[dict[str, threading.Lock]] = {}

    def __init__(
        self,
        root: str | Path | None = None,
        max_storage_mb: int = 2048,
        retention_days: int = DEFAULT_ARTIFACT_RETENTION_DAYS,
    ) -> None:
        self.root = Path(root or Path(CLIPS_DIR) / "artifacts")
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.retention_days = retention_days

    @classmethod
    def _lock_for(cls, key: str) -> threading.Lock:
        with cls._locks_guard:
            return cls._locks.setdefault(key, threading.Lock())

    @staticmethod
    def _artifact_valid(artifact: MediaArtifact | None) -> bool:
        if artifact is None:
            return False
        path = Path(artifact.path)
        if not path.is_file() or path.stat().st_size != artifact.byte_size:
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256

    def get(self, artifact_id: str | None) -> MediaArtifact | None:
        if not artifact_id:
            return None
        artifact = MediaArtifact.get_or_none(MediaArtifact.id == artifact_id)
        return artifact if self._artifact_valid(artifact) else None

    def latest_for_event(self, event_id: str) -> MediaArtifact | None:
        artifact = (
            MediaArtifact.select()
            .where(
                (MediaArtifact.event_id == event_id)
                & (MediaArtifact.profile == "canonical")
            )
            .order_by(MediaArtifact.revision.desc())
            .first()
        )
        return artifact if self._artifact_valid(artifact) else None

    def bytes(self, artifact_id: str | None) -> bytes | None:
        artifact = self.get(artifact_id)
        return Path(artifact.path).read_bytes() if artifact else None

    def _render(self, evidence: EventEvidence, label: str) -> bytes:
        image = cv2.imread(evidence.frame_ref, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(evidence.frame_ref)
        if image.shape[1] != evidence.width or image.shape[0] != evidence.height:
            raise ValueError("evidence dimensions do not match the full frame")
        boxes = list(evidence.boxes or [])
        object_boxes = [box for box in boxes if box.get("role") == "object"]
        if not object_boxes:
            raise ValueError("canonical evidence requires one object bbox")
        box = object_boxes[0]
        if box.get("evidence_id", evidence.id) != evidence.id:
            raise EvidenceMismatch("bbox belongs to another evidence frame")
        x1, y1, x2, y2 = box["normalized_xyxy"]
        pixels = (
            round(x1 * evidence.width),
            round(y1 * evidence.height),
            round(x2 * evidence.width),
            round(y2 * evidence.height),
        )
        # Canonical profile always has exactly one white object rectangle.
        cv2.rectangle(image, pixels[:2], pixels[2:], (255, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.7, 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        text_x = max(0, pixels[0])
        text_y = max(text_height + baseline + 2, pixels[1])
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - baseline - 2),
            (min(evidence.width - 1, text_x + text_width + 4), text_y + 2),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            image,
            label,
            (text_x + 2, text_y - baseline),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("unable to encode canonical artifact")
        return encoded.tobytes()

    def materialize(
        self, spec: RenderSpec, evidence: EventEvidence, label: str
    ) -> MediaArtifact | None:
        if evidence.id != spec.evidence_id:
            raise EvidenceMismatch("render spec and evidence do not match")
        existing = self.get(spec.artifact_id)
        if existing:
            return existing
        with self._lock_for(spec.key):
            existing = self.get(spec.artifact_id)
            if existing:
                return existing
            content = self._render(evidence, label)
            if not self.reserve(len(content)):
                return None
            checksum = hashlib.sha256(content).hexdigest()
            target = self.root / spec.artifact_id[:2] / f"{spec.artifact_id}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, target)
            now = utcnow()
            manifest = {
                "render_spec": {
                    "event_id": spec.event_id,
                    "revision": spec.revision,
                    "evidence_id": spec.evidence_id,
                    "profile": spec.profile,
                    "render_version": spec.render_version,
                },
                "display_label": label,
                "object_bbox": next(
                    box["normalized_xyxy"]
                    for box in evidence.boxes
                    if box.get("role") == "object"
                ),
                "sha256": checksum,
            }
            try:
                return MediaArtifact.create(
                    id=spec.artifact_id,
                    event_id=spec.event_id,
                    revision=spec.revision,
                    evidence_id=spec.evidence_id,
                    profile=spec.profile,
                    render_version=spec.render_version,
                    path=str(target),
                    sha256=checksum,
                    byte_size=len(content),
                    manifest=manifest,
                    created_at=now,
                    expires_at=now + datetime.timedelta(days=self.retention_days),
                )
            except IntegrityError:
                target.unlink(missing_ok=True)
                return self.get(spec.artifact_id)

    def reserve(self, new_bytes: int) -> bool:
        self.cleanup_expired()
        used = sum(
            row.byte_size for row in MediaArtifact.select(MediaArtifact.byte_size)
        )
        return used + new_bytes <= self.max_storage_bytes

    def cleanup_expired(self, now: datetime.datetime | None = None) -> int:
        now = now or utcnow()
        protected = {
            row.media_artifact_id
            for row in NotificationDelivery.select(
                NotificationDelivery.media_artifact_id
            ).where(NotificationDelivery.status << ("pending", "processing"))
            if row.media_artifact_id
        }
        removed = 0
        expired = MediaArtifact.select().where(
            (MediaArtifact.expires_at < now) & (MediaArtifact.pinned == False)
        )
        for artifact in expired:
            if artifact.id in protected:
                continue
            Path(artifact.path).unlink(missing_ok=True)
            artifact.delete_instance()
            removed += 1
        return removed


class EventAggregator:
    """Durable observation reducer and sole writer of canonical Event fields."""

    def __init__(
        self,
        media: CanonicalMediaStore | None = None,
        finalization_timeout: float = 5.0,
    ) -> None:
        self.media = media or CanonicalMediaStore()
        self.finalization_timeout = finalization_timeout

    def add_evidence(
        self,
        *,
        evidence_id: str,
        event_id: str,
        frame_ref: str,
        frame_time: float,
        width: int,
        height: int,
        boxes: list[dict[str, Any]],
        technical: dict[str, Any] | None = None,
    ) -> EventEvidence:
        normalized = []
        for box in boxes:
            owner = box.get("evidence_id", evidence_id)
            if owner != evidence_id:
                raise EvidenceMismatch("all evidence boxes must belong to the same frame")
            normalized.append(
                {
                    **box,
                    "evidence_id": evidence_id,
                    "normalized_xyxy": normalized_xyxy(
                        box.get("normalized_xyxy") or box.get("box"), width, height
                    ),
                }
            )
        EventEvidence.insert(
            id=evidence_id,
            event_id=event_id,
            frame_ref=frame_ref,
            frame_time=frame_time,
            width=width,
            height=height,
            boxes=normalized,
            technical=technical or {},
            created_at=utcnow(),
        ).on_conflict_ignore().execute()
        return EventEvidence.get_by_id(evidence_id)

    def observe(
        self,
        *,
        observation_id: str,
        event_id: str,
        kind: str,
        payload: dict[str, Any],
        observed_at: datetime.datetime | None = None,
        frame_time: float | None = None,
        evidence_id: str | None = None,
    ) -> bool:
        now = observed_at or utcnow()
        try:
            EventObservation.create(
                observation_id=observation_id,
                event_id=event_id,
                kind=kind,
                observed_at=now,
                frame_time=frame_time,
                evidence_id=evidence_id,
                payload=payload,
                expires_at=now
                + datetime.timedelta(days=OBSERVATION_RETENTION_DAYS),
            )
        except IntegrityError:
            return False
        event = Event.get_or_none(Event.id == event_id)
        if event is None:
            return True
        if kind == "event_ended":
            Event.update(state="END_SEEN").where(Event.id == event_id).execute()
        elif event.state == "FINALIZED" and kind in ("lpr", "face", "genai"):
            self.finalize(event_id, late=True)
        return True

    def finalize_due(self, now: datetime.datetime | None = None) -> list[str]:
        now = now or utcnow()
        EventObservation.delete().where(EventObservation.expires_at < now).execute()
        committed = []
        for event in Event.select().where(Event.state == "END_SEEN"):
            ended = (
                EventObservation.select()
                .where(
                    (EventObservation.event_id == event.id)
                    & (EventObservation.kind == "event_ended")
                )
                .order_by(EventObservation.observed_at.desc())
                .first()
            )
            if ended and (now - as_utc(ended.observed_at)).total_seconds() >= self.finalization_timeout:
                self.finalize(event.id)
                committed.append(event.id)
        return committed

    def finalize(self, event_id: str, late: bool = False) -> MediaArtifact | None:
        event = Event.get_by_id(event_id)
        observations = list(
            EventObservation.select()
            .where(EventObservation.event_id == event_id)
            .order_by(EventObservation.observed_at)
        )
        facts: dict[str, Any] = {}
        chosen_evidence = event.canonical_evidence_id
        for observation in observations:
            payload = observation.payload or {}
            if observation.kind == "lpr" and payload.get("plate"):
                facts["plate"] = payload["plate"]
                facts["plate_score"] = payload.get("score")
            if observation.kind == "face" and payload.get("sub_label"):
                facts["sub_label"] = payload["sub_label"]
            if observation.evidence_id:
                chosen_evidence = observation.evidence_id
        plate = facts.get("plate") or event.canonical_plate
        sub_label = facts.get("sub_label") or event.canonical_sub_label or event.sub_label
        label = display_label(event.label, sub_label, plate)
        unchanged = (
            event.revision > 0
            and event.canonical_plate == plate
            and event.canonical_sub_label == sub_label
            and event.canonical_evidence_id == chosen_evidence
            and event.display_label == label
        )
        if unchanged:
            return self.media.get(event.canonical_artifact_id)
        revision = event.revision + 1
        Event.update(
            state="FINALIZING",
            revision=revision,
            canonical_plate=plate,
            canonical_plate_score=facts.get("plate_score", event.canonical_plate_score),
            canonical_sub_label=sub_label,
            display_label=label,
            canonical_evidence_id=chosen_evidence,
        ).where(Event.id == event_id).execute()
        artifact = None
        if chosen_evidence:
            evidence = EventEvidence.get_or_none(EventEvidence.id == chosen_evidence)
            if evidence:
                try:
                    artifact = self.media.materialize(
                        RenderSpec(event_id, revision, chosen_evidence), evidence, label
                    )
                except FileNotFoundError:
                    logger.warning(
                        "Canonical evidence file is missing for event %s: %s",
                        event_id,
                        evidence.frame_ref,
                    )
        Event.update(
            state="FINALIZED",
            finalized_at=utcnow(),
            canonical_artifact_id=artifact.id if artifact else None,
        ).where(Event.id == event_id).execute()
        return artifact

def artifact_manifest(artifact: MediaArtifact | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.id,
        "revision": artifact.revision,
        "evidence_id": artifact.evidence_id,
        "profile": artifact.profile,
        "render_version": artifact.render_version,
        "sha256": artifact.sha256,
        "byte_size": artifact.byte_size,
    }
