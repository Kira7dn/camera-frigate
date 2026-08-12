"""Frigate SQLite acceptance transaction for tracker journal entries."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict

from peewee import Database

from frigate.models import EdgeMediaManifest, EventObservation, TrackerJournalEntry

from ..domain.contracts import TrackerOperation, TrackerUpdate
from .ingest import TrackerIngestError


class TrackerCanonicalStore:
    """Atomically accept journal metadata and every referenced media manifest."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def accept(self, update: TrackerUpdate) -> bool:
        payload_json = update.to_json()
        payload = json.loads(payload_json)
        now = datetime.datetime.now(datetime.UTC)
        key = (
            update.node_id,
            update.node_epoch,
            update.journal_sequence,
        )
        with self.database.atomic():
            existing = TrackerJournalEntry.get_or_none(
                (TrackerJournalEntry.node_id == key[0])
                & (TrackerJournalEntry.node_epoch == key[1])
                & (TrackerJournalEntry.journal_sequence == key[2])
            )
            if existing is not None:
                if existing.event_id != update.event_id or existing.payload != payload:
                    raise TrackerIngestError("durable_sequence_conflict")
                return False

            for manifest in update.media:
                stored = EdgeMediaManifest.get_or_none(
                    EdgeMediaManifest.media_id == manifest.media_id
                )
                expected = asdict(manifest)
                if stored is not None:
                    if stored.node_id != update.node_id:
                        raise TrackerIngestError("media_manifest_owner_conflict")
                    actual = {
                        name: getattr(stored, name)
                        for name in expected
                        if name != "expiry_unix_ms"
                    }
                    expected_without_expiry = {
                        name: value
                        for name, value in expected.items()
                        if name != "expiry_unix_ms"
                    }
                    if actual != expected_without_expiry:
                        raise TrackerIngestError("media_manifest_conflict")
                    continue
                EdgeMediaManifest.create(
                    media_id=manifest.media_id,
                    node_id=update.node_id,
                    camera_id=manifest.camera_id,
                    event_id=manifest.event_id,
                    media_type=manifest.media_type,
                    codec=manifest.codec,
                    start_time=manifest.start_time,
                    end_time=manifest.end_time,
                    byte_size=manifest.byte_size,
                    sha256=manifest.sha256,
                    expires_at=datetime.datetime.fromtimestamp(
                        manifest.expiry_unix_ms / 1000, datetime.UTC
                    ),
                )

            TrackerJournalEntry.create(
                node_id=update.node_id,
                node_epoch=update.node_epoch,
                journal_sequence=update.journal_sequence,
                camera_id=update.camera_id,
                stream_epoch=update.stream_epoch,
                event_id=update.event_id,
                operation=update.operation.value,
                payload=payload,
                accepted_at=now,
            )
            observation_key = ":".join(str(value) for value in key)
            EventObservation.create(
                observation_id=hashlib.sha256(
                    observation_key.encode("utf-8")
                ).hexdigest(),
                event_id=update.event_id,
                kind={
                    TrackerOperation.START: "tracker_start",
                    TrackerOperation.UPDATE: "tracker_update",
                    TrackerOperation.END: "event_ended",
                }[update.operation],
                observed_at=now,
                frame_time=update.frame_time,
                evidence_id=(
                    update.evidence.evidence_id
                    if update.evidence is not None
                    else None
                ),
                payload=payload,
                expires_at=now + datetime.timedelta(days=2),
            )
        return True

    def last_sequence(self, node_id: str, _node_epoch: str) -> int:
        row = (
            TrackerJournalEntry.select(TrackerJournalEntry.journal_sequence)
            .where(TrackerJournalEntry.node_id == node_id)
            .order_by(TrackerJournalEntry.journal_sequence.desc())
            .first()
        )
        return 0 if row is None else int(row.journal_sequence)
