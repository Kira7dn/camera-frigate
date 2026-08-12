"""Durable operational journal for ordered offline tracker replay."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .contracts import EvidenceReference, TrackerOperation, TrackerUpdate


class SpoolFullError(RuntimeError):
    pass


class EdgeJournal:
    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 256 * 1024 * 1024,
        retention_seconds: int = 24 * 60 * 60,
    ) -> None:
        if max_bytes <= 0 or retention_seconds <= 0:
            raise ValueError("spool budget and retention must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.retention_seconds = retention_seconds
        self._db = sqlite3.connect(self.path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS state (
              key TEXT PRIMARY KEY,
              value INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO state(key, value) VALUES ('last_sequence', 0);
            CREATE TABLE IF NOT EXISTS journal (
              sequence INTEGER PRIMARY KEY,
              created REAL NOT NULL,
              operation TEXT NOT NULL,
              event_id TEXT NOT NULL,
              canonical INTEGER NOT NULL,
              acked INTEGER NOT NULL DEFAULT 0,
              payload TEXT NOT NULL,
              evidence_id TEXT,
              payload_bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
              evidence_id TEXT PRIMARY KEY,
              created REAL NOT NULL,
              expiry_ms INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              payload BLOB NOT NULL,
              shape_height INTEGER NOT NULL DEFAULT 0,
              shape_width INTEGER NOT NULL DEFAULT 0,
              payload_bytes INTEGER NOT NULL,
              pins INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS gaps (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created REAL NOT NULL,
              reason TEXT NOT NULL,
              next_sequence INTEGER NOT NULL
            );
            """
        )
        journal_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(journal)")
        }
        if "evidence_id" not in journal_columns:
            self._db.execute("ALTER TABLE journal ADD COLUMN evidence_id TEXT")
        evidence_columns = {
            row[1] for row in self._db.execute("PRAGMA table_info(evidence)")
        }
        if "shape_height" not in evidence_columns:
            self._db.execute(
                "ALTER TABLE evidence ADD COLUMN shape_height INTEGER NOT NULL DEFAULT 0"
            )
        if "shape_width" not in evidence_columns:
            self._db.execute(
                "ALTER TABLE evidence ADD COLUMN shape_width INTEGER NOT NULL DEFAULT 0"
            )

    @property
    def last_sequence(self) -> int:
        row = self._db.execute(
            "SELECT value FROM state WHERE key='last_sequence'"
        ).fetchone()
        return int(row[0])

    @property
    def pending_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM journal WHERE acked=0"
        ).fetchone()
        return int(row[0])

    @property
    def logical_bytes(self) -> int:
        journal_bytes = self._db.execute(
            "SELECT COALESCE(SUM(payload_bytes), 0) FROM journal"
        ).fetchone()[0]
        evidence_bytes = self._db.execute(
            "SELECT COALESCE(SUM(payload_bytes), 0) FROM evidence"
        ).fetchone()[0]
        return int(journal_bytes) + int(evidence_bytes)

    def append(self, update: TrackerUpdate, *, canonical: bool = True) -> TrackerUpdate:
        sequence = self.last_sequence + 1
        if update.journal_sequence not in (0, sequence):
            raise ValueError("journal_sequence must be zero or the next sequence")
        stored = TrackerUpdate(
            **{
                name: getattr(update, name)
                for name in update.__dataclass_fields__
                if name != "journal_sequence"
            },
            journal_sequence=sequence,
        )
        payload = stored.to_json()
        payload_bytes = len(payload.encode("utf-8"))
        if self.logical_bytes + payload_bytes > self.max_bytes:
            self._db.execute(
                "INSERT INTO gaps(created, reason, next_sequence) VALUES (?, ?, ?)",
                (time.time(), "spool_full", sequence),
            )
            raise SpoolFullError("spool_full")
        with self._db:
            self._db.execute(
                """INSERT INTO journal(
                    sequence, created, operation, event_id, canonical, payload,
                    evidence_id, payload_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sequence,
                    time.time(),
                    stored.operation.value,
                    stored.event_id,
                    int(canonical),
                    payload,
                    stored.evidence.evidence_id
                    if stored.evidence is not None
                    else None,
                    payload_bytes,
                ),
            )
            self._db.execute(
                "UPDATE state SET value=? WHERE key='last_sequence'", (sequence,)
            )
        return stored

    def ack(self, sequence: int, event_id: str) -> bool:
        row = self._db.execute(
            "SELECT event_id, acked FROM journal WHERE sequence=?", (sequence,)
        ).fetchone()
        if row is None or row["event_id"] != event_id:
            return False
        if row["acked"]:
            return True
        evidence_row = self._db.execute(
            "SELECT evidence_id FROM journal WHERE sequence=?", (sequence,)
        ).fetchone()
        with self._db:
            self._db.execute(
                "UPDATE journal SET acked=1 WHERE sequence=?", (sequence,)
            )
            if evidence_row is not None and evidence_row["evidence_id"]:
                self.release_evidence(evidence_row["evidence_id"])
        return True

    def replay(self, after_sequence: int = 0) -> list[TrackerUpdate]:
        rows = self._db.execute(
            "SELECT payload FROM journal WHERE sequence>? AND acked=0 ORDER BY sequence",
            (after_sequence,),
        ).fetchall()
        return [TrackerUpdate.from_json(row["payload"]) for row in rows]

    def pin_evidence(
        self,
        reference: EvidenceReference,
        data: bytes,
        shape: tuple[int, int],
        *,
        created: float | None = None,
    ) -> None:
        if len(data) != reference.byte_length:
            raise ValueError("durable evidence byte length mismatch")
        additional = 0
        existing = self._db.execute(
            "SELECT payload_bytes FROM evidence WHERE evidence_id=?",
            (reference.evidence_id,),
        ).fetchone()
        if existing is None:
            additional = len(data)
        if self.logical_bytes + additional > self.max_bytes:
            raise SpoolFullError("spool_full")
        self._db.execute(
            """INSERT INTO evidence(
                evidence_id, created, expiry_ms, sha256, payload,
                shape_height, shape_width, payload_bytes, pins
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(evidence_id) DO UPDATE SET pins=pins+1""",
            (
                reference.evidence_id,
                time.time() if created is None else created,
                reference.expiry_unix_ms,
                reference.sha256,
                data,
                shape[0],
                shape[1],
                len(data),
            ),
        )

    def get_evidence(
        self, evidence_id: str
    ) -> tuple[EvidenceReference, bytes, tuple[int, int]]:
        row = self._db.execute(
            """SELECT payload, expiry_ms, sha256, shape_height, shape_width
               FROM evidence WHERE evidence_id=?""",
            (evidence_id,),
        ).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        data = bytes(row["payload"])
        reference = EvidenceReference(
            evidence_id,
            len(data),
            row["sha256"],
            int(row["expiry_ms"]),
            True,
        )
        return reference, data, (int(row["shape_height"]), int(row["shape_width"]))

    def release_evidence(self, evidence_id: str) -> None:
        self._db.execute(
            "UPDATE evidence SET pins=MAX(0, pins-1) WHERE evidence_id=?",
            (evidence_id,),
        )

    def compact(self, now: float | None = None) -> tuple[int, int]:
        now = time.time() if now is None else now
        cutoff = now - self.retention_seconds
        journal_deleted = self._db.execute(
            """DELETE FROM journal
               WHERE acked=1 AND created<?
               AND operation NOT IN (?, ?)""",
            (cutoff, TrackerOperation.START.value, TrackerOperation.END.value),
        ).rowcount
        evidence_deleted = self._db.execute(
            "DELETE FROM evidence WHERE pins=0 AND expiry_ms<=?",
            (int(now * 1000),),
        ).rowcount
        return int(journal_deleted), int(evidence_deleted)

    def gaps(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._db.execute("SELECT * FROM gaps ORDER BY id")]

    def close(self) -> None:
        self._db.close()
