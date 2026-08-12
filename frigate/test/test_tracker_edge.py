from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from peewee import SqliteDatabase
from pydantic import ValidationError

from frigate.config.tracker import TrackerConfig
from frigate.models import EdgeMediaManifest, EventObservation, TrackerJournalEntry
from frigate.tracker.canonical import TrackerCanonicalStore
from frigate.tracker.contracts import (
    BoundingBox,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
)
from frigate.tracker.evidence import (
    EvidenceCapacityError,
    EvidenceRing,
    EvidenceUnavailableError,
)
from frigate.tracker.host import TrackerHostIngest, TrackerIngestError
from frigate.tracker.journal import EdgeJournal, SpoolFullError
from frigate.tracker.media import MediaAuthority
from frigate.tracker.ownership import should_start_local_camera
from frigate.tracker.producer import ProducerContext, TrackerProducerCore
from frigate.tracker.service.grpc_server import TrackerGrpcService
from frigate.tracker.service.v1 import tracker_pb2 as pb
from frigate.tracker.service.wire import update_from_proto, update_to_proto


def _update(
    operation: TrackerOperation = TrackerOperation.START,
    *,
    sequence: int = 0,
    event_id: str = "event-1",
    node_epoch: str = "node-epoch-1",
    stream_epoch: str = "stream-epoch-1",
) -> TrackerUpdate:
    return TrackerUpdate(
        node_id="edge-local",
        node_epoch=node_epoch,
        camera_id="face_camera",
        stream_epoch=stream_epoch,
        journal_sequence=sequence,
        frame_seq=7,
        source_pts=700,
        frame_time=10.5,
        event_id=event_id,
        track_id="raw-4",
        operation=operation,
        label="person",
        score_history=(0.7, 0.8),
        score=0.8,
        bbox=BoundingBox(1, 2, 20, 30),
        current_zones=("gate",),
        path=((1.0, 2.0),),
        speed=3.5,
    )


def _tracker_config() -> dict[str, object]:
    return {
        "nodes": {
            "edge-local": {
                "endpoint": "tracker-edge-local:50052",
                "cameras": ["face_camera"],
                "tls": {
                    "ca": "/tls/ca.crt",
                    "certificate": "/tls/client.crt",
                    "key": "/tls/client.key",
                    "server_name": "tracker-edge-local",
                },
            }
        }
    }


def test_proto_round_trip_preserves_producer_event_and_lineage() -> None:
    update = _update(sequence=9)
    assert update_from_proto(update_to_proto(update)) == update
    assert update.identity.track_id == "raw-4"
    assert update.event_id != update.identity.track_id


def test_proto_service_and_field_numbers_are_compatible() -> None:
    service = pb.DESCRIPTOR.services_by_name["TrackerService"]
    assert [method.name for method in service.methods] == [
        "Connect",
        "Configure",
        "GetCapabilities",
        "GetEvidence",
        "StreamMedia",
        "ControlCamera",
    ]
    fields = pb.TrackerUpdate.DESCRIPTOR.fields_by_name
    assert fields["journal_sequence"].number == 5
    assert fields["event_id"].number == 9
    assert fields["operation"].number == 11
    assert fields["evidence"].number == 23
    assert fields["media"].number == 25


def test_tracker_source_has_no_direct_recognition_dependency() -> None:
    tracker_root = Path(__file__).parents[1] / "tracker"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in tracker_root.rglob("*.py")
    )
    assert "frigate.recognition" not in sources


def test_tracker_config_requires_unique_camera_owner_and_mtls() -> None:
    config = TrackerConfig.model_validate(_tracker_config())
    assert config.owner_for("face_camera") == "edge-local"
    duplicate = _tracker_config()
    duplicate["nodes"]["edge-two"] = {  # type: ignore[index]
        **duplicate["nodes"]["edge-local"],  # type: ignore[index]
        "endpoint": "edge-two:50052",
    }
    with pytest.raises(ValidationError, match="belongs to tracker nodes"):
        TrackerConfig.model_validate(duplicate)
    missing_tls = _tracker_config()
    missing_tls["nodes"]["edge-local"]["tls"]["key"] = ""  # type: ignore[index]
    with pytest.raises(ValidationError, match="require mTLS"):
        TrackerConfig.model_validate(missing_tls)


def test_tracker_mtls_identity_rejects_unknown_client(tmp_path) -> None:
    journal = EdgeJournal(tmp_path / "edge.db")
    service = TrackerGrpcService(
        node_id="edge-local",
        node_epoch="epoch",
        journal=journal,
        evidence={},
        media=MediaAuthority(tmp_path / "media"),
        allowed_client_identities=frozenset({"frigate-main"}),
    )

    class Context:
        def auth_context(self):
            return {"x509_common_name": (b"unknown",)}

        async def abort(self, code, message):
            import grpc

            assert code is grpc.StatusCode.UNAUTHENTICATED
            raise PermissionError(message)

    with pytest.raises(PermissionError, match="not allowed"):
        asyncio.run(service._authorize(Context()))
    journal.close()


def test_edge_owned_camera_does_not_start_local_capture_or_tracker() -> None:
    tracker = TrackerConfig.model_validate(_tracker_config())
    assert not should_start_local_camera(tracker, "face_camera")
    assert should_start_local_camera(tracker, "car_camera")
    assert should_start_local_camera(
        tracker, "face_camera", edge_node_id="edge-local"
    )
    assert not should_start_local_camera(
        tracker, "face_camera", edge_node_id="edge-two"
    )


def test_embedded_and_edge_use_same_producer_contract() -> None:
    observation = {
        "frame_seq": 4,
        "frame_time": 1.5,
        "track_id": "9",
        "label": "car",
        "score": 0.75,
        "score_history": [0.6, 0.75],
        "box": [2, 3, 20, 30],
        "path": [[3.0, 4.0]],
    }
    context = ProducerContext("edge-local", "ne", "car_camera", "se")
    embedded = TrackerProducerCore(context).emit(
        observation, operation=TrackerOperation.UPDATE, event_id="producer-event"
    )
    edge = TrackerProducerCore(context).emit(
        observation, operation=TrackerOperation.UPDATE, event_id="producer-event"
    )
    assert embedded == edge


def test_evidence_ring_ttl_checksum_pin_and_capacity() -> None:
    ring = EvidenceRing("edge-local", "face_camera", max_bytes=24, ttl_seconds=10)
    frame = bytes(range(12))
    reference = ring.put(
        stream_epoch="se", frame_seq=1, data=frame, shape=(2, 4), now=100
    )
    assert reference.evidence_id.startswith("ev1:edge-local:face_camera:se:1:")
    assert ring.get(reference.evidence_id, now=101)[1] == frame
    ring.pin(reference.evidence_id)
    second = ring.put(
        stream_epoch="se", frame_seq=2, data=frame, shape=(2, 4), now=101
    )
    ring.pin(second.evidence_id)
    with pytest.raises(EvidenceCapacityError):
        ring.put(
            stream_epoch="se", frame_seq=3, data=frame, shape=(2, 4), now=101
        )
    ring.release(reference.evidence_id)
    ring.release(second.evidence_id)
    assert ring.expire(now=111) == 2
    with pytest.raises(EvidenceUnavailableError):
        ring.get(reference.evidence_id, now=111)


def test_journal_ack_replay_restart_and_monotonic_sequence(tmp_path) -> None:
    path = tmp_path / "edge.db"
    journal = EdgeJournal(path, max_bytes=32_000)
    start = journal.append(_update())
    update = journal.append(_update(TrackerOperation.UPDATE))
    assert (start.journal_sequence, update.journal_sequence) == (1, 2)
    assert journal.ack(1, "event-1")
    assert journal.ack(1, "event-1")
    assert [item.journal_sequence for item in journal.replay()] == [2]
    journal.close()
    reopened = EdgeJournal(path, max_bytes=32_000)
    end = reopened.append(_update(TrackerOperation.END))
    assert end.journal_sequence == 3
    reopened.close()


def test_journal_pins_exact_i420_until_correlated_ack(tmp_path) -> None:
    journal = EdgeJournal(tmp_path / "edge.db", max_bytes=32_000)
    data = bytes(range(12))
    ring = EvidenceRing("edge-local", "face_camera", max_bytes=24, ttl_seconds=10)
    reference = ring.put(
        stream_epoch="stream-epoch-1",
        frame_seq=7,
        data=data,
        shape=(2, 4),
        now=100,
    )
    journal.pin_evidence(reference, data, (2, 4), created=100)
    durable = replace(reference, durable=True)
    stored = journal.append(replace(_update(), evidence=durable))
    stored_reference, stored_data, shape = journal.get_evidence(
        reference.evidence_id
    )
    assert stored_reference.durable
    assert stored_data == data and shape == (2, 4)
    assert journal.ack(stored.journal_sequence, stored.event_id)
    journal.compact(now=111)
    with pytest.raises(KeyError):
        journal.get_evidence(reference.evidence_id)
    journal.close()


def test_journal_spool_full_records_gap_without_silent_drop(tmp_path) -> None:
    journal = EdgeJournal(tmp_path / "edge.db", max_bytes=1200)
    journal.append(_update())
    with pytest.raises(SpoolFullError, match="spool_full"):
        while True:
            journal.append(_update(TrackerOperation.UPDATE))
    assert journal.gaps()[-1]["reason"] == "spool_full"
    assert journal.pending_count >= 1
    journal.close()


def test_journal_keeps_unacked_canonical_lifecycle_during_compaction(tmp_path) -> None:
    journal = EdgeJournal(tmp_path / "edge.db", max_bytes=32_000, retention_seconds=1)
    start = journal.append(_update())
    journal._db.execute("UPDATE journal SET created=0")
    assert journal.compact(now=100) == (0, 0)
    assert journal.replay()[0].journal_sequence == start.journal_sequence
    journal.close()


def test_host_ingest_is_ordered_idempotent_and_acks_after_commit() -> None:
    committed: list[TrackerUpdate] = []
    ingest = TrackerHostIngest({"face_camera": "edge-local"}, committed.append)
    start = _update(sequence=1)
    ack = ingest.accept(start)
    assert ack.event_id == "event-1" and committed == [start]
    duplicate = ingest.accept(start)
    assert duplicate.duplicate and committed == [start]
    update = _update(TrackerOperation.UPDATE, sequence=2)
    ingest.accept(update)
    end = _update(TrackerOperation.END, sequence=3)
    ingest.accept(end)
    assert ingest.active_count == 0


def test_host_ingest_rejects_gaps_epoch_join_and_changed_event() -> None:
    ingest = TrackerHostIngest({"face_camera": "edge-local"}, lambda _: None)
    ingest.accept(_update(sequence=1))
    with pytest.raises(TrackerIngestError, match="journal_sequence_gap"):
        ingest.accept(_update(TrackerOperation.UPDATE, sequence=3))
    with pytest.raises(TrackerIngestError, match="producer_event_id_changed"):
        ingest.accept(
            _update(TrackerOperation.UPDATE, sequence=2, event_id="synthetic-event")
        )
    with pytest.raises(TrackerIngestError, match="stream_epoch_changed"):
        ingest.accept(
            _update(TrackerOperation.UPDATE, sequence=2, stream_epoch="new-stream")
        )


def test_failed_canonical_commit_does_not_advance_ack_state() -> None:
    def fail(_: TrackerUpdate) -> None:
        raise sqlite3.IntegrityError("transaction failed")

    ingest = TrackerHostIngest({"face_camera": "edge-local"}, fail)
    with pytest.raises(sqlite3.IntegrityError):
        ingest.accept(_update(sequence=1))
    assert ingest.active_count == 0


def test_canonical_store_commits_journal_observation_and_manifest_atomically() -> None:
    database = SqliteDatabase(":memory:")
    models = [TrackerJournalEntry, EventObservation, EdgeMediaManifest]
    with database.bind_ctx(models):
        database.create_tables(models)
        store = TrackerCanonicalStore(database)
        manifest = MediaManifest(
            "clip-1",
            "event-1",
            "face_camera",
            1,
            2,
            "h264",
            10,
            "0" * 64,
            4_000_000_000_000,
            "clip",
        )
        update = replace(_update(sequence=1), media=(manifest,))
        assert store.accept(update)
        assert not store.accept(update)
        assert store.last_sequence("edge-local", "node-epoch-1") == 1
        assert TrackerJournalEntry.select().count() == 1
        assert EventObservation.select().count() == 1
        assert EdgeMediaManifest.select().count() == 1


def test_media_manifest_range_retain_and_idempotent_delete(tmp_path) -> None:
    authority = MediaAuthority(tmp_path)
    payload = b"0123456789"
    manifest = authority.register(
        media_id="clip-1",
        event_id="event-1",
        camera_id="face_camera",
        data=payload,
        start_time=1,
        end_time=2,
        codec="h264",
        media_type="clip",
        ttl_seconds=10,
    )
    assert manifest.sha256 == hashlib.sha256(payload).hexdigest()
    assert authority.read_range("clip-1", 2, 4) == b"2345"
    assert authority.retain("clip-1")
    assert authority.delete("clip-1")
    assert authority.delete("clip-1")


def test_media_manifest_cannot_change_producer_event_id() -> None:
    manifest = MediaManifest(
        "clip", "another-event", "face_camera", 1, 2, "h264", 0, "0" * 64, 3, "clip"
    )
    with pytest.raises(ValueError, match="producer event_id"):
        replace(_update(), media=(manifest,))
