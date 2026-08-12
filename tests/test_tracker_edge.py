from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from peewee import SqliteDatabase
from pydantic import ValidationError

from frigate.api.media import _edge_media_response
from frigate.domain.camera.runtime import CAMERA_RUNTIME_MODELS, camera_runtime_config
from frigate.domain.camera.state import CameraState
from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.config.tracker import TrackerConfig
from frigate.models import EdgeMediaManifest, EventObservation, TrackerJournalEntry
from camera_platform.topology.compiler import compile_topology, materialize_topology
from camera_platform.tracker.adapters.canonical import TrackerCanonicalStore
from camera_platform.tracker.adapters.frigate import TrackerMaintainer
from camera_platform.tracker.adapters.ingest import TrackerHostIngest, TrackerIngestError
from camera_platform.tracker.adapters.media import resolve_event_media, resolve_media_id
from camera_platform.tracker.config.fingerprint import (
    canonical_tracker_config_json,
    tracker_config_fingerprint,
)
from camera_platform.tracker.domain.contracts import (
    BoundingBox,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
)
from camera_platform.tracker.runtime.evidence import (
    EvidenceCapacityError,
    EvidenceRing,
    EvidenceUnavailableError,
)
from camera_platform.tracker.runtime.journal import EdgeJournal, SpoolFullError
from camera_platform.tracker.runtime.media import MediaAuthority
from camera_platform.tracker.runtime.processor import EdgeTrackedObjectProcessor
from camera_platform.tracker.runtime.producer import ProducerContext, TrackerProducerCore
from camera_platform.tracker.service.grpc_server import TrackerGrpcService
from camera_platform.tracker.service.v1 import tracker_pb2 as pb
from camera_platform.tracker.service.wire import update_from_proto, update_to_proto
from camera_platform.tracker.domain.lifecycle import apply_media_policy, project_tracker_observation


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


def _full_tracker_config(*, detect_fps: int = 5, tls_key: str = "/tls/key.pem"):
    return FrigateConfig(
        mqtt={"host": "mqtt"},
        cameras={
            "face_camera": {
                "ffmpeg": {
                    "inputs": [
                        {"path": "rtsp://camera/face", "roles": ["detect"]}
                    ]
                },
                "detect": {"width": 1280, "height": 720, "fps": detect_fps},
            },
            "car_camera": {
                "ffmpeg": {
                    "inputs": [
                        {"path": "rtsp://camera/car", "roles": ["detect"]}
                    ]
                },
                "detect": {"width": 1280, "height": 720, "fps": 5},
            },
        },
        tracker={
            "edge-local": {
                "endpoint": "tracker-edge-local:50052",
                "cameras": ["face_camera"],
                "tls": {
                    "ca": "/tls/ca.pem",
                    "certificate": "/tls/client.pem",
                    "key": tls_key,
                    "server_name": "tracker-edge-local",
                },
            }
        },
    )


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
    assert fields["state_json"].number == 26
    assert pb.NodeHealth.DESCRIPTOR.fields_by_name["active_lifecycles"].number == 9
    assert pb.CapabilitiesResponse.DESCRIPTOR.fields_by_name["health"].number == 10
    assert pb.CapabilitiesResponse.DESCRIPTOR.fields_by_name["config_hash"].number == 11


def test_checked_in_proto_bindings_support_production_runtime_versions() -> None:
    generated = Path(pb.__file__).read_text(encoding="utf-8")
    grpc_generated = Path(pb.__file__).with_name("tracker_pb2_grpc.py").read_text(
        encoding="utf-8"
    )
    assert "google.protobuf import runtime_version" not in generated
    assert "GRPC_GENERATED_VERSION" not in grpc_generated


def test_tracker_health_reports_durable_terminal_state(tmp_path) -> None:
    journal = EdgeJournal(tmp_path / "edge.db")
    media = MediaAuthority(tmp_path / "media")
    service = TrackerGrpcService(
        node_id="edge-local",
        node_epoch="epoch",
        journal=journal,
        evidence={},
        media=media,
        terminal_state=lambda: {
            "pending_ack": 2,
            "spool_bytes": 128,
            "pinned_evidence": 1,
            "active": 3,
        },
    )
    health = service._node_health()
    assert health.pending_ack == 2
    assert health.spool_bytes == 128
    assert health.pinned_evidence == 1
    assert health.active_lifecycles == 3
    media.close()
    journal.close()


def test_host_projection_keeps_event_id_and_raw_track_lineage_separate() -> None:
    runtime = object.__new__(TrackerMaintainer)
    update = replace(
        _update(sequence=1),
        state={
            "id": "raw-4",
            "start_time": 9.5,
            "end_time": None,
            "has_clip": True,
            "has_snapshot": False,
        },
    )
    data = runtime._event_data(update, "")
    assert data["id"] == "event-1"
    assert data["raw_track_id"] == "raw-4"
    assert data["has_clip"] is True
    assert data["start_time"] == 9.5


def test_tracker_source_has_no_direct_recognition_dependency() -> None:
    tracker_root = Path(__file__).parents[1] / "track" / "edge"
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in tracker_root.rglob("*.py")
    )
    assert "frigate.application.recognition" not in sources


def test_tracker_config_requires_unique_camera_owner_and_mtls() -> None:
    config = TrackerConfig.model_validate(_tracker_config())
    assert config.camera_owners == {"face_camera": "edge-local"}
    assert "edge-local" in config.model_dump()
    assert "nodes" not in config.model_dump()
    duplicate = _tracker_config()
    duplicate["edge-two"] = {  # type: ignore[index]
        **duplicate["edge-local"],  # type: ignore[index]
        "endpoint": "edge-two:50052",
    }
    with pytest.raises(ValidationError, match="belongs to tracker nodes"):
        TrackerConfig.model_validate(duplicate)
    missing_tls = _tracker_config()
    missing_tls["edge-local"]["tls"]["key"] = ""  # type: ignore[index]
    with pytest.raises(ValidationError, match="require mTLS"):
        TrackerConfig.model_validate(missing_tls)


def test_shared_camera_runtime_config_selects_main_and_edge_owners() -> None:
    config = _full_tracker_config()
    topology = compile_topology(config)
    assert topology.embedded_cameras == ("car_camera",)
    assert topology.camera_owners == {"face_camera": "edge-local"}
    assert tuple(
        camera_runtime_config(config, topology=topology).cameras
    ) == ("car_camera",)
    assert tuple(
        camera_runtime_config(
            config, edge_node_id="edge-local", topology=topology
        ).cameras
    ) == ("face_camera",)
    assert {model.__name__ for model in CAMERA_RUNTIME_MODELS} == {
        "Recordings",
        "ReviewSegment",
        "Regions",
    }


def test_topology_compiler_materializes_main_and_isolated_node_configs(
    tmp_path: Path,
) -> None:
    config = _full_tracker_config()
    raw = config.model_dump(mode="json")
    raw["go2rtc"] = {
        "streams": {
            "face_camera": "rtsp://camera/face",
            "car_camera": "rtsp://camera/car",
        }
    }
    plan = compile_topology(config)

    manifest = materialize_topology(raw, plan, tmp_path)

    main = yaml.safe_load((tmp_path / "config.main.yml").read_text(encoding="utf-8"))
    edge = yaml.safe_load(
        (tmp_path / "config.tracker.edge-local.yml").read_text(encoding="utf-8")
    )
    assert manifest["camera_owners"] == {"face_camera": "edge-local"}
    assert main["go2rtc"]["streams"]["face_camera"] == (
        "rtsp://tracker-edge-local:8554/face_camera"
    )
    assert tuple(edge["cameras"]) == ("face_camera",)
    assert tuple(edge["tracker"]) == ("edge-local",)
    assert main["runtime"]["topology_role"] == "main"
    assert edge["runtime"]["topology_role"] == "tracker"
    assert edge["runtime"]["topology_node_id"] == "edge-local"
    assert main["runtime"]["topology_revision"] == manifest["revision"]
    assert edge["runtime"]["topology_revision"] == manifest["revision"]


def test_tracker_config_fingerprint_is_behavioral_and_excludes_private_key() -> None:
    original = _full_tracker_config()
    changed_key = _full_tracker_config(tls_key="/different/private-key.pem")
    changed_camera = _full_tracker_config(detect_fps=6)
    assert tracker_config_fingerprint(original, "edge-local") == (
        tracker_config_fingerprint(changed_key, "edge-local")
    )
    assert tracker_config_fingerprint(original, "edge-local") != (
        tracker_config_fingerprint(changed_camera, "edge-local")
    )


def test_tracker_configure_validates_the_mounted_canonical_revision(tmp_path) -> None:
    config = _full_tracker_config()
    config_json = canonical_tracker_config_json(config, "edge-local")
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    journal = EdgeJournal(tmp_path / "edge.db")
    media = MediaAuthority(tmp_path / "media")
    service = TrackerGrpcService(
        node_id="edge-local",
        node_epoch="epoch",
        journal=journal,
        evidence={},
        media=media,
        allowed_client_identities=frozenset({"frigate-main"}),
        config_hash=config_hash,
    )

    class Context:
        def auth_context(self):
            return {"x509_common_name": (b"frigate-main",)}

        async def abort(self, code, message):
            raise RuntimeError(f"{code.name}:{message}")

    response = asyncio.run(
        service.Configure(
            pb.ConfigureRequest(
                client_id="frigate-main",
                config_json=config_json,
                config_hash=config_hash,
            ),
            Context(),
        )
    )
    assert response.config_hash == config_hash
    journal.close()
    media.close()


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
    topology = compile_topology(_full_tracker_config())
    assert topology.embedded_cameras == ("car_camera",)
    assert topology.node("edge-local").cameras == ("face_camera",)
    with pytest.raises(ValueError, match="unknown tracker node"):
        topology.node("edge-two")


def test_shared_lifecycle_projection_preserves_frigate_state() -> None:
    class Object:
        false_positive = False
        face_snapshot = None
        max_severity = "detection"
        has_snapshot = False
        has_clip = False
        score_history = [0.6, 0.75]
        path_data = [((3.0, 4.0), 1.5)]
        current_estimated_speed = 2.5
        current_zones = ["gate"]
        entered_zones = ["gate"]
        obj_data = {"id": "9", "position_changes": 1}

        def to_dict(self):
            return {
                "id": "9",
                "frame_time": 1.5,
                "label": "car",
                "score": 0.75,
                "box": [2, 3, 20, 30],
                "attributes": {},
                "current_zones": self.current_zones,
                "entered_zones": self.entered_zones,
                "has_snapshot": self.has_snapshot,
                "has_clip": self.has_clip,
            }

    config = SimpleNamespace(
        cameras={
            "car_camera": SimpleNamespace(
                snapshots=SimpleNamespace(enabled=True, required_zones=[]),
                record=SimpleNamespace(enabled=True),
            )
        }
    )
    obj = Object()
    apply_media_policy(config, "car_camera", obj)  # type: ignore[arg-type]
    observation = project_tracker_observation(
        obj,  # type: ignore[arg-type]
        frame_seq=4,
        motion_boxes=[(1, 2, 3, 4)],
        regions=[(0, 0, 10, 10)],
    )
    assert observation["has_snapshot"] and observation["has_clip"]
    assert observation["score_history"] == (0.6, 0.75)
    assert observation["path"] == ((3.0, 4.0),)
    context = ProducerContext("edge-local", "ne", "car_camera", "se")
    update = TrackerProducerCore(context).emit(
        observation, operation=TrackerOperation.UPDATE, event_id="producer-event"
    )
    assert update.track_id == "9"
    assert update.current_zones == ("gate",)
    assert update.path == ((3.0, 4.0),)
    assert update.speed == 2.5


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


def test_restart_recovery_ends_real_active_state_before_new_epoch(tmp_path) -> None:
    path = tmp_path / "restart-active.db"
    journal = EdgeJournal(path)
    journal.append(_update(sequence=0, node_epoch="old-epoch"))
    journal.close()

    restarted = EdgeJournal(path)
    recovered = restarted.recover_active()
    assert len(recovered) == 1
    assert recovered[0].operation is TrackerOperation.END
    assert recovered[0].node_epoch == "old-epoch"
    assert recovered[0].failure is not None
    assert recovered[0].failure.code == "process_restart"
    assert recovered[0].journal_sequence == 2

    ingest = TrackerHostIngest({"face_camera": "edge-local"}, lambda _: None)
    ingest.accept(restarted.replay()[0])
    ingest.accept(recovered[0])
    new_start = restarted.append(
        _update(sequence=0, node_epoch="new-epoch", stream_epoch="new-stream")
    )
    ingest.accept(new_start)
    assert ingest.active_count == 1
    restarted.close()


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
    authority.close()


def test_media_manifest_survives_authority_restart(tmp_path) -> None:
    payload = b"durable-edge-media"
    authority = MediaAuthority(tmp_path)
    authority.register(
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
    assert authority.retain("clip-1")
    authority.close()

    restored = MediaAuthority(tmp_path)
    assert restored.manifest("clip-1").event_id == "event-1"
    assert restored.read_range("clip-1", 8, 4) == b"edge"
    assert restored.expire(now=time.time() + 60) == 0
    assert restored.delete("clip-1")
    restored.close()


def test_edge_media_resolver_preserves_http_byte_range() -> None:
    database = SqliteDatabase(":memory:")
    with database.bind_ctx([EdgeMediaManifest]):
        database.create_tables([EdgeMediaManifest])
        EdgeMediaManifest.create(
            media_id="clip-1",
            node_id="edge-local",
            camera_id="face_camera",
            event_id="event-1",
            media_type="clip",
            codec="h264",
            start_time=1,
            end_time=2,
            byte_size=10,
            sha256="0" * 64,
            expires_at="2099-01-01T00:00:00+00:00",
        )
        resolved = resolve_event_media("event-1", "clip", "bytes=2-5")
        assert resolved is not None
        assert (resolved.offset, resolved.length) == (2, 4)
        assert resolve_media_id("clip-1") == resolved.manifest


def test_edge_media_api_proxies_authenticated_range() -> None:
    database = SqliteDatabase(":memory:")

    class Runtime:
        async def fetch_media(self, node_id, media_id, *, offset, length):
            assert (node_id, media_id, offset, length) == (
                "edge-local",
                "clip-1",
                2,
                4,
            )
            return b"2345"

    request = SimpleNamespace(
        app=SimpleNamespace(tracker_maintainer=Runtime()),
        headers={"range": "bytes=2-5"},
    )
    with database.bind_ctx([EdgeMediaManifest]):
        database.create_tables([EdgeMediaManifest])
        EdgeMediaManifest.create(
            media_id="clip-1",
            node_id="edge-local",
            camera_id="face_camera",
            event_id="event-1",
            media_type="clip",
            codec="h264",
            start_time=1,
            end_time=2,
            byte_size=10,
            sha256="0" * 64,
            expires_at="2099-01-01T00:00:00+00:00",
        )
        response = asyncio.run(
            _edge_media_response(request, "event-1", "clip", "video/mp4")
        )
    assert response is not None
    assert response.status_code == 206
    assert response.body == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"


def test_media_manifest_cannot_change_producer_event_id() -> None:
    manifest = MediaManifest(
        "clip", "another-event", "face_camera", 1, 2, "h264", 0, "0" * 64, 3, "clip"
    )
    with pytest.raises(ValueError, match="producer event_id"):
        replace(_update(), media=(manifest,))


def test_frozen_camera_state_replay_has_embedded_edge_lifecycle_parity() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "tracker_edge"
        / "lifecycle_replay.json"
    )
    frames = json.loads(fixture.read_text(encoding="utf-8"))
    config = FrigateConfig(
        mqtt={"host": "mqtt"},
        cameras={
            "face_camera": {
                "ffmpeg": {
                    "inputs": [{"path": "fixture", "roles": ["detect"]}]
                },
                "detect": {"width": 100, "height": 100, "fps": 5},
            }
        },
    )

    class FrameManager:
        def get(self, name, shape):
            return None

    class AutoTracker:
        def autotrack_object(self, camera, obj):
            return None

        def end_object(self, camera, obj):
            return None

    ptz = SimpleNamespace(ptz_autotracker=AutoTracker())
    embedded = CameraState("face_camera", config, FrameManager(), ptz)
    embedded_observations: list[tuple[TrackerOperation, dict[str, object]]] = []
    frame_seq = 0

    def capture(operation):
        def callback(camera, obj, frame_name, observed):
            apply_media_policy(config, camera, obj)
            embedded_observations.append(
                (
                    operation,
                    project_tracker_observation(
                        obj, frame_seq=frame_seq, motion_boxes=[], regions=[]
                    ),
                )
            )

        return callback

    for name, operation in (
        ("start", TrackerOperation.START),
        ("update", TrackerOperation.UPDATE),
        ("end", TrackerOperation.END),
    ):
        embedded.on(name, capture(operation))

    context = ProducerContext(
        "edge-local", "node-epoch", "face_camera", "stream-epoch"
    )
    edge_updates: list[TrackerUpdate] = []

    class Publisher:
        def publish(self, payload, topic):
            return None

        def stop(self):
            return None

    edge = EdgeTrackedObjectProcessor(
        config,
        context,
        TrackerProducerCore(context),
        EvidenceRing(
            "edge-local",
            "face_camera",
            max_bytes=1024 * 1024,
            ttl_seconds=45,
        ),
        ptz,
        edge_updates.append,
        detection_publisher=Publisher(),
    )
    try:
        for index, frame in enumerate(frames, start=1):
            frame_seq = index
            embedded.update(
                "missing-frame",
                frame["frame_time"],
                copy.deepcopy(frame["objects"]),
                [],
                [],
            )
            edge.process(
                "missing-frame",
                frame["frame_time"],
                copy.deepcopy(frame["objects"]),
                [],
                [],
            )
    finally:
        edge.close()

    assert [item[0] for item in embedded_observations] == [
        update.operation for update in edge_updates
    ]
    assert [item[1] for item in embedded_observations] == [
        update.state for update in edge_updates
    ]
