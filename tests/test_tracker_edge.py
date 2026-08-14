"""Contracts for the thin external tracker wrapper."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from extension.tracker.runtime import (
    BoundingBox,
    CameraTrackAdapter,
    TrackerJournal,
    TrackerOperation,
    TrackerRuntime,
    TrackerUpdate,
    tracker_config_fingerprint,
)
from extension.tracker.transport import (
    TrackerCanonicalStore,
    TrackerHostIngest,
    TrackerIngestError,
)
from playhouse.sqlite_ext import SqliteExtDatabase
from playhouse.sqliteq import SqliteQueueDatabase

from frigate.models import (
    EdgeMediaManifest,
    Event,
    EventObservation,
    TrackerJournalEntry,
)


def _update(
    operation: TrackerOperation = TrackerOperation.START,
    *,
    sequence: int = 1,
) -> TrackerUpdate:
    return TrackerUpdate(
        node_id="edge-local",
        node_epoch="node-epoch",
        camera_id="face_camera",
        stream_epoch="stream-epoch",
        journal_sequence=sequence,
        frame_seq=sequence,
        source_pts=sequence * 1_000_000,
        frame_time=float(sequence),
        event_id="producer-trace-id",
        track_id="native-track-id",
        operation=operation,
        label="person",
        score_history=(0.8,),
        score=0.8,
        bbox=BoundingBox(1, 2, 20, 30),
    )


def test_tracker_extension_contains_exactly_four_python_files() -> None:
    root = Path("src/extension/tracker")
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py"))
    assert files == ["__init__.py", "app.py", "runtime.py", "transport.py"]


def test_tracker_runtime_does_not_depend_on_backup() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/extension/tracker").glob("*.py")
    )
    assert "tracker-backup" not in sources
    assert "tracker_backup" not in sources


def test_update_json_preserves_producer_trace_and_native_track_id() -> None:
    update = _update()
    restored = TrackerUpdate.from_json(update.to_json())
    assert restored == update
    assert restored.trace_id == "producer-trace-id"
    assert restored.track_id == "native-track-id"


def test_update_json_accepts_native_defaultdict_state() -> None:
    update = _update()
    update.state["attributes"] = defaultdict(list, {"face": [{"score": 0.9}]})
    restored = TrackerUpdate.from_json(update.to_json())
    assert restored.state["attributes"] == {"face": [{"score": 0.9}]}


def test_media_manifest_roundtrip_and_canonical_persistence() -> None:
    database = SqliteExtDatabase(":memory:")
    models = (TrackerJournalEntry, EventObservation, EdgeMediaManifest, Event)
    database.bind(models)
    database.create_tables(models)
    update = _update()
    update = TrackerUpdate.from_json(
        update.to_json().replace(
            '"media":[]',
            '"media":[{"byte_size":3,"camera_id":"face_camera",'
            '"codec":"jpeg","end_time":1.0,"event_id":"producer-trace-id",'
            '"expiry_unix_ms":4102444800000,"media_id":"abc",'
            '"media_type":"snapshot","sha256":"abc","start_time":1.0}]',
        )
    )
    TrackerCanonicalStore(database).accept(update)
    row = EdgeMediaManifest.get_by_id("abc")
    assert row.event_id == update.event_id
    assert row.byte_size == 3
    database.close()


def test_canonical_store_reconstructs_active_lifecycle() -> None:
    database = SqliteExtDatabase(":memory:")
    models = (TrackerJournalEntry, EventObservation, EdgeMediaManifest, Event)
    database.bind(models)
    database.create_tables(models)
    store = TrackerCanonicalStore(database)
    store.accept(_update(TrackerOperation.START, sequence=1))
    assert list(store.active_lifecycles("edge-local").values()) == [
        "producer-trace-id"
    ]
    store.accept(_update(TrackerOperation.END, sequence=2))
    assert store.active_lifecycles("edge-local") == {}
    database.close()


def test_canonical_store_supports_main_sqlite_queue_database(tmp_path: Path) -> None:
    database_path = tmp_path / "main.db"
    models = (TrackerJournalEntry, EventObservation, EdgeMediaManifest, Event)
    schema_database = SqliteExtDatabase(database_path)
    with schema_database.bind_ctx(models):
        schema_database.create_tables(models)
    schema_database.close()
    database = SqliteQueueDatabase(database_path, autostart=True)
    with database.bind_ctx(models):
        TrackerCanonicalStore(database).accept(_update())
        assert TrackerJournalEntry.get().event_id == "producer-trace-id"
        assert EventObservation.get().event_id == "producer-trace-id"
    database.stop()


def test_tracker_event_state_preserves_native_region_box() -> None:
    update = _update()
    update.state["region"] = [0, 1, 20, 30]
    update.state["detection_regions"] = [[0, 0, 50, 50]]
    restored = TrackerUpdate.from_json(update.to_json())
    assert restored.state["region"] == [0, 1, 20, 30]
    assert restored.state["detection_regions"] == [[0, 0, 50, 50]]


def test_journal_replay_and_exact_ack(tmp_path: Path) -> None:
    journal = TrackerJournal(tmp_path / "journal.db")
    persisted = journal.append(_update(sequence=0))
    assert persisted.journal_sequence == 1
    assert journal.next_pending(persisted.node_epoch) == persisted
    assert journal.acknowledge_through(persisted.node_epoch, 0) == 0
    assert journal.acknowledge_through(persisted.node_epoch, 1) == 1
    assert journal.next_pending(persisted.node_epoch) is None
    journal.close()


def test_journal_health_counts_only_current_epoch(tmp_path: Path) -> None:
    journal = TrackerJournal(tmp_path / "journal.db")
    stale = _update(sequence=0)
    current = TrackerUpdate.from_json(stale.to_json())
    object.__setattr__(current, "node_epoch", "epoch-current")
    journal.append(stale)
    persisted = journal.append(current)

    assert journal.pending_count == 2
    assert journal.pending_count_for_epoch("epoch-current") == 1
    journal.acknowledge_through("epoch-current", persisted.journal_sequence)
    assert journal.pending_count_for_epoch("epoch-current") == 0
    assert journal.pending_count == 1
    journal.close()


def test_host_ingest_rejects_wrong_camera_owner() -> None:
    ingest = TrackerHostIngest({"face_camera": "edge-two"}, lambda update: None)
    with pytest.raises(TrackerIngestError, match="camera_owner_mismatch"):
        ingest.accept(_update())


def test_host_ingest_requires_start_update_end_order() -> None:
    committed: list[TrackerUpdate] = []
    ingest = TrackerHostIngest({"face_camera": "edge-local"}, committed.append)
    ingest.accept(_update(TrackerOperation.START, sequence=1))
    ingest.accept(_update(TrackerOperation.UPDATE, sequence=2))
    ingest.accept(_update(TrackerOperation.END, sequence=3))
    assert [update.operation for update in committed] == [
        TrackerOperation.START,
        TrackerOperation.UPDATE,
        TrackerOperation.END,
    ]
    assert ingest.active == {}


def test_launcher_probe_uses_private_json_grpc_contract() -> None:
    launcher = Path("../deploy/run.ps1").read_text(encoding="utf-8")
    assert "/camera.tracker.v1.TrackerService/GetCapabilities" in launcher
    assert "extension.tracker.service.v1" not in launcher


def test_tracker_startup_retry_is_not_reported_as_a_disconnect_traceback() -> None:
    source = Path("src/extension/tracker/transport.py").read_text(encoding="utf-8")
    assert "session_started = False" in source
    assert "Tracker node %s is not ready; retrying" in source
    assert 'logger.exception("Tracker node %s disconnected"' not in source


def test_finite_source_finalize_ends_each_active_track_once() -> None:
    adapter = CameraTrackAdapter.__new__(CameraTrackAdapter)
    first = SimpleNamespace()
    second = SimpleNamespace()
    adapter.camera = "car_camera"
    adapter.state = SimpleNamespace(tracked_objects={"1": first, "2": second})
    adapter.event_ids = {"1": "event-one"}
    ended: list[tuple[str, object]] = []
    adapter._end = lambda camera, obj: ended.append((camera, obj))  # type: ignore[method-assign]
    adapter.finalize()
    assert ended == [("car_camera", first)]


def test_tracker_writes_completion_after_all_sources_and_media_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = TrackerRuntime.__new__(TrackerRuntime)
    runtime.node_id = "edge-local"
    runtime.node_epoch = "epoch-current"
    runtime.adapters = {
        "face_camera": SimpleNamespace(
            finalize=lambda: None, event_ids={"1": "face-event"}
        ),
        "car_camera": SimpleNamespace(
            finalize=lambda: None, event_ids={"2": "car-event"}
        ),
    }
    runtime.finalized_sources = set()
    runtime.source_idle_polls = {camera: 1 for camera in runtime.adapters}
    runtime.session_complete_written = False
    runtime.media = SimpleNamespace(
        pending_count=lambda: 0,
        completed_event_ids=lambda: ("car-event", "face-event"),
    )
    for camera in runtime.adapters:
        (tmp_path / f"{camera}.end").write_text("1.0\n", encoding="utf-8")
    monkeypatch.setenv("PASSAGE_SOURCE_START_DIR", str(tmp_path))

    runtime._finalize_ended_sources()

    marker = json.loads(
        (tmp_path / "tracker-session-complete.json").read_text(encoding="utf-8")
    )
    assert marker["node_epoch"] == "epoch-current"
    assert marker["cameras"] == ["car_camera", "face_camera"]
    assert marker["events"] == ["car-event", "face-event"]


def test_tracker_reports_active_lifecycles_from_owned_adapters() -> None:
    runtime = TrackerRuntime.__new__(TrackerRuntime)
    runtime.adapters = {
        "face_camera": SimpleNamespace(event_ids={"1": "face-event"}),
        "car_camera": SimpleNamespace(event_ids={}),
    }

    assert runtime.active_lifecycle_count() == 1


def test_tracker_clip_temporary_path_keeps_mp4_suffix() -> None:
    source = Path("src/extension/tracker/runtime.py").read_text(encoding="utf-8")
    assert 'f"{clip_path.stem}.tmp{clip_path.suffix}"' in source
    assert 'command.extend(["-t",' in source
    assert 'with_suffix(".mp4.tmp")' not in source


def test_tracker_omits_unconsumed_continuous_media_pipeline() -> None:
    source = Path("src/extension/tracker/runtime.py").read_text(encoding="utf-8")
    assert "RecordProcess" not in source
    assert "OutputProcess" not in source
    assert "DetectionPublisher" not in source
    assert "camera_config.record.enabled = False" in source


def test_entrypoint_only_delegates_to_runtime_main() -> None:
    source = Path("src/extension/tracker/app.py").read_text(encoding="utf-8")
    assert source.count("from extension.tracker.runtime import main") == 1
    assert "CameraMaintainer" not in source


def test_tracker_maintainer_uses_one_grpc_response_api() -> None:
    """Prevent grpc.aio UsageError from mixing read and iterator styles."""
    source = Path("src/extension/tracker/transport.py").read_text(encoding="utf-8")
    assert "async for raw in call" not in source
    assert "raw = await call.read()" in source
    assert "if raw is aio.EOF" in source


def test_tracker_handshake_uses_compiler_owned_topology_revision() -> None:
    revision = "a" * 64
    main = SimpleNamespace(
        runtime=SimpleNamespace(topology_role="main", topology_revision=revision),
        tracker={"edge-local": object()},
    )
    edge = SimpleNamespace(
        runtime=SimpleNamespace(topology_role="tracker", topology_revision=revision),
        tracker={"edge-local": object()},
    )
    assert tracker_config_fingerprint(main, "edge-local") == revision
    assert tracker_config_fingerprint(edge, "edge-local") == revision
