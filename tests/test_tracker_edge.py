"""Contracts for the thin external tracker wrapper."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from playhouse.sqlite_ext import SqliteExtDatabase
from extension.tracker.runtime import (
    BoundingBox,
    MediaManifest,
    TrackerJournal,
    TrackerOperation,
    TrackerUpdate,
)
from extension.tracker.transport import (
    TrackerCanonicalStore,
    TrackerHostIngest,
    TrackerIngestError,
)
from frigate.models import EdgeMediaManifest, EventObservation, TrackerJournalEntry


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
    models = (TrackerJournalEntry, EventObservation, EdgeMediaManifest)
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
    models = (TrackerJournalEntry, EventObservation, EdgeMediaManifest)
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


def test_journal_replay_and_exact_ack(tmp_path: Path) -> None:
    journal = TrackerJournal(tmp_path / "journal.db")
    persisted = journal.append(_update(sequence=0))
    assert persisted.journal_sequence == 1
    assert journal.replay() == (persisted,)
    assert not journal.acknowledge(1, "wrong", persisted.node_epoch)
    assert journal.acknowledge(1, persisted.event_id, persisted.node_epoch)
    assert journal.replay() == ()
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
