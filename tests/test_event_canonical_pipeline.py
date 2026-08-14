import datetime
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from playhouse.sqlite_ext import SqliteExtDatabase

from frigate.api.defs.response.event_response import EventResponse
from frigate.application.events.canonical import (
    CanonicalMediaStore,
    EventAggregator,
    EvidenceMismatch,
    RenderSpec,
)
from frigate.application.events.maintainer import EventProcessor
from frigate.models import (
    Event,
    EventEvidence,
    EventObservation,
    MediaArtifact,
    NotificationDelivery,
)

MODELS = [
    Event,
    EventEvidence,
    EventObservation,
    MediaArtifact,
    NotificationDelivery,
]


@pytest.fixture()
def canonical_db():
    with tempfile.TemporaryDirectory() as directory:
        database = SqliteExtDatabase(
            str(Path(directory) / "canonical.db"), pragmas={"journal_mode": "wal"}
        )
        database.bind(MODELS)
        database.create_tables(MODELS)
        yield database, Path(directory)
        database.close()


def create_event(event_id: str) -> Event:
    now = datetime.datetime.now(datetime.UTC)
    return Event.create(
        id=event_id,
        label="car",
        camera="car_camera",
        start_time=now - datetime.timedelta(seconds=10),
        end_time=now,
        top_score=0.96,
        score=0.96,
        false_positive=False,
        zones=[],
        thumbnail="",
        region=[0, 0, 200, 100],
        box=[20, 20, 180, 80],
        area=9600,
        plus_id="",
        model_hash="",
        detector_type="",
        model_type="",
        data={},
    )


def create_evidence(
    aggregator: EventAggregator, root: Path, event_id: str, evidence_id: str
) -> EventEvidence:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:] = (20, 40, 60)
    frame_path = root / f"{evidence_id}.jpg"
    assert cv2.imwrite(str(frame_path), frame)
    return aggregator.add_evidence(
        evidence_id=evidence_id,
        event_id=event_id,
        frame_ref=str(frame_path),
        frame_time=100.5,
        width=200,
        height=100,
        boxes=[
            {"role": "object", "box": [20, 20, 180, 80]},
            {"role": "plate", "box": [70, 55, 130, 75]},
        ],
        technical={"detector": "lpr"},
    )


def test_edge_event_processor_persists_recognition_metadata(canonical_db):
    _, root = canonical_db
    event = create_event("edge-event")
    processor = EventProcessor.__new__(EventProcessor)
    processor.events_in_process = {event.id: {}}
    processor.event_aggregator = EventAggregator(CanonicalMediaStore(root / "artifacts"))

    processor._apply_recognition_metadata(
        event.id, "recognized_license_plate", "ABC123", 0.97, "lpr"
    )
    processor._apply_recognition_metadata(event.id, "sub_label", "Alice", 0.98, "face")
    processor.event_aggregator.finalize(event.id)

    event = Event.get_by_id(event.id)
    assert processor.events_in_process[event.id]["recognized_license_plate"] == (
        "ABC123",
        0.97,
    )
    assert event.data["recognized_license_plate"] == "ABC123"
    assert event.data["recognized_license_plate_score"] == 0.97
    assert event.sub_label == "Alice"
    assert event.canonical_plate == "ABC123"
    assert event.canonical_sub_label == "Alice"


def test_rejects_bbox_from_another_evidence(canonical_db):
    _, root = canonical_db
    aggregator = EventAggregator(CanonicalMediaStore(root / "artifacts"))
    with pytest.raises(EvidenceMismatch):
        aggregator.add_evidence(
            evidence_id="frame-a",
            event_id="event-a",
            frame_ref=str(root / "missing.jpg"),
            frame_time=1,
            width=100,
            height=100,
            boxes=[
                {
                    "role": "object",
                    "evidence_id": "frame-b",
                    "box": [0, 0, 50, 50],
                }
            ],
        )


def test_single_flight_reuses_one_artifact_for_100_consumers(canonical_db):
    database, root = canonical_db

    class CountingStore(CanonicalMediaStore):
        count = 0
        guard = threading.Lock()

        def _render(self, evidence, label):
            with self.guard:
                type(self).count += 1
            return super()._render(evidence, label)

    store = CountingStore(root / "artifacts")
    aggregator = EventAggregator(store)
    evidence = create_evidence(aggregator, root, "event-a", "frame-a")
    spec = RenderSpec("event-a", 1, evidence.id)
    def consume(_):
        try:
            return store.materialize(spec, evidence, "FKH9211")
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=20) as executor:
        artifacts = list(
            executor.map(consume, range(100))
        )
    assert {artifact.id for artifact in artifacts if artifact} == {spec.artifact_id}
    assert CountingStore.count == 1
    assert MediaArtifact.select().count() == 1
    artifact = artifacts[0]
    assert artifact.manifest["display_label"] == "FKH9211"
    assert artifact.manifest["object_bbox"] == [0.1, 0.2, 0.9, 0.8]


def test_restart_preserves_enrichment_and_late_identity_creates_revision(canonical_db):
    _, root = canonical_db
    event = create_event("event-restart")
    first = EventAggregator(CanonicalMediaStore(root / "artifacts"), 5)
    evidence = create_evidence(first, root, event.id, "frame-restart")
    first.observe(
        observation_id="lpr-1",
        event_id=event.id,
        kind="lpr",
        payload={"plate": "FKH9211", "score": 0.96},
        evidence_id=evidence.id,
    )

    restarted = EventAggregator(CanonicalMediaStore(root / "artifacts"), 5)
    restarted.observe(
        observation_id="end-1",
        event_id=event.id,
        kind="event_ended",
        payload={},
        observed_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=6),
    )
    assert restarted.finalize_due() == [event.id]
    event = Event.get_by_id(event.id)
    assert (event.state, event.revision, event.display_label) == (
        "FINALIZED",
        1,
        "FKH9211",
    )
    first_artifact = event.canonical_artifact_id

    restarted.observe(
        observation_id="face-late-1",
        event_id=event.id,
        kind="face",
        payload={"sub_label": "Nguyễn An"},
        evidence_id=evidence.id,
    )
    event = Event.get_by_id(event.id)
    assert event.revision == 2
    assert event.display_label == "Nguyễn An"
    assert event.canonical_artifact_id != first_artifact
    assert MediaArtifact.get_by_id(first_artifact).sha256


def test_missing_evidence_file_does_not_abort_finalization(canonical_db):
    _, root = canonical_db
    event = create_event("event-missing-evidence")
    aggregator = EventAggregator(CanonicalMediaStore(root / "artifacts"), 0)
    evidence = create_evidence(aggregator, root, event.id, "frame-missing")
    Path(evidence.frame_ref).unlink()
    aggregator.observe(
        observation_id="end-missing",
        event_id=event.id,
        kind="event_ended",
        payload={},
        observed_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1),
    )

    assert aggregator.finalize_due() == [event.id]
    event = Event.get_by_id(event.id)
    assert event.state == "FINALIZED"
    assert event.canonical_artifact_id is None


def test_pending_delivery_protects_expired_artifact(canonical_db):
    _, root = canonical_db
    store = CanonicalMediaStore(root / "artifacts")
    aggregator = EventAggregator(store)
    evidence = create_evidence(aggregator, root, "event-pin", "frame-pin")
    artifact = store.materialize(RenderSpec("event-pin", 1, evidence.id), evidence, "car")
    MediaArtifact.update(
        expires_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=1)
    ).where(MediaArtifact.id == artifact.id).execute()
    now = datetime.datetime.now(datetime.UTC)
    NotificationDelivery.create(
        id="delivery-pin",
        provider="telegram",
        recipient_id="security",
        source_type="event_revision",
        source_id="event-pin",
        payload={},
        status="pending",
        next_attempt=now,
        created_at=now,
        updated_at=now,
        media_artifact_id=artifact.id,
    )
    assert store.cleanup_expired() == 0
    assert Path(artifact.path).exists()
    NotificationDelivery.update(status="sent").where(
        NotificationDelivery.id == "delivery-pin"
    ).execute()
    assert store.cleanup_expired() == 1
    assert not Path(artifact.path).exists()


def test_event_response_accepts_finalized_datetime():
    now = datetime.datetime.now(datetime.UTC)
    response = EventResponse(
        id="event-response",
        label="car",
        sub_label=None,
        camera="car_camera",
        start_time=1.0,
        end_time=2.0,
        false_positive=False,
        zones=[],
        thumbnail=None,
        has_clip=True,
        has_snapshot=True,
        retain_indefinitely=False,
        plus_id=None,
        model_hash=None,
        detector_type=None,
        model_type=None,
        data={},
        finalized_at=now.isoformat(),
    )
    assert response.finalized_at == now
