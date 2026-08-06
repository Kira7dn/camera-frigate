"""Tests for bounded, event-safe face snapshot handling."""

import json
import os
import tempfile
import threading
import time
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from frigate.api.classification import _sanitize_face_name, get_faces
from frigate.embeddings.maintainer import EmbeddingMaintainer, FaceRealTimeProcessor
from frigate.events.maintainer import EventProcessor
from frigate.track.object_processing import TrackedObjectProcessor
from frigate.track.tracked_object import TrackedObject
from frigate.util.face_snapshot import (
    FACE_EVENT_STAGING_DIR,
    CleanupJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    LatestPerObjectWorker,
    SnapshotCommitJob,
    SnapshotCommitted,
    commit_snapshot_job,
    finalize_snapshot_commit,
    is_face_identity_directory,
    is_track_discontinuity,
    is_unknown_face_attempt,
    parse_face_attempt_filename,
    rollback_snapshot_commit,
    write_face_snapshot_artifact,
)


class FaceSnapshotPipelineTest(unittest.TestCase):
    def test_reserved_and_hidden_face_names_are_rejected(self) -> None:
        for name in ("train", "events", "staging", "face-events", ".hidden", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _sanitize_face_name(name)
        self.assertEqual(_sanitize_face_name("Person One"), "Person_One")

    def test_face_library_api_exposes_only_unknown_training_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "alice"))
            os.makedirs(os.path.join(folder, "train"))
            os.makedirs(os.path.join(folder, "events"))
            open(os.path.join(folder, "alice", "reference.webp"), "wb").close()
            open(
                os.path.join(folder, "train", "event-1-1.0-unknown-0.4.webp"),
                "wb",
            ).close()
            open(
                os.path.join(folder, "train", "event-2-2.0-unknown-0.3.webp"),
                "wb",
            ).close()
            open(os.path.join(folder, "events", "runtime.webp"), "wb").close()

            with (
                patch("frigate.api.classification.FACE_DIR", folder),
                patch(
                    "frigate.api.classification._identified_face_event_ids",
                    return_value={"event-2"},
                ),
            ):
                response = get_faces()

            payload = json.loads(response.body)
            self.assertEqual(payload["alice"], ["reference.webp"])
            self.assertEqual(payload["train"], ["event-1-1.0-unknown-0.4.webp"])
            self.assertNotIn("events", payload)

    def test_recent_face_attempt_contract_only_accepts_unknown(self) -> None:
        self.assertTrue(
            is_unknown_face_attempt("camera-1234.5-unknown-0.42.webp")
        )
        self.assertFalse(
            is_unknown_face_attempt("camera-1234.5-alice-0.95.webp")
        )
        self.assertFalse(is_unknown_face_attempt("unknown.txt"))
        self.assertEqual(
            parse_face_attempt_filename(
                "event-with-hyphens-1234.5-unknown-0.42.webp"
            ),
            ("event-with-hyphens", "unknown"),
        )

    def test_only_unknown_face_attempts_are_queued(self) -> None:
        processor = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        processor.config = SimpleNamespace(
            face_recognition=SimpleNamespace(save_attempts=200)
        )
        queued = []
        processor.face_attempt_worker = SimpleNamespace(
            submit=lambda key, job: queued.append((key, job)) or True
        )
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        processor.queue_face_attempt(
            "face_camera", frame, "event-1", 10.0, "alice", 0.95
        )
        processor.queue_face_attempt(
            "face_camera", frame, "event-2", 11.0, "unknown", 0.42
        )

        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0][0], ("face_camera", "event-2"))
        self.assertEqual(queued[0][1].sub_label, "unknown")

    def test_empty_metadata_sentinel_is_ignored(self) -> None:
        processor = EventProcessor.__new__(EventProcessor)
        processor.face_snapshot_receiver = SimpleNamespace(
            check_for_update=lambda timeout=0: (None, None)
        )
        processor._drain_face_snapshot_requests()

    def test_event_committer_cleanup_control_is_not_a_commit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "artifact.webp")
            cv2.imwrite(artifact, np.zeros((4, 4, 3), dtype=np.uint8))
            result = EventProcessor._process_snapshot_job(CleanupJob((artifact,)))
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(artifact))

    def test_track_discontinuity_and_stale_frame(self) -> None:
        self.assertTrue(
            is_track_discontinuity((0, 0, 100, 100), (500, 500, 600, 600), 0.2)
        )
        self.assertTrue(
            is_track_discontinuity((0, 0, 100, 100), (10, 10, 110, 110), 2.1)
        )
        self.assertFalse(
            is_track_discontinuity((0, 0, 100, 100), (10, 10, 110, 110), 0.2)
        )

    def test_duplicate_result_keeps_newer_candidate(self) -> None:
        obj = SimpleNamespace(face_snapshot={"frame_time": 20.0, "path": "new.webp"})
        stale = {"frame_time": 19.0, "path": "stale.webp"}
        obsolete = TrackedObject.set_face_snapshot(obj, stale)
        self.assertEqual(obsolete, "stale.webp")
        self.assertEqual(obj.face_snapshot["path"], "new.webp")

    def test_queue_is_latest_per_object_and_bounded_to_four_objects(self) -> None:
        started = threading.Event()
        release = threading.Event()
        handled: list[str] = []

        def handler(job: str) -> None:
            started.set()
            release.wait(2)
            handled.append(job)

        worker = LatestPerObjectWorker(handler, max_objects=4)
        try:
            self.assertTrue(worker.submit(("cam", "1"), "first"))
            self.assertTrue(started.wait(1))
            self.assertTrue(worker.submit(("cam", "2"), "old"))
            self.assertTrue(worker.submit(("cam", "2"), "latest"))
            self.assertTrue(worker.submit(("cam", "3"), "third"))
            self.assertTrue(worker.submit(("cam", "4"), "fourth"))
            self.assertFalse(worker.submit(("cam", "5"), "overflow"))
            release.set()
            deadline = time.time() + 2
            while len(handled) < 4 and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(handled, ["first", "latest", "third", "fourth"])
        finally:
            release.set()
            worker.stop()

    def test_atomic_artifact_commit_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            job = FaceSnapshotJob(
                camera="face_camera",
                event_id="event-1",
                frame_time=10.5,
                person_box=(0, 0, 4, 4),
                face_box=(1, 1, 3, 3),
                sub_label="person_1",
                face_score=0.99,
                frame=np.zeros((6, 4), dtype=np.uint8),
            )
            result = write_face_snapshot_artifact(job, folder)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isfile(result.artifact_path))
            self.assertEqual(
                [name for name in os.listdir(folder) if ".tmp-" in name],
                [],
            )

    def test_stale_active_result_is_not_attached(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "event.webp")
            cv2.imwrite(artifact, np.zeros((4, 4, 3), dtype=np.uint8))
            processor = TrackedObjectProcessor.__new__(TrackedObjectProcessor)
            tracked = SimpleNamespace(
                obj_data={"start_time": 10.0, "frame_time": 20.0},
                set_face_snapshot=lambda _: self.fail("stale result was attached"),
            )
            processor.config = SimpleNamespace(
                cameras={
                    "face_camera": SimpleNamespace(
                        snapshots=SimpleNamespace(enabled=True)
                    )
                }
            )
            processor.camera_states = {
                "face_camera": SimpleNamespace(tracked_objects={"event-1": tracked})
            }
            queued = []
            processor.face_media_publisher = SimpleNamespace(
                publish=lambda payload, topic: queued.append((topic, payload))
            )
            payload = self._payload(frame_time=9.0, artifact_path=artifact)
            with patch(
                "frigate.track.object_processing.FACE_EVENT_STAGING_DIR", folder
            ):
                processor.set_face_snapshot(payload)
            self.assertEqual(queued[0][1]["paths"], (payload["artifact_path"],))

    def test_late_result_updates_only_matching_event_and_commits_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            clips = os.path.join(folder, "clips")
            thumbs = os.path.join(folder, "thumbs")
            os.makedirs(clips)
            artifact = os.path.join(folder, "artifact.webp")
            cv2.imwrite(artifact, np.zeros((20, 30, 3), dtype=np.uint8))
            event = SimpleNamespace(
                id="event-1",
                camera="face_camera",
                data={"snapshot_frame_time": 1.0},
                has_snapshot=False,
                save=lambda: None,
            )
            result = FaceRecognitionResult(
                **self._payload(frame_time=10.0, artifact_path=artifact)
            )
            completion = commit_snapshot_job(
                SnapshotCommitJob(
                    result=result,
                    canonical_path=os.path.join(
                        clips, "face_camera-event-1-clean.webp"
                    ),
                    thumbnail_path=os.path.join(thumbs, "face_camera", "event-1.webp"),
                )
            )
            processor = EventProcessor.__new__(EventProcessor)
            processor.config = SimpleNamespace(
                cameras={
                    "face_camera": SimpleNamespace(
                        detect=SimpleNamespace(width=30, height=20)
                    )
                }
            )
            processor.face_snapshot_worker = SimpleNamespace(
                drain_results=lambda: [completion]
            )
            processor.face_snapshot_metrics = Counter()
            with (
                patch("frigate.events.maintainer.Event.get", return_value=event),
            ):
                processor._apply_snapshot_completions()

            self.assertTrue(
                os.path.isfile(os.path.join(clips, "face_camera-event-1-clean.webp"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(thumbs, "face_camera", "event-1.webp"))
            )
            self.assertEqual(event.data["snapshot_frame_time"], 10.0)
            self.assertEqual(event.sub_label, "person_1")
            self.assertFalse(os.path.exists(artifact))
            self.assertEqual(
                [
                    name
                    for root, _, files in os.walk(folder)
                    for name in files
                    if ".tmp-" in name
                ],
                [],
            )

    def test_late_camera_mismatch_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "artifact.webp")
            cv2.imwrite(artifact, np.zeros((4, 4, 3), dtype=np.uint8))
            result = FaceRecognitionResult(**self._payload(artifact_path=artifact))
            canonical = os.path.join(folder, "canonical.webp")
            thumbnail = os.path.join(folder, "thumbnail.webp")
            cv2.imwrite(canonical, np.zeros((4, 4, 3), dtype=np.uint8))
            cv2.imwrite(thumbnail, np.zeros((4, 4, 3), dtype=np.uint8))
            completion = SnapshotCommitted(result, canonical, thumbnail)
            event = SimpleNamespace(id="event-1", camera="another_camera", data={})
            processor = EventProcessor.__new__(EventProcessor)
            processor.face_snapshot_worker = SimpleNamespace(
                drain_results=lambda: [completion]
            )
            processor.face_snapshot_metrics = Counter()
            with patch("frigate.events.maintainer.Event.get", return_value=event):
                processor._apply_snapshot_completions()
            self.assertFalse(os.path.exists(canonical))
            self.assertFalse(os.path.exists(thumbnail))

    def test_commit_job_is_immutable_and_cleanup_does_not_consume_slot(self) -> None:
        result = FaceRecognitionResult(**self._payload())
        job = SnapshotCommitJob(result, "canonical.webp", "thumbnail.webp")
        with self.assertRaises(FrozenInstanceError):
            job.canonical_path = "changed.webp"  # type: ignore[misc]

        started = threading.Event()
        release = threading.Event()

        def handler(value):
            if not isinstance(value, CleanupJob):
                started.set()
                release.wait(2)

        worker = LatestPerObjectWorker(handler, max_objects=1)
        try:
            self.assertTrue(worker.submit(("cam", "1"), "work"))
            self.assertTrue(started.wait(1))
            self.assertTrue(worker.submit_control(CleanupJob(())))
            self.assertFalse(worker.submit(("cam", "2"), "overflow"))
        finally:
            release.set()
            worker.stop()

    def test_face_attempts_are_rate_limited_and_confirm_on_second_match(self) -> None:
        processor = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        processor.config = SimpleNamespace(
            cameras={
                "face_camera": SimpleNamespace(
                    face_recognition=SimpleNamespace(enabled=True, min_area=1)
                )
            }
        )
        processor.face_config = SimpleNamespace(
            unknown_score=0.5, recognition_threshold=0.8, min_faces=2
        )
        processor.requires_face_detection = False
        processor.face_tracks = {}
        processor.face_counters = Counter()
        processor.last_face_metrics_log = time.monotonic()
        processor.metrics = SimpleNamespace(face_rec_fps=SimpleNamespace(value=0))
        processor.faces_per_second = SimpleNamespace(eps=lambda: 0)
        processor.recognizer = SimpleNamespace(classify=lambda _: ("alice", 0.95))
        processor.queue_face_attempt = lambda *args: None
        processor._FaceRealTimeProcessor__update_metrics = lambda duration: None
        queued = []
        processor.face_snapshot_worker = SimpleNamespace(
            submit=lambda key, job: queued.append((key, job)) or True
        )
        frame = np.zeros((6, 4), dtype=np.uint8)
        obj = {
            "camera": "face_camera",
            "id": "event-1",
            "label": "person",
            "box": (0, 0, 4, 4),
            "area": 16,
            "sub_label": None,
            "current_attributes": [
                {"label": "face", "score": 0.9, "box": (1, 1, 3, 3)}
            ],
        }
        for frame_time in (100.0, 100.2, 100.5):
            processor.process_frame({**obj, "frame_time": frame_time}, frame)
        self.assertEqual(processor.face_counters["classified"], 2)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0][1].frame_time, 100.5)

    def test_detection_stream_limits_face_work_to_four_people(self) -> None:
        processor = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        processed = []
        processor.face_tracks = {}
        processor.submit_frame = lambda obj, frame: processed.append(obj["id"])
        maintainer = EmbeddingMaintainer.__new__(EmbeddingMaintainer)
        people = [
            {"id": str(index), "label": "person", "box": (0, 0, 2, 2), "area": index}
            for index in range(1, 6)
        ]
        updates = iter(
            [
                (
                    "video",
                    (
                        "face_camera",
                        "stale-frame",
                        9.8,
                        [{"id": "stale", "label": "person", "box": (0, 0, 2, 2)}],
                        [],
                        [],
                    ),
                ),
                (
                    "video",
                    ("face_camera", "frame", 10.0, people, [], []),
                ),
                (None, None),
            ]
        )
        maintainer.detection_subscriber = SimpleNamespace(
            check_for_update=lambda timeout=None: next(updates)
        )
        camera_config = SimpleNamespace(
            type="camera",
            objects=SimpleNamespace(track=["person"]),
            face_recognition=SimpleNamespace(enabled=True),
            frame_shape_yuv=(6, 4),
        )
        maintainer.config = SimpleNamespace(
            cameras={"face_camera": camera_config},
            classification=SimpleNamespace(custom={}),
        )
        maintainer.realtime_processors = [processor]
        maintainer.frame_manager = SimpleNamespace(
            get=lambda *args: np.zeros((6, 4), dtype=np.uint8),
            close=lambda *args: None,
        )
        maintainer._process_frame_updates()
        self.assertEqual(processed, ["5", "4", "3", "2"])

    def test_reserved_and_hidden_face_directories_are_not_identities(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for name in ("alice", "train", "events", "staging", ".hidden"):
                os.makedirs(os.path.join(folder, name))
            self.assertTrue(
                is_face_identity_directory("alice", os.path.join(folder, "alice"))
            )
            for name in ("train", "events", "staging", ".hidden"):
                self.assertFalse(
                    is_face_identity_directory(name, os.path.join(folder, name))
                )

    def test_face_state_reconciles_with_active_detection_set(self) -> None:
        processor = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        processor.face_tracks = {
            ("face_camera", "active"): object(),
            ("face_camera", "ended"): object(),
            ("other_camera", "other"): object(),
        }
        processor.expire_missing_objects("face_camera", {"active"})
        self.assertEqual(
            set(processor.face_tracks),
            {("face_camera", "active"), ("other_camera", "other")},
        )

    def test_active_identity_is_published_only_after_commit_ack(self) -> None:
        processor = TrackedObjectProcessor.__new__(TrackedObjectProcessor)
        sent = []
        processor.requestor = SimpleNamespace(
            send_data=lambda topic, payload: sent.append((topic, payload))
        )
        tracked = SimpleNamespace(
            face_snapshot={
                "frame_time": 10.0,
                "path": "/tmp/cache/face-events/event.webp",
            },
            face_snapshot_state="pending",
            obj_data={"label": "person"},
        )
        processor.camera_states = {
            "face_camera": SimpleNamespace(tracked_objects={"event-1": tracked})
        }
        payload = {
            **self._payload(),
            "status": "committed",
            "canonical_path": "/media/frigate/clips/face_camera-event-1-clean.webp",
        }
        processor.apply_face_snapshot_completion(payload)
        self.assertEqual(tracked.face_snapshot_state, "committed")
        self.assertEqual(tracked.obj_data["sub_label"], ("person_1", 0.99))
        self.assertEqual(len(sent), 1)

    def test_failed_commit_does_not_publish_identity(self) -> None:
        processor = TrackedObjectProcessor.__new__(TrackedObjectProcessor)
        processor.should_save_snapshot = lambda camera, tracked: False
        processor.requestor = SimpleNamespace(
            send_data=lambda *args: self.fail("failed identity was published")
        )
        tracked = SimpleNamespace(
            face_snapshot={"frame_time": 10.0, "path": "artifact.webp"},
            face_snapshot_state="pending",
            obj_data={"label": "person"},
            has_snapshot=True,
        )
        processor.camera_states = {
            "face_camera": SimpleNamespace(tracked_objects={"event-1": tracked})
        }
        processor.apply_face_snapshot_completion(
            {**self._payload(), "status": "failed", "reason": "database"}
        )
        self.assertIsNone(tracked.face_snapshot)
        self.assertEqual(tracked.face_snapshot_state, "failed")
        self.assertNotIn("sub_label", tracked.obj_data)

    def test_atomic_commit_restores_previous_media_on_thumbnail_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "artifact.webp")
            canonical = os.path.join(folder, "canonical.webp")
            thumbnail = os.path.join(folder, "thumbnail.webp")
            cv2.imwrite(artifact, np.full((10, 10, 3), 200, dtype=np.uint8))
            cv2.imwrite(canonical, np.full((10, 10, 3), 10, dtype=np.uint8))
            cv2.imwrite(thumbnail, np.full((10, 10, 3), 20, dtype=np.uint8))
            original_replace = os.replace

            def fail_thumbnail(source, destination):
                if destination == thumbnail and ".tmp-" in source:
                    raise OSError("thumbnail replace failed")
                return original_replace(source, destination)

            job = SnapshotCommitJob(
                FaceRecognitionResult(**self._payload(artifact_path=artifact)),
                canonical,
                thumbnail,
            )
            with (
                patch("frigate.util.face_snapshot.os.replace", fail_thumbnail),
                self.assertRaises(OSError),
            ):
                commit_snapshot_job(job)
            self.assertLess(cv2.imread(canonical).mean(), 15)
            self.assertLess(cv2.imread(thumbnail).mean(), 25)
            self.assertFalse(os.path.exists(artifact))

    def test_media_transaction_rolls_back_until_database_commit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "artifact.webp")
            canonical = os.path.join(folder, "canonical.webp")
            thumbnail = os.path.join(folder, "thumbnail.webp")
            journal = os.path.join(folder, "journal")
            cv2.imwrite(artifact, np.full((10, 10, 3), 200, dtype=np.uint8))
            cv2.imwrite(canonical, np.full((10, 10, 3), 10, dtype=np.uint8))
            cv2.imwrite(thumbnail, np.full((10, 10, 3), 20, dtype=np.uint8))
            job = SnapshotCommitJob(
                FaceRecognitionResult(
                    **self._payload(artifact_path=artifact),
                    transaction_id="transaction-1",
                ),
                canonical,
                thumbnail,
            )
            with patch(
                "frigate.util.face_snapshot.FACE_COMMIT_JOURNAL_DIR", journal
            ):
                completion = commit_snapshot_job(job)
            self.assertGreater(cv2.imread(canonical).mean(), 190)
            self.assertTrue(os.path.exists(completion.canonical_backup))
            rollback_snapshot_commit(completion)
            self.assertLess(cv2.imread(canonical).mean(), 15)
            self.assertLess(cv2.imread(thumbnail).mean(), 25)
            self.assertFalse(os.path.exists(artifact))

    def test_media_transaction_finalizes_only_after_database_commit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artifact = os.path.join(folder, "artifact.webp")
            canonical = os.path.join(folder, "canonical.webp")
            thumbnail = os.path.join(folder, "thumbnail.webp")
            journal = os.path.join(folder, "journal")
            cv2.imwrite(artifact, np.full((10, 10, 3), 200, dtype=np.uint8))
            job = SnapshotCommitJob(
                FaceRecognitionResult(
                    **self._payload(artifact_path=artifact),
                    transaction_id="transaction-2",
                ),
                canonical,
                thumbnail,
            )
            with patch(
                "frigate.util.face_snapshot.FACE_COMMIT_JOURNAL_DIR", journal
            ):
                completion = commit_snapshot_job(job)
            finalize_snapshot_commit(completion)
            self.assertTrue(os.path.exists(canonical))
            self.assertTrue(os.path.exists(thumbnail))
            self.assertFalse(os.path.exists(artifact))
            self.assertFalse(os.path.exists(completion.journal_path))

    @staticmethod
    def _payload(frame_time: float = 10.0, artifact_path: str | None = None) -> dict:
        return {
            "camera": "face_camera",
            "event_id": "event-1",
            "frame_time": frame_time,
            "person_box": (0, 0, 10, 10),
            "face_box": (2, 2, 8, 8),
            "sub_label": "person_1",
            "face_score": 0.99,
            "artifact_path": artifact_path
            or os.path.join(FACE_EVENT_STAGING_DIR, "event.webp"),
        }


if __name__ == "__main__":
    unittest.main()
