"""Tests for bounded, event-safe face snapshot handling."""

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

from frigate.const import FACE_DIR
from frigate.events.maintainer import EventProcessor
from frigate.track.object_processing import TrackedObjectProcessor
from frigate.track.tracked_object import TrackedObject
from frigate.util.face_snapshot import (
    CleanupJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    LatestPerObjectWorker,
    SnapshotCommitJob,
    SnapshotCommitted,
    commit_snapshot_job,
    is_track_discontinuity,
    write_face_snapshot_artifact,
)


class FaceSnapshotPipelineTest(unittest.TestCase):
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
                [
                    name
                    for name in os.listdir(os.path.join(folder, "events"))
                    if ".tmp-" in name
                ],
                [],
            )

    def test_stale_active_result_is_not_attached(self) -> None:
        processor = TrackedObjectProcessor.__new__(TrackedObjectProcessor)
        tracked = SimpleNamespace(
            obj_data={"start_time": 10.0, "frame_time": 20.0},
            set_face_snapshot=lambda _: self.fail("stale result was attached"),
        )
        processor.config = SimpleNamespace(
            cameras={
                "face_camera": SimpleNamespace(snapshots=SimpleNamespace(enabled=True))
            }
        )
        processor.camera_states = {
            "face_camera": SimpleNamespace(tracked_objects={"event-1": tracked})
        }
        queued = []
        processor.face_media_publisher = SimpleNamespace(
            publish=lambda payload, topic: queued.append((topic, payload))
        )
        payload = self._payload(frame_time=9.0)
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
            "artifact_path": artifact_path or os.path.join(FACE_DIR, "event.webp"),
        }


if __name__ == "__main__":
    unittest.main()
