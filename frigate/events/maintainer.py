import logging
import os
import queue
import threading
import time
from collections import Counter, OrderedDict
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataSubscriber,
    EventMetadataTypeEnum,
)
from frigate.comms.events_updater import EventEndPublisher, EventUpdateSubscriber
from frigate.config import FrigateConfig
from frigate.config.classification import ObjectClassificationType
from frigate.const import CLIPS_DIR, REPLAY_CAMERA_PREFIX, THUMB_DIR
from frigate.events.types import EventStateEnum, EventTypeEnum
from frigate.models import Event
from frigate.util.builtin import to_relative_box
from frigate.util.face_snapshot import (
    FACE_EVENT_STAGING_DIR,
    CleanupJob,
    FaceRecognitionResult,
    LatestPerObjectWorker,
    SnapshotCommitJob,
    SnapshotCommitted,
    SnapshotFailed,
    cleanup_paths,
    commit_snapshot_job,
    finalize_snapshot_commit,
    load_snapshot_journal,
    rollback_snapshot_commit,
)

logger = logging.getLogger(__name__)


def should_update_db(prev_event: dict[str, Any], current_event: dict[str, Any]) -> bool:
    """If current_event has updated fields and (clip or snapshot)."""
    # If event is ending and was previously saved, always update to set end_time
    # This ensures events are properly ended even when alerts/detections are disabled
    # mid-event (which can cause has_clip/has_snapshot to become False)
    if (
        prev_event["end_time"] is None
        and current_event["end_time"] is not None
        and (prev_event["has_clip"] or prev_event["has_snapshot"])
    ):
        return True

    if current_event["has_clip"] or current_event["has_snapshot"]:
        # if this is the first time has_clip or has_snapshot turned true
        if not prev_event["has_clip"] and not prev_event["has_snapshot"]:
            return True
        # or if any of the following values changed
        if (
            prev_event["top_score"] != current_event["top_score"]
            or prev_event["entered_zones"] != current_event["entered_zones"]
            or prev_event["end_time"] != current_event["end_time"]
            or prev_event["average_estimated_speed"]
            != current_event["average_estimated_speed"]
            or prev_event["velocity_angle"] != current_event["velocity_angle"]
            or prev_event["recognized_license_plate"]
            != current_event["recognized_license_plate"]
            or prev_event["path_data"] != current_event["path_data"]
        ):
            return True
    return False


def should_update_state(
    prev_event: dict[str, Any], current_event: dict[str, Any]
) -> bool:
    """If current event should update state, but not necessarily update the db."""
    if prev_event["stationary"] != current_event["stationary"]:
        return True

    if prev_event["attributes"] != current_event["attributes"]:
        return True

    if prev_event["sub_label"] != current_event["sub_label"]:
        return True

    if set(prev_event["current_zones"]) != set(current_event["current_zones"]):
        return True

    return False


class EventProcessor(threading.Thread):
    def __init__(
        self,
        config: FrigateConfig,
        timeline_queue: Queue,
        stop_event: MpEvent,
        face_commit_queue: Any | None = None,
        face_completion_queue: Any | None = None,
        event_update_queue: Any | None = None,
    ):
        super().__init__(name="event_processor")
        self.config = config
        self.timeline_queue = timeline_queue
        self.events_in_process: dict[str, dict[str, Any]] = {}
        self.stop_event = stop_event
        self.face_commit_queue = face_commit_queue or Queue(maxsize=4)
        self.face_completion_queue = face_completion_queue or Queue(maxsize=8)
        self.event_update_queue = event_update_queue

        self.event_receiver = EventUpdateSubscriber()
        self.event_end_publisher = EventEndPublisher()
        self.face_snapshot_receiver = EventMetadataSubscriber(EventMetadataTypeEnum.all)
        self.face_snapshot_publisher = EventMetadataPublisher()
        self.face_snapshot_worker = LatestPerObjectWorker(
            self._process_snapshot_job,
            max_objects=4,
            name="event_face_media_committer",
            drop_handler=self._drop_snapshot_job,
        )
        self.deferred_face_jobs: OrderedDict[
            tuple[str, str], tuple[SnapshotCommitJob, float]
        ] = OrderedDict()
        self.face_snapshot_metrics: Counter[str] = Counter()
        self.face_snapshot_states: dict[tuple[str, str], str] = {}
        self.face_transactions_in_progress: set[str] = set()
        self.deferred_face_completions: OrderedDict[
            tuple[str, str, float], dict[str, Any]
        ] = OrderedDict()
        self.last_face_metrics_log = time.monotonic()
        self.recovered_face_commits = load_snapshot_journal()

    def run(self) -> None:
        # A crash has no final end message. Close recovered events at their
        # last persisted observation instead of inventing a fixed duration.
        for open_event in Event.select().where(Event.end_time == None):
            last_seen = float(
                (open_event.data or {}).get(
                    "last_seen_frame_time", open_event.start_time + 30
                )
            )
            open_event.end_time = max(open_event.start_time, last_seen)
            open_event.save(only=[Event.end_time])

        while not self.stop_event.is_set():
            self._flush_face_completions()
            self._drain_face_snapshot_requests()
            self._retry_deferred_face_jobs()
            self._apply_snapshot_completions()
            self._log_face_snapshot_metrics()
            if self.event_update_queue is None:
                update = self.event_receiver.check_for_update(timeout=1)
            else:
                try:
                    update = self.event_update_queue.get(timeout=0.25)
                except queue.Empty:
                    update = None

            if update == None:
                continue

            source_type, event_type, camera, _, event_data = update  # type: ignore[misc]

            logger.debug(
                f"Event received: {source_type} {event_type} {camera} {event_data['id']}"
            )

            if source_type == EventTypeEnum.tracked_object:
                id = event_data["id"]
                self.timeline_queue.put(
                    (
                        camera,
                        source_type,
                        event_type,
                        self.events_in_process.get(id),
                        event_data,
                    )
                )

                # if this is the first message, just store it and continue, its not time to insert it in the db
                if (
                    event_type == EventStateEnum.start
                    or id not in self.events_in_process
                ):
                    self.events_in_process[id] = event_data
                    continue

                self.handle_object_detection(event_type, camera, event_data)
            elif source_type == EventTypeEnum.api:
                self.timeline_queue.put(
                    (
                        camera,
                        source_type,
                        event_type,
                        {},
                        event_data,
                    )
                )

                self.handle_external_detection(event_type, event_data)

        self.event_receiver.stop()
        self.event_end_publisher.stop()
        for job, _ in self.deferred_face_jobs.values():
            self._drop_snapshot_job(job)
        self.deferred_face_jobs.clear()
        self.face_snapshot_worker.stop()
        self._apply_snapshot_completions()
        self.face_snapshot_receiver.stop()
        self.face_snapshot_publisher.stop()
        logger.info("Exiting event processor...")

    def _drain_face_snapshot_requests(self) -> None:
        face_commit_queue = getattr(self, "face_commit_queue", None)
        if face_commit_queue is not None:
            while True:
                try:
                    item = face_commit_queue.get_nowait()
                except queue.Empty:
                    break
                if item.get("type") == "cleanup":
                    paths = tuple(str(path) for path in item.get("paths", []))
                    if not self.face_snapshot_worker.submit_control(CleanupJob(paths)):
                        cleanup_paths(CleanupJob(paths))
                    self.face_snapshot_metrics["released"] += len(paths)
                elif item.get("type") == "commit":
                    self._accept_snapshot_payload(item.get("payload", {}))

        while True:
            update = self.face_snapshot_receiver.check_for_update(timeout=0)
            if not update:
                return
            topic, payload = update
            if not topic or payload is None:
                return
            if topic.endswith(EventMetadataTypeEnum.face_snapshot_cleanup.value):
                paths = tuple(str(path) for path in payload.get("paths", []))
                if not self.face_snapshot_worker.submit_control(CleanupJob(paths)):
                    cleanup_paths(CleanupJob(paths))
                self.face_snapshot_metrics["released"] += len(paths)
            elif topic.endswith(EventMetadataTypeEnum.face_snapshot_commit.value):
                self._accept_snapshot_payload(payload)

    def _accept_snapshot_payload(self, payload: dict[str, Any]) -> None:
        try:
            result = FaceRecognitionResult(
                camera=str(payload["camera"]),
                event_id=str(payload["event_id"]),
                frame_time=float(payload["frame_time"]),
                person_box=tuple(int(value) for value in payload["person_box"]),
                face_box=tuple(int(value) for value in payload["face_box"]),
                sub_label=str(payload["sub_label"]),
                face_score=float(payload["face_score"]),
                artifact_path=os.path.abspath(str(payload["artifact_path"])),
                transaction_id=str(payload.get("transaction_id", "")),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Ignoring malformed face snapshot commit request")
            self.face_snapshot_metrics["rejected"] += 1
            return
        transactions = getattr(self, "face_transactions_in_progress", None)
        if transactions is None:
            transactions = set()
            self.face_transactions_in_progress = transactions
        if result.transaction_id and result.transaction_id in transactions:
            self.face_snapshot_metrics["duplicate"] += 1
            return
        staging_dir = os.path.abspath(FACE_EVENT_STAGING_DIR)
        if (
            len(result.person_box) != 4
            or len(result.face_box) != 4
            or result.frame_time <= 0
            or not result.artifact_path.startswith(staging_dir + os.sep)
            or not os.path.isfile(result.artifact_path)
        ):
            self.face_snapshot_metrics["rejected"] += 1
            return
        job = SnapshotCommitJob(
            result=result,
            canonical_path=os.path.join(
                CLIPS_DIR, f"{result.camera}-{result.event_id}-clean.webp"
            ),
            thumbnail_path=os.path.join(
                THUMB_DIR, result.camera, f"{result.event_id}.webp"
            ),
        )
        if result.transaction_id:
            transactions.add(result.transaction_id)
        self.face_snapshot_states[result.key] = "pending"
        self._submit_or_defer_snapshot(job)

    def _submit_or_defer_snapshot(self, job: SnapshotCommitJob) -> None:
        try:
            event = Event.get(Event.id == job.result.event_id)
        except Event.DoesNotExist:
            previous = self.deferred_face_jobs.pop(job.result.key, None)
            if previous is not None:
                self._drop_snapshot_job(previous[0])
                self.face_snapshot_metrics["replaced"] += 1
                self._publish_snapshot_completion(
                    previous[0].result, "failed", "replaced"
                )
            elif len(self.deferred_face_jobs) >= 4:
                self._drop_snapshot_job(job)
                self.face_snapshot_metrics["rejected"] += 1
                self._publish_snapshot_completion(
                    job.result, "failed", "deferred_queue_full"
                )
                return
            self.deferred_face_jobs[job.result.key] = (job, time.monotonic() + 5)
            self.face_snapshot_metrics["late_result"] += 1
            return
        if event.camera != job.result.camera:
            self._drop_snapshot_job(job)
            self.face_snapshot_metrics["camera_mismatch"] += 1
            self._publish_snapshot_completion(job.result, "failed", "camera_mismatch")
            return
        existing = (event.data or {}).get("face_snapshot_frame_time", 0)
        if float(existing or 0) >= job.result.frame_time:
            self._drop_snapshot_job(job)
            self.face_snapshot_metrics["stale_result"] += 1
            self._publish_snapshot_completion(job.result, "failed", "stale")
            return
        if not self.face_snapshot_worker.submit(job.result.key, job):
            self._drop_snapshot_job(job)
            self.face_snapshot_metrics["rejected"] += 1
            self._publish_snapshot_completion(job.result, "failed", "queue_full")

    def _retry_deferred_face_jobs(self) -> None:
        now = time.monotonic()
        for key, (job, deadline) in list(self.deferred_face_jobs.items()):
            try:
                Event.get(Event.id == job.result.event_id)
            except Event.DoesNotExist:
                if now >= deadline:
                    self.deferred_face_jobs.pop(key, None)
                    self._drop_snapshot_job(job)
                    self.face_snapshot_metrics["rejected"] += 1
                    self._publish_snapshot_completion(
                        job.result, "failed", "event_not_found"
                    )
                continue
            self.deferred_face_jobs.pop(key, None)
            self._submit_or_defer_snapshot(job)

    def _apply_snapshot_completions(self) -> None:
        completions = getattr(
            self, "recovered_face_commits", []
        ) + self.face_snapshot_worker.drain_results()
        self.recovered_face_commits = []
        for completion in completions:
            if isinstance(completion, SnapshotFailed):
                self.face_snapshot_metrics["failed"] += 1
                self._publish_snapshot_completion(
                    completion.result, "failed", completion.reason
                )
                continue
            if not isinstance(completion, SnapshotCommitted):
                continue
            result = completion.result
            try:
                event = Event.get(Event.id == result.event_id)
            except Event.DoesNotExist:
                rollback_snapshot_commit(completion)
                self.face_snapshot_metrics["rejected"] += 1
                self._publish_snapshot_completion(result, "failed", "event_not_found")
                continue
            if event.camera != result.camera:
                rollback_snapshot_commit(completion)
                self.face_snapshot_metrics["camera_mismatch"] += 1
                self._publish_snapshot_completion(result, "failed", "camera_mismatch")
                continue
            data = event.data or {}
            if float(data.get("face_snapshot_frame_time", 0) or 0) >= result.frame_time:
                if (
                    float(data.get("face_snapshot_frame_time", 0) or 0)
                    == result.frame_time
                    and data.get("face_snapshot_sub_label") == result.sub_label
                ):
                    finalize_snapshot_commit(completion)
                    self.face_snapshot_metrics["recovered"] += 1
                    self._publish_snapshot_completion(
                        result,
                        "committed",
                        canonical_path=completion.canonical_path,
                        thumbnail_path=completion.thumbnail_path,
                    )
                    continue
                rollback_snapshot_commit(completion)
                self.face_snapshot_metrics["stale_result"] += 1
                self._publish_snapshot_completion(result, "failed", "stale")
                continue
            camera_config = self.config.cameras.get(result.camera)
            if camera_config is None:
                rollback_snapshot_commit(completion)
                self.face_snapshot_metrics["rejected"] += 1
                self._publish_snapshot_completion(
                    result, "failed", "camera_not_configured"
                )
                continue
            width = camera_config.detect.width
            height = camera_config.detect.height
            if width is None or height is None:
                rollback_snapshot_commit(completion)
                self.face_snapshot_metrics["rejected"] += 1
                self._publish_snapshot_completion(
                    result, "failed", "detect_dimensions_unavailable"
                )
                continue
            data["box"] = to_relative_box(width, height, result.person_box)
            data["face_box"] = to_relative_box(width, height, result.face_box)
            data["region"] = None
            data["score"] = result.face_score
            data["attributes"] = []
            data["snapshot_frame_time"] = result.frame_time
            data["face_snapshot_frame_time"] = result.frame_time
            data["snapshot_area"] = max(
                0, result.person_box[2] - result.person_box[0]
            ) * max(0, result.person_box[3] - result.person_box[1])
            data["snapshot_estimated_speed"] = 0
            data["snapshot_clean"] = True
            data["snapshot_source"] = "face_recognition"
            data["face_snapshot_score"] = result.face_score
            data["face_snapshot_sub_label"] = result.sub_label
            data["sub_label_score"] = result.face_score
            event.data = data
            event.sub_label = result.sub_label
            event.has_snapshot = True
            try:
                event.save()
            except Exception:
                logger.exception(
                    "Unable to save face snapshot metadata for %s", result.event_id
                )
                self.face_snapshot_metrics["failed"] += 1
                rollback_snapshot_commit(completion)
                self._publish_snapshot_completion(result, "failed", "database")
                continue
            finalize_snapshot_commit(completion)
            self.face_snapshot_metrics["committed"] += 1
            self._publish_snapshot_completion(
                result,
                "committed",
                canonical_path=completion.canonical_path,
                thumbnail_path=completion.thumbnail_path,
            )

    @staticmethod
    def _process_snapshot_job(
        job: SnapshotCommitJob | CleanupJob,
    ) -> SnapshotCommitted | SnapshotFailed | None:
        if isinstance(job, CleanupJob):
            cleanup_paths(job)
            return None
        try:
            return commit_snapshot_job(job)
        except Exception as error:
            logger.exception(
                "Unable to commit face snapshot for %s", job.result.event_id
            )
            return SnapshotFailed(job.result, type(error).__name__)

    def _publish_snapshot_completion(
        self,
        result: FaceRecognitionResult,
        status: str,
        reason: str | None = None,
        canonical_path: str | None = None,
        thumbnail_path: str | None = None,
    ) -> None:
        if not hasattr(self, "face_snapshot_states"):
            self.face_snapshot_states = {}
        self.face_snapshot_states[result.key] = status
        transactions = getattr(self, "face_transactions_in_progress", None)
        if transactions is not None and result.transaction_id:
            transactions.discard(result.transaction_id)
        payload = {
            **result.as_payload(),
            "status": status,
            "reason": reason,
            "canonical_path": canonical_path,
            "thumbnail_path": thumbnail_path,
        }
        publisher = getattr(self, "face_snapshot_publisher", None)
        completion_queue = getattr(self, "face_completion_queue", None)
        if completion_queue is not None:
            try:
                completion_queue.put_nowait(payload)
            except queue.Full:
                deferred = getattr(self, "deferred_face_completions", None)
                if deferred is not None:
                    deferred[(result.camera, result.event_id, result.frame_time)] = payload
        if publisher is not None:
            publisher.publish(
                payload, EventMetadataTypeEnum.face_snapshot_committed.value
            )
        self.face_snapshot_states.pop(result.key, None)

    def _flush_face_completions(self) -> None:
        """Retry reliable acknowledgements without blocking event persistence."""
        for key, payload in list(self.deferred_face_completions.items()):
            try:
                self.face_completion_queue.put_nowait(payload)
            except queue.Full:
                return
            self.deferred_face_completions.pop(key, None)

    def _drop_snapshot_job(self, job: SnapshotCommitJob | CleanupJob) -> None:
        if isinstance(job, CleanupJob):
            cleanup_paths(job)
        else:
            cleanup_paths(CleanupJob((job.result.artifact_path,)))
        self.face_snapshot_metrics["released"] += 1

    def _log_face_snapshot_metrics(self) -> None:
        now = time.monotonic()
        if now - self.last_face_metrics_log < 60:
            return
        worker = self.face_snapshot_worker.stats()
        logger.info(
            "Face snapshot metrics pending=%d replaced=%d rejected=%d committed=%d failed=%d released=%d stale=%d late=%d camera_mismatch=%d",
            worker.get("pending", 0) + len(self.deferred_face_jobs),
            worker.get("replaced", 0) + self.face_snapshot_metrics["replaced"],
            worker.get("rejected", 0) + self.face_snapshot_metrics["rejected"],
            self.face_snapshot_metrics["committed"],
            worker.get("failed", 0),
            self.face_snapshot_metrics["released"],
            self.face_snapshot_metrics["stale_result"],
            self.face_snapshot_metrics["late_result"],
            self.face_snapshot_metrics["camera_mismatch"],
        )
        self.last_face_metrics_log = now

    def handle_object_detection(
        self,
        event_type: str,
        camera: str,
        event_data: dict[str, Any],
    ) -> None:
        """handle tracked object event updates."""
        updated_db = False

        if should_update_db(self.events_in_process[event_data["id"]], event_data):
            updated_db = True
            camera_config = self.config.cameras.get(camera)
            if camera_config is None:
                return

            width = camera_config.detect.width
            height = camera_config.detect.height

            if width is None or height is None:
                return

            first_detector = list(self.config.detectors.values())[0]

            start_time = event_data["start_time"]
            end_time = (
                None if event_data["end_time"] is None else event_data["end_time"]
            )
            snapshot = event_data["snapshot"]
            face_snapshot_pending = (
                snapshot is not None and snapshot.get("face_score") is not None
            )
            # score of the snapshot
            score = (
                None if snapshot is None or face_snapshot_pending else snapshot["score"]
            )
            # detection region in the snapshot
            region = (
                None
                if snapshot is None or face_snapshot_pending
                else to_relative_box(
                    width,
                    height,
                    snapshot["region"],
                )
            )
            # bounding box for the snapshot
            box = (
                None
                if snapshot is None or face_snapshot_pending
                else to_relative_box(
                    width,
                    height,
                    snapshot["box"],
                )
            )

            attributes = (
                None
                if snapshot is None or face_snapshot_pending
                else [
                    {
                        "box": to_relative_box(
                            width,
                            height,
                            a["box"],
                        ),
                        "label": a["label"],
                        "score": a["score"],
                    }
                    for a in snapshot["attributes"]
                ]
            )
            snapshot_frame_time = (
                None
                if snapshot is None or face_snapshot_pending
                else snapshot["frame_time"]
            )
            snapshot_area = (
                None if snapshot is None or face_snapshot_pending else snapshot["area"]
            )
            snapshot_estimated_speed = (
                None
                if snapshot is None or face_snapshot_pending
                else snapshot["current_estimated_speed"]
            )

            # keep these from being set back to false because the event
            # may have started while recordings/snapshots/alerts/detections were enabled
            # this would be an issue for long running events
            if self.events_in_process[event_data["id"]]["has_clip"]:
                event_data["has_clip"] = True
            if self.events_in_process[event_data["id"]]["has_snapshot"]:
                event_data["has_snapshot"] = True

            event = {
                Event.id: event_data["id"],
                Event.label: event_data["label"],
                Event.camera: camera,
                Event.start_time: start_time,
                Event.end_time: end_time,
                Event.zones: list(event_data["entered_zones"]),
                Event.thumbnail: event_data.get("thumbnail"),
                Event.has_clip: event_data["has_clip"],
                Event.has_snapshot: event_data["has_snapshot"]
                if not face_snapshot_pending
                else False,
                Event.model_hash: first_detector.model.model_hash
                if first_detector.model
                else None,
                Event.model_type: first_detector.model.model_type
                if first_detector.model
                else None,
                Event.detector_type: first_detector.type,
                Event.data: {
                    "box": box,
                    "region": region,
                    "score": score,
                    "top_score": event_data["top_score"],
                    "attributes": attributes,
                    "snapshot_clean": event_data.get("snapshot_clean", False),
                    "snapshot_frame_time": snapshot_frame_time,
                    "snapshot_area": snapshot_area,
                    "snapshot_estimated_speed": snapshot_estimated_speed,
                    "snapshot_source": "tracking"
                    if snapshot is not None and not face_snapshot_pending
                    else None,
                    "face_snapshot_score": None,
                    "face_snapshot_sub_label": None,
                    "average_estimated_speed": event_data["average_estimated_speed"],
                    "velocity_angle": event_data["velocity_angle"],
                    "type": "object",
                    "max_severity": event_data.get("max_severity"),
                    "path_data": event_data.get("path_data"),
                    "last_seen_frame_time": float(
                        event_data.get("frame_time")
                        or event_data.get("end_time")
                        or event_data["start_time"]
                    ),
                },
            }

            # only overwrite the sub_label in the database if it's set
            if event_data.get("sub_label") is not None and not face_snapshot_pending:
                event[Event.sub_label] = event_data["sub_label"][0]
                event[Event.data]["sub_label_score"] = event_data["sub_label"][1]

            # only overwrite the recognized_license_plate in the database if it's set
            if event_data.get("recognized_license_plate") is not None:
                event[Event.data]["recognized_license_plate"] = event_data[
                    "recognized_license_plate"
                ][0]
                event[Event.data]["recognized_license_plate_score"] = event_data[
                    "recognized_license_plate"
                ][1]

            # only overwrite attribute-type custom model fields in the database if they're set
            for name, model_config in self.config.classification.custom.items():
                if (
                    model_config.object_config
                    and model_config.object_config.classification_type
                    == ObjectClassificationType.attribute
                ):
                    value = event_data.get(name)
                    if value is not None:
                        event[Event.data][name] = value[0]
                        event[Event.data][f"{name}_score"] = value[1]

            # A completed recognition snapshot is canonical. Object updates
            # and the end callback may still carry the older tracking snapshot
            # or the pre-commit pending payload, but must never replace its
            # identity or frame-aligned media metadata.
            try:
                existing_event = Event.get(Event.id == event_data["id"])
            except Event.DoesNotExist:
                existing_event = None
            if existing_event is not None:
                existing_data = existing_event.data or {}
                if existing_data.get("snapshot_source") == "face_recognition":
                    for key in (
                        "box",
                        "face_box",
                        "region",
                        "score",
                        "attributes",
                        "snapshot_clean",
                        "snapshot_frame_time",
                        "snapshot_area",
                        "snapshot_estimated_speed",
                        "snapshot_source",
                        "face_snapshot_frame_time",
                        "face_snapshot_score",
                        "face_snapshot_sub_label",
                        "sub_label_score",
                    ):
                        if key in existing_data:
                            event[Event.data][key] = existing_data[key]
                    event[Event.has_snapshot] = True
                    if existing_event.sub_label is not None:
                        event[Event.sub_label] = existing_event.sub_label

            (
                Event.insert(event)
                .on_conflict(
                    conflict_target=[Event.id],
                    update=event,
                )
                .execute()
            )

        # check if the stored event_data should be updated
        if updated_db or should_update_state(
            self.events_in_process[event_data["id"]], event_data
        ):
            # update the stored copy for comparison on future update messages
            self.events_in_process[event_data["id"]] = event_data

        if event_type == EventStateEnum.end:
            del self.events_in_process[event_data["id"]]
            self.event_end_publisher.publish((event_data["id"], camera, updated_db))  # type: ignore[arg-type]

    def handle_external_detection(
        self, event_type: EventStateEnum, event_data: dict[str, Any]
    ) -> None:
        # Skip replay cameras
        if event_data.get("camera", "").startswith(REPLAY_CAMERA_PREFIX):
            return

        if event_type == EventStateEnum.start:
            event = {
                Event.id: event_data["id"],
                Event.label: event_data["label"],
                Event.sub_label: event_data["sub_label"],
                Event.camera: event_data["camera"],
                Event.start_time: event_data["start_time"],
                Event.end_time: event_data["end_time"],
                Event.thumbnail: event_data.get("thumbnail"),
                Event.has_clip: event_data["has_clip"],
                Event.has_snapshot: event_data["has_snapshot"],
                Event.zones: [],
                Event.data: {
                    "type": event_data["type"],
                    "score": event_data["score"],
                    "top_score": event_data["score"],
                    "snapshot_clean": event_data.get("snapshot_clean", False),
                },
            }
            if event_data.get("draw") is not None:
                event[Event.data]["draw"] = event_data["draw"]
            if event_data.get("recognized_license_plate") is not None:
                event[Event.data]["recognized_license_plate"] = event_data[
                    "recognized_license_plate"
                ]
                event[Event.data]["recognized_license_plate_score"] = event_data[
                    "score"
                ]
            Event.insert(event).execute()
        elif event_type == EventStateEnum.end:
            event = {
                Event.id: event_data["id"],
                Event.end_time: event_data["end_time"],
            }

            try:
                Event.update(event).where(Event.id == event_data["id"]).execute()
            except Exception:
                logger.warning(f"Failed to update manual event: {event_data['id']}")
