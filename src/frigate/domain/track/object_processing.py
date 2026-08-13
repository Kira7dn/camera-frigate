import base64
import datetime
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, cast

import cv2
import numpy as np
from peewee import SQL, DoesNotExist

from frigate.domain.camera.state import CameraState
from frigate.infrastructure.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.infrastructure.comms.dispatcher import Dispatcher
from frigate.infrastructure.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataSubscriber,
    EventMetadataTypeEnum,
)
from frigate.infrastructure.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.infrastructure.comms.inter_process import InterProcessRequestor
from frigate.infrastructure.config import (
    CameraMqttConfig,
    FrigateConfig,
)
from frigate.infrastructure.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.const import (
    FAST_QUEUE_TIMEOUT,
    UPDATE_CAMERA_ACTIVITY,
    UPSERT_REVIEW_SEGMENT,
)
from frigate.application.events.types import EventStateEnum, EventTypeEnum
from frigate.models import Event, ReviewSegment, Timeline
from frigate.domain.ptz.autotrack import PtzAutoTrackerThread
from frigate.domain.track.tracked_object import TrackedObject
from extension.tracker.domain.lifecycle import apply_media_policy, publish_video_detection
from extension.tracker.domain.policy import should_retain_recording, should_save_snapshot
from frigate.util.face_snapshot import (
    FACE_EVENT_STAGING_DIR,
    FaceRecognitionResult,
)
from frigate.util.image import SharedMemoryFrameManager

logger = logging.getLogger(__name__)


class ManualEventState(str, Enum):
    complete = "complete"
    start = "start"
    end = "end"


class TrackedObjectProcessor(threading.Thread):
    def __init__(
        self,
        config: FrigateConfig,
        dispatcher: Dispatcher,
        tracked_objects_queue: MpQueue,
        ptz_autotracker_thread: PtzAutoTrackerThread,
        stop_event: MpEvent,
        face_result_queue: Any | None = None,
        face_commit_queue: Any | None = None,
        face_completion_queue: Any | None = None,
        event_update_queue: Any | None = None,
    ) -> None:
        super().__init__(name="detected_frames_processor")
        self.config = config
        self.dispatcher = dispatcher
        self.tracked_objects_queue = tracked_objects_queue
        self.stop_event: MpEvent = stop_event
        self.face_result_queue = face_result_queue or MpQueue(maxsize=4)
        self.face_commit_queue = face_commit_queue or MpQueue(maxsize=4)
        self.face_completion_queue = face_completion_queue or MpQueue(maxsize=8)
        self.event_update_queue = event_update_queue
        self.face_pending: dict[
            tuple[str, str], tuple[float, FaceRecognitionResult, int]
        ] = {}
        self.camera_states: dict[str, CameraState] = {}
        self.frame_manager = SharedMemoryFrameManager()
        self.last_motion_detected: dict[str, float] = {}
        self.ptz_autotracker_thread = ptz_autotracker_thread

        self.camera_config_subscriber = CameraConfigUpdateSubscriber(
            self.config,
            self.config.cameras,
            [
                CameraConfigUpdateEnum.add,
                CameraConfigUpdateEnum.enabled,
                CameraConfigUpdateEnum.motion,
                CameraConfigUpdateEnum.objects,
                CameraConfigUpdateEnum.remove,
                CameraConfigUpdateEnum.zones,
            ],
        )

        self.requestor = InterProcessRequestor()
        self.detection_publisher = DetectionPublisher(DetectionTypeEnum.all.value)
        self.event_sender = EventUpdatePublisher()
        self.event_end_subscriber = EventEndSubscriber()
        self.sub_label_subscriber = EventMetadataSubscriber(EventMetadataTypeEnum.all)
        self.face_media_publisher = EventMetadataPublisher()

        self.camera_activity: dict[str, dict[str, Any]] = {}
        self.ongoing_manual_events: dict[str, str] = {}

        # {
        #   'zone_name': {
        #       'person': {
        #           'camera_1': 2,
        #           'camera_2': 1
        #       }
        #   }
        # }
        self.zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.active_zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for camera in self.config.cameras.keys():
            self.create_camera_state(camera)

    def create_camera_state(self, camera: str) -> None:
        """Creates a new camera state."""

        def recognition_update(
            camera: str,
            obj: TrackedObject,
            frame_name: str,
            observed_in_frame: bool,
        ) -> tuple[str, dict[str, Any]]:
            """Bind recognition input to the exact tracked-object frame.

            Capture frame names are a bounded ring and may be reused before the
            embeddings process drains its event subscription. An observed
            recognition candidate therefore receives a one-shot shared-memory
            handle. The embeddings consumer owns and deletes that handle after
            its synchronous recognition call.
            """
            data = obj.to_dict()
            data["observed_in_frame"] = observed_in_frame
            if not observed_in_frame:
                return frame_name, data

            camera_config = self.config.cameras[camera]
            is_face = (
                data.get("label") == "person"
                and camera_config.face_recognition.enabled
            )
            is_lpr = (
                data.get("label") in ("car", "motorcycle")
                and camera_config.lpr.enabled
            )
            if not (is_face or is_lpr):
                return frame_name, data

            source = self.frame_manager.get(
                frame_name, camera_config.frame_shape_yuv
            )
            if source is None:
                data["observed_in_frame"] = False
                return frame_name, data

            evidence_name = f"recognition_{camera}_{uuid.uuid4().hex}"
            evidence_buffer = self.frame_manager.create(evidence_name, source.nbytes)
            np.ndarray(
                camera_config.frame_shape_yuv,
                dtype=np.uint8,
                buffer=evidence_buffer,
            )[:] = source
            self.frame_manager.close(evidence_name)
            data["_recognition_evidence_owned"] = True
            return evidence_name, data

        def start(
            camera: str,
            obj: TrackedObject,
            frame_name: str,
            observed_in_frame: bool,
        ) -> None:
            evidence_name, data = recognition_update(
                camera, obj, frame_name, observed_in_frame
            )
            self._publish_event_update(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.start,
                    camera,
                    evidence_name,
                    data,
                )
            )

        def update(
            camera: str,
            obj: TrackedObject,
            frame_name: str,
            observed_in_frame: bool,
        ) -> None:
            apply_media_policy(self.config, camera, obj)
            after = obj.to_dict()
            message = {
                "before": obj.previous,
                "after": after,
                "type": "new" if obj.previous["false_positive"] else "update",
            }
            self.dispatcher.publish("events", json.dumps(message), retain=False)
            obj.previous = after
            evidence_name, data = recognition_update(
                camera, obj, frame_name, observed_in_frame
            )
            self._publish_event_update(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.update,
                    camera,
                    evidence_name,
                    data,
                )
            )

        def autotrack(
            camera: str,
            obj: TrackedObject,
            frame_name: str,
            observed_in_frame: bool,
        ) -> None:
            self.ptz_autotracker_thread.ptz_autotracker.autotrack_object(camera, obj)

        def end(
            camera: str,
            obj: TrackedObject,
            frame_name: str,
            observed_in_frame: bool,
        ) -> None:
            # populate has_snapshot
            apply_media_policy(self.config, camera, obj)

            if obj.face_snapshot_state in ("pending", "committed"):
                # The recognition result was already queued when it arrived.
                # Never let the end callback replace recognition media.
                pass
            else:
                # Existing snapshots retain their normal synchronous path.
                if obj.has_snapshot or obj.has_clip:
                    obj.write_thumbnail_to_disk()
                if obj.has_snapshot:
                    obj.write_snapshot_to_disk()

            if not obj.false_positive:
                message = {
                    "before": obj.previous,
                    "after": obj.to_dict(),
                    "type": "end",
                }
                self.dispatcher.publish("events", json.dumps(message), retain=False)
                self.ptz_autotracker_thread.ptz_autotracker.end_object(camera, obj)

            self._publish_event_update(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.end,
                    camera,
                    frame_name,
                    {**obj.to_dict(), "observed_in_frame": observed_in_frame},
                )
            )

        def snapshot(camera: str, obj: TrackedObject) -> bool:
            mqtt_config: CameraMqttConfig = self.config.cameras[camera].mqtt
            if mqtt_config.enabled and self.should_mqtt_snapshot(camera, obj):
                jpg_bytes, _ = obj.get_img_bytes(
                    ext="jpg",
                    timestamp=mqtt_config.timestamp,
                    bounding_box=mqtt_config.bounding_box,
                    crop=mqtt_config.crop,
                    height=mqtt_config.height,
                    quality=mqtt_config.quality,
                )

                if jpg_bytes is None:
                    logger.warning(
                        f"Unable to send mqtt snapshot for {obj.obj_data['id']}."
                    )
                else:
                    self.dispatcher.publish(
                        f"{camera}/{obj.obj_data['label']}/snapshot",
                        jpg_bytes,
                        retain=True,
                    )

                    if obj.obj_data.get("sub_label"):
                        sub_label = obj.obj_data["sub_label"][0]

                        if sub_label in self.config.model.all_attribute_logos:
                            self.dispatcher.publish(
                                f"{camera}/{sub_label}/snapshot",
                                jpg_bytes,
                                retain=True,
                            )

                    return True

            return False

        def camera_activity(camera: str, activity: dict[str, Any]) -> None:
            last_activity = self.camera_activity.get(camera)

            if not last_activity or activity != last_activity:
                self.camera_activity[camera] = activity
                self.requestor.send_data(UPDATE_CAMERA_ACTIVITY, self.camera_activity)

        camera_state = CameraState(
            camera, self.config, self.frame_manager, self.ptz_autotracker_thread
        )
        camera_state.on("start", start)
        camera_state.on("autotrack", autotrack)
        camera_state.on("update", update)
        camera_state.on("end", end)
        camera_state.on("snapshot", snapshot)
        camera_state.on("camera_activity", camera_activity)
        self.camera_states[camera] = camera_state

    def should_save_snapshot(self, camera: str, obj: TrackedObject) -> bool:
        return should_save_snapshot(self.config, camera, obj)

    def should_retain_recording(self, camera: str, obj: TrackedObject) -> bool:
        return should_retain_recording(self.config, camera, obj)

    def should_mqtt_snapshot(self, camera: str, obj: TrackedObject) -> bool:
        # object never changed position
        if obj.is_stationary():
            return False

        # if there are required zones and there is no overlap
        required_zones = self.config.cameras[camera].mqtt.required_zones
        if len(required_zones) > 0 and not set(obj.entered_zones) & set(required_zones):
            logger.debug(
                f"Not sending mqtt for {obj.obj_data['id']} because it did not enter required zones"
            )
            return False

        return True

    def update_mqtt_motion(
        self, camera: str, frame_time: float, motion_boxes: list
    ) -> None:
        # publish if motion is currently being detected
        if motion_boxes:
            # only send ON if motion isn't already active
            if self.last_motion_detected.get(camera, 0) == 0:
                self.dispatcher.publish(
                    f"{camera}/motion",
                    "ON",
                    retain=False,
                )

            # always updated latest motion
            self.last_motion_detected[camera] = frame_time
        elif self.last_motion_detected.get(camera, 0) > 0:
            mqtt_delay = self.config.cameras[camera].motion.mqtt_off_delay

            # If no motion, make sure the off_delay has passed
            if frame_time - self.last_motion_detected.get(camera, 0) >= mqtt_delay:
                self.dispatcher.publish(
                    f"{camera}/motion",
                    "OFF",
                    retain=False,
                )
                # reset the last_motion so redundant `off` commands aren't sent
                self.last_motion_detected[camera] = 0

    def get_best(self, camera: str, label: str) -> dict[str, Any]:
        # TODO: need a lock here
        camera_state = self.camera_states[camera]
        if label in camera_state.best_objects:
            best_obj = camera_state.best_objects[label]

            if not best_obj.thumbnail_data:
                return {}

            best = best_obj.thumbnail_data.copy()
            best["frame"] = camera_state.frame_cache.get(
                best_obj.thumbnail_data["frame_time"]
            )
            return best
        else:
            return {}

    def get_current_frame(
        self, camera: str, draw_options: dict[str, Any] = {}
    ) -> np.ndarray | None:
        if camera == "birdseye":
            return self.frame_manager.get(
                "birdseye",
                (self.config.birdseye.height * 3 // 2, self.config.birdseye.width),
            )

        if camera not in self.camera_states:
            return None

        return self.camera_states[camera].get_current_frame(draw_options)

    def get_current_frame_time(self, camera: str) -> float:
        """Returns the latest frame time for a given camera."""
        if camera not in self.camera_states:
            return 0.0

        return self.camera_states[camera].current_frame_time

    def set_sub_label(
        self, event_id: str, sub_label: str | None, score: float | None
    ) -> None:
        """Update sub label for given event id."""
        tracked_obj: TrackedObject | None = None

        for state in self.camera_states.values():
            tracked_obj = state.tracked_objects.get(event_id)

            if tracked_obj is not None:
                break

        try:
            event: Event | None = Event.get(Event.id == event_id)
        except DoesNotExist:
            event = None

        if not tracked_obj and not event:
            return

        if tracked_obj:
            tracked_obj.obj_data["sub_label"] = (sub_label, score)

        if event:
            event.sub_label = cast(Any, sub_label)
            data = cast(dict[str, Any], event.data)
            if sub_label is None:
                data["sub_label_score"] = None
            elif score is not None:
                data["sub_label_score"] = score
            cast(Any, event).data = data
            event.save()

            # update timeline items
            Timeline.update(
                data=Timeline.data.update({"sub_label": (sub_label, score)})
            ).where(Timeline.source_id == event_id).execute()

            # only update ended review segments
            # manually updating a sub_label from the UI is only possible for ended tracked objects
            try:
                review_segment = ReviewSegment.get(
                    (
                        SQL(
                            "json_extract(data, '$.detections') LIKE ?",
                            [f'%"{event_id}"%'],
                        )
                    )
                    & (ReviewSegment.end_time.is_null(False))
                )

                segment_data = review_segment.data
                detection_ids = segment_data.get("detections", [])

                # Rebuild objects list and sync sub_labels
                objects_list = []
                sub_labels = set()
                events = Event.select(Event.id, Event.label, Event.sub_label).where(
                    Event.id.in_(detection_ids)  # type: ignore[call-arg, misc]
                )
                for det_event in events:
                    if det_event.sub_label:
                        sub_labels.add(det_event.sub_label)
                        objects_list.append(
                            f"{det_event.label}-verified"
                        )  # eg, "bird-verified"
                    else:
                        objects_list.append(det_event.label)  # eg, "bird"

                segment_data["sub_labels"] = list(sub_labels)
                segment_data["objects"] = objects_list

                updated_data = {
                    ReviewSegment.id.name: review_segment.id,
                    ReviewSegment.camera.name: review_segment.camera,
                    ReviewSegment.start_time.name: review_segment.start_time,
                    ReviewSegment.end_time.name: review_segment.end_time,
                    ReviewSegment.severity.name: review_segment.severity,
                    ReviewSegment.thumb_path.name: review_segment.thumb_path,
                    ReviewSegment.data.name: segment_data,
                }

                self.requestor.send_data(UPSERT_REVIEW_SEGMENT, updated_data)
                logger.debug(
                    f"Updated sub_label for event {event_id} in review segment {review_segment.id}"
                )

            except DoesNotExist:
                logger.debug(
                    f"No review segment found with event ID {event_id} when updating sub_label"
                )

    def set_face_snapshot(self, payload: dict[str, Any]) -> None:
        """Validate and attach snapshot metadata without filesystem I/O."""
        required = {
            "camera",
            "event_id",
            "frame_time",
            "person_box",
            "face_box",
            "sub_label",
            "face_score",
            "artifact_path",
        }
        if not required.issubset(payload):
            logger.warning("Ignoring malformed face snapshot payload: %s", payload)
            return

        event_id = str(payload["event_id"])
        camera = str(payload["camera"])
        source_frame_time = float(payload["frame_time"])
        artifact_path = os.path.abspath(str(payload["artifact_path"]))
        person_box = tuple(int(value) for value in payload["person_box"])
        face_box = tuple(int(value) for value in payload["face_box"])

        if (
            not event_id
            or not camera
            or source_frame_time <= 0
            or len(person_box) != 4
            or len(face_box) != 4
        ):
            logger.warning("Ignoring malformed face snapshot payload: %s", payload)
            return

        configured_camera = self.config.cameras.get(camera)
        if configured_camera is None or not configured_camera.snapshots.enabled:
            self._queue_face_cleanup(artifact_path)
            return

        # The producer is an internal Frigate process, but keep the path
        # constrained to the face artifact directory before touching media.
        staging_dir = os.path.abspath(FACE_EVENT_STAGING_DIR)
        if not artifact_path.startswith(staging_dir + os.sep) or not os.path.isfile(
            artifact_path
        ):
            logger.warning("Ignoring invalid face snapshot artifact: %s", artifact_path)
            return
        snapshot = {
            "path": artifact_path,
            "frame_time": source_frame_time,
            "box": person_box,
            "face_box": face_box,
            "area": max(0, person_box[2] - person_box[0])
            * max(0, person_box[3] - person_box[1]),
            "score": float(payload["face_score"]),
            "attributes": [],
            "current_estimated_speed": 0,
            "face_score": float(payload["face_score"]),
            "sub_label": str(payload["sub_label"]),
            "transaction_id": str(payload.get("transaction_id", "")),
        }
        result = FaceRecognitionResult(
            camera=camera,
            event_id=event_id,
            frame_time=source_frame_time,
            person_box=person_box,
            face_box=face_box,
            sub_label=str(payload["sub_label"]),
            face_score=float(payload["face_score"]),
            artifact_path=artifact_path,
            transaction_id=str(payload.get("transaction_id", "")),
        )

        tracked_obj: TrackedObject | None = None
        state = self.camera_states.get(camera)
        if state is not None:
            tracked_obj = state.tracked_objects.get(event_id)

        if tracked_obj is not None:
            track_start = float(tracked_obj.obj_data.get("start_time", 0))
            track_frame = float(tracked_obj.obj_data.get("frame_time", 0))
            if source_frame_time < track_start or source_frame_time > track_frame:
                self._queue_face_cleanup(artifact_path)
                return
            obsolete_artifact = tracked_obj.set_face_snapshot(snapshot)
            if obsolete_artifact == artifact_path:
                self._queue_face_cleanup(artifact_path)
                return
            if obsolete_artifact:
                self._queue_face_cleanup(obsolete_artifact)
            tracked_obj.has_snapshot = True
            if not self._queue_face_commit(result):
                tracked_obj.face_snapshot = None
                tracked_obj.face_snapshot_state = "failed"
                self._queue_face_cleanup(artifact_path)
                return
            if not hasattr(self, "face_pending"):
                self.face_pending = {}
            self.face_pending[(camera, event_id)] = (
                time.monotonic() + 10,
                result,
                0,
            )
            logger.debug(
                "Face snapshot commit pending for active event %s at frame %.3f",
                event_id,
                source_frame_time,
            )
            return

        if not self._queue_face_commit(result):
            logger.warning("Face snapshot media queue full for %s", event_id)
            self._queue_face_cleanup(artifact_path)
        else:
            if not hasattr(self, "face_pending"):
                self.face_pending = {}
            self.face_pending[(camera, event_id)] = (
                time.monotonic() + 10,
                result,
                0,
            )

    def apply_face_snapshot_completion(self, payload: dict[str, Any]) -> None:
        """Publish identity only after canonical media and DB are complete."""
        try:
            camera = str(payload["camera"])
            event_id = str(payload["event_id"])
            frame_time = float(payload["frame_time"])
            status = str(payload["status"])
        except (KeyError, TypeError, ValueError):
            return
        state = self.camera_states.get(camera)
        getattr(self, "face_pending", {}).pop((camera, event_id), None)
        tracked_obj = state.tracked_objects.get(event_id) if state else None
        if tracked_obj is None or tracked_obj.face_snapshot is None:
            return
        snapshot = tracked_obj.face_snapshot
        if float(snapshot.get("frame_time", 0)) != frame_time:
            return
        if status != "committed":
            tracked_obj.face_snapshot = None
            tracked_obj.face_snapshot_state = "failed"
            tracked_obj.has_snapshot = self.should_save_snapshot(camera, tracked_obj)
            return

        snapshot["path"] = str(payload["canonical_path"])
        tracked_obj.face_snapshot_state = "committed"
        tracked_obj.obj_data["sub_label"] = (
            str(payload["sub_label"]),
            float(payload["face_score"]),
        )
        self.requestor.send_data(
            "tracked_object_update",
            json.dumps(
                {
                    "type": "face",
                    "name": payload["sub_label"],
                    "score": payload["face_score"],
                    "id": event_id,
                    "camera": camera,
                    "timestamp": frame_time,
                    "source_frame_time": frame_time,
                    "frame_ref": payload["canonical_path"],
                    "person_box": payload["person_box"],
                    "face_box": payload["face_box"],
                "evidence_id": payload.get("transaction_id", ""),
                }
            ),
        )

    @staticmethod
    def _result_from_snapshot(
        camera: str, event_id: str, snapshot: dict[str, Any]
    ) -> FaceRecognitionResult:
        return FaceRecognitionResult(
            camera=camera,
            event_id=event_id,
            frame_time=float(snapshot["frame_time"]),
            person_box=tuple(snapshot["box"]),
            face_box=tuple(snapshot["face_box"]),
            sub_label=str(snapshot["sub_label"]),
            face_score=float(snapshot["face_score"]),
            artifact_path=str(snapshot["path"]),
            transaction_id=str(snapshot.get("transaction_id", "")),
        )

    def _queue_face_commit(self, result: FaceRecognitionResult) -> bool:
        face_commit_queue = getattr(self, "face_commit_queue", None)
        if face_commit_queue is None:
            self.face_media_publisher.publish(
                result.as_payload(),
                EventMetadataTypeEnum.face_snapshot_commit.value,
            )
            return True
        try:
            face_commit_queue.put_nowait(
                {"type": "commit", "payload": result.as_payload()}
            )
            return True
        except queue.Full:
            return False

    def _publish_event_update(self, payload: Any) -> None:
        """Send canonical persistence through a bounded reliable queue."""
        event_update_queue = getattr(self, "event_update_queue", None)
        if event_update_queue is not None:
            try:
                event_update_queue.put(payload, timeout=0.25)
            except queue.Full:
                event_state = payload[1] if len(payload) > 1 else None
                if event_state in (EventStateEnum.start, EventStateEnum.end):
                    while not self.stop_event.is_set():
                        try:
                            event_update_queue.put(payload, timeout=0.25)
                            break
                        except queue.Full:
                            continue
                else:
                    logger.warning("Dropping coalescible event update from full queue")
        self.event_sender.publish(payload)

    def _queue_face_cleanup(self, *paths: str) -> None:
        face_commit_queue = getattr(self, "face_commit_queue", None)
        if face_commit_queue is None:
            self.face_media_publisher.publish(
                {"paths": paths},
                EventMetadataTypeEnum.face_snapshot_cleanup.value,
            )
            return
        try:
            face_commit_queue.put_nowait({"type": "cleanup", "paths": paths})
        except queue.Full:
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def _drain_face_lifecycle_queues(self) -> None:
        """Apply reliable face candidates and commit acknowledgements."""
        while True:
            try:
                payload = self.face_result_queue.get_nowait()
            except queue.Empty:
                break
            self.set_face_snapshot(payload)
        while True:
            try:
                payload = self.face_completion_queue.get_nowait()
            except queue.Empty:
                break
            self.apply_face_snapshot_completion(payload)

    def _expire_face_pending(self) -> None:
        """Retry one missing acknowledgement, then restore tracking fallback."""
        now = time.monotonic()
        for key, (deadline, result, retries) in list(self.face_pending.items()):
            if now < deadline:
                continue
            if retries == 0 and self._queue_face_commit(result):
                self.face_pending[key] = (now + 10, result, 1)
                continue
            camera, event_id = key
            state = self.camera_states.get(camera)
            tracked_obj = state.tracked_objects.get(event_id) if state else None
            if tracked_obj is not None and tracked_obj.face_snapshot_state == "pending":
                tracked_obj.face_snapshot = None
                tracked_obj.face_snapshot_state = "failed"
                tracked_obj.has_snapshot = self.should_save_snapshot(camera, tracked_obj)
            self._queue_face_cleanup(result.artifact_path)
            self.face_pending.pop(key, None)

    def set_object_attribute(
        self,
        event_id: str,
        field_name: str,
        field_value: str | None,
        score: float | None,
    ) -> None:
        """Update attribute for given event id."""
        tracked_obj: TrackedObject | None = None

        for state in self.camera_states.values():
            tracked_obj = state.tracked_objects.get(event_id)

            if tracked_obj is not None:
                break

        try:
            event: Event | None = Event.get(Event.id == event_id)
        except DoesNotExist:
            event = None

        if not tracked_obj and not event:
            return

        if tracked_obj:
            tracked_obj.obj_data[field_name] = (
                field_value,
                score,
            )

        if event:
            data = cast(dict[str, Any], event.data)
            data[field_name] = field_value
            if field_value is None:
                data[f"{field_name}_score"] = None
            elif score is not None:
                data[f"{field_name}_score"] = score
            cast(Any, event).data = data
            event.save()

    def save_lpr_snapshot(self, payload: tuple) -> None:
        # save the snapshot image
        (frame, event_id, camera) = payload

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        self.camera_states[camera].save_manual_event_image(
            img, event_id, "license_plate", {}
        )

    def create_manual_event(self, payload: tuple) -> None:
        (
            frame_time,
            camera_name,
            label,
            event_id,
            include_recording,
            score,
            sub_label,
            duration,
            source_type,
            draw,
            pre_capture,
        ) = payload

        # save the snapshot image
        self.camera_states[camera_name].save_manual_event_image(
            None, event_id, label, draw
        )
        end_time = frame_time + duration if duration is not None else None
        start_time = (
            frame_time - self.config.cameras[camera_name].record.event_pre_capture
            if pre_capture is None
            else frame_time - pre_capture
        )

        # send event to event maintainer
        self._publish_event_update(
            (
                EventTypeEnum.api,
                EventStateEnum.start,
                camera_name,
                "",
                {
                    "id": event_id,
                    "label": label,
                    "sub_label": sub_label,
                    "score": score,
                    "camera": camera_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "has_clip": self.config.cameras[camera_name].record.enabled
                    and include_recording,
                    "has_snapshot": True,
                    "snapshot_clean": True,
                    "snapshot_frame_time": frame_time,
                    "type": source_type,
                    "draw": draw,
                },
            )
        )

        if source_type == "api":
            self.ongoing_manual_events[event_id] = camera_name
            self.detection_publisher.publish(
                (
                    camera_name,
                    frame_time,
                    {
                        "state": (
                            ManualEventState.complete
                            if end_time
                            else ManualEventState.start
                        ),
                        "label": f"{label}: {sub_label}" if sub_label else label,
                        "event_id": event_id,
                        "end_time": end_time,
                    },
                ),
                DetectionTypeEnum.api.value,
            )

    def create_lpr_event(self, payload: tuple) -> None:
        (
            frame_time,
            camera_name,
            label,
            event_id,
            include_recording,
            score,
            sub_label,
            plate,
        ) = payload

        # send event to event maintainer
        self._publish_event_update(
            (
                EventTypeEnum.api,
                EventStateEnum.start,
                camera_name,
                "",
                {
                    "id": event_id,
                    "label": label,
                    "sub_label": sub_label,
                    "score": score,
                    "camera": camera_name,
                    "start_time": frame_time
                    - self.config.cameras[camera_name].record.event_pre_capture,
                    "end_time": None,
                    "has_clip": self.config.cameras[camera_name].record.enabled
                    and include_recording,
                    "has_snapshot": True,
                    "snapshot_clean": True,
                    "type": "api",
                    "recognized_license_plate": plate,
                    "recognized_license_plate_score": score,
                },
            )
        )

        self.ongoing_manual_events[event_id] = camera_name
        self.detection_publisher.publish(
            (
                camera_name,
                frame_time,
                {
                    "state": ManualEventState.start,
                    "label": f"{label}: {sub_label}" if sub_label else label,
                    "event_id": event_id,
                    "end_time": None,
                },
            ),
            DetectionTypeEnum.lpr.value,
        )

    def end_manual_event(self, payload: tuple) -> None:
        (event_id, end_time) = payload

        self._publish_event_update(
            (
                EventTypeEnum.api,
                EventStateEnum.end,
                None,
                "",
                {"id": event_id, "end_time": end_time},
            )
        )

        if event_id in self.ongoing_manual_events:
            self.detection_publisher.publish(
                (
                    self.ongoing_manual_events[event_id],
                    end_time,
                    {
                        "state": ManualEventState.end,
                        "event_id": event_id,
                        "end_time": end_time,
                    },
                ),
                DetectionTypeEnum.api.value,
            )
            self.ongoing_manual_events.pop(event_id)

    def force_end_all_events(self, camera: str, camera_state: CameraState) -> None:
        """Ends all active events on camera when disabling."""
        last_frame_name = camera_state.previous_frame_id
        for obj_id, obj in list(camera_state.tracked_objects.items()):
            if "end_time" not in obj.obj_data:
                logger.debug(f"Camera {camera} disabled, ending active event {obj_id}")
                obj.obj_data["end_time"] = datetime.datetime.now().timestamp()
                # end callbacks
                for callback in camera_state.callbacks["end"]:
                    callback(camera, obj, last_frame_name, False)

                # camera activity callbacks
                for callback in camera_state.callbacks["camera_activity"]:
                    callback(
                        camera,
                        {"enabled": False, "motion": 0, "objects": []},
                    )

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._drain_face_lifecycle_queues()
            self._expire_face_pending()
            # check for config updates
            updated_topics = self.camera_config_subscriber.check_for_updates()

            # a single drain can carry several topics at once, so add and
            # remove are handled independently rather than as exclusive branches
            for camera in updated_topics.get("add", []):
                self.config.cameras[camera] = (
                    self.camera_config_subscriber.camera_configs[camera]
                )
                self.create_camera_state(camera)

            if "remove" in updated_topics:
                for camera in updated_topics["remove"]:
                    camera_state = self.camera_states.get(camera)
                    if camera_state is None:
                        continue

                    camera_state.shutdown()
                    self.camera_states.pop(camera)
                    self.camera_activity.pop(camera, None)
                    self.last_motion_detected.pop(camera, None)

                self.requestor.send_data(UPDATE_CAMERA_ACTIVITY, self.camera_activity)

            # manage camera disabled state
            for camera, config in self.config.cameras.items():
                if not config.enabled_in_config:
                    continue

                current_enabled = config.enabled
                camera_state = self.camera_states.get(camera)
                if camera_state is None:
                    continue

                camera_state = self.camera_states[camera]

                if camera_state.prev_enabled and not current_enabled:
                    logger.debug(f"Not processing objects for disabled camera {camera}")
                    self.force_end_all_events(camera, camera_state)

                camera_state.prev_enabled = current_enabled

                if not current_enabled:
                    continue

            # check for sub label updates
            while True:
                update = self.sub_label_subscriber.check_for_update(timeout=0)

                if not update:
                    break

                (raw_topic, payload) = update

                if not raw_topic or not payload:
                    break

                topic = str(raw_topic)

                if topic.endswith(EventMetadataTypeEnum.sub_label.value):
                    (event_id, sub_label, score) = payload
                    self.set_sub_label(event_id, sub_label, score)
                elif topic.endswith(EventMetadataTypeEnum.face_snapshot.value):
                    self.set_face_snapshot(payload)
                elif topic.endswith(
                    EventMetadataTypeEnum.face_snapshot_committed.value
                ):
                    self.apply_face_snapshot_completion(payload)
                elif topic.endswith(EventMetadataTypeEnum.attribute.value):
                    (event_id, field_name, field_value, score) = payload
                    self.set_object_attribute(event_id, field_name, field_value, score)
                elif topic.endswith(EventMetadataTypeEnum.lpr_event_create.value):
                    self.create_lpr_event(payload)
                elif topic.endswith(EventMetadataTypeEnum.save_lpr_snapshot.value):
                    self.save_lpr_snapshot(payload)
                elif topic.endswith(EventMetadataTypeEnum.manual_event_create.value):
                    self.create_manual_event(payload)
                elif topic.endswith(EventMetadataTypeEnum.manual_event_end.value):
                    self.end_manual_event(payload)

            try:
                (
                    camera,
                    frame_name,
                    frame_time,
                    current_tracked_objects,
                    motion_boxes,
                    regions,
                ) = self.tracked_objects_queue.get(True, 1)
            except queue.Empty:
                continue

            camera_config = self.config.cameras.get(camera)
            if camera_config is None:
                continue

            if not camera_config.enabled:
                logger.debug(f"Camera {camera} disabled, skipping update")
                continue

            camera_state = self.camera_states.get(camera)
            if camera_state is None:
                continue

            camera_state.update(
                frame_name, frame_time, current_tracked_objects, motion_boxes, regions
            )

            self.update_mqtt_motion(camera, frame_time, motion_boxes)

            tracked_objects = [
                o.to_dict() for o in camera_state.tracked_objects.values()
            ]

            # publish info on this frame
            publish_video_detection(
                self.detection_publisher,
                camera,
                frame_name,
                frame_time,
                tracked_objects,
                motion_boxes,
                regions,
            )

            # cleanup event finished queue
            while not self.stop_event.is_set():
                update = self.event_end_subscriber.check_for_update(
                    timeout=FAST_QUEUE_TIMEOUT
                )

                if not update:
                    break

                event_id, camera, _ = update
                self.camera_states[camera].finished(event_id)

        # shut down camera states
        for state in self.camera_states.values():
            for tracked_obj in state.tracked_objects.values():
                snapshot = tracked_obj.face_snapshot
                if snapshot and snapshot.get("path"):
                    self._queue_face_cleanup(str(snapshot["path"]))
            state.shutdown()

        self.requestor.stop()
        self.detection_publisher.stop()
        self.event_sender.stop()
        self.event_end_subscriber.stop()
        self.sub_label_subscriber.stop()
        self.face_media_publisher.stop()
        self.camera_config_subscriber.stop()

        logger.info("Exiting object processor...")
