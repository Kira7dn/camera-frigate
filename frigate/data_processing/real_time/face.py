"""Handle processing images for face detection and recognition."""

import base64
import datetime
import json
import logging
import os
import shutil
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.face_snapshot import (
    FaceAttemptJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    FaceTrackState,
    FaceVote,
    LatestPerObjectWorker,
    is_track_discontinuity,
    reap_stale_staging,
    write_face_attempt,
    write_face_snapshot_artifact,
)
from frigate.util.image import area

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


MAX_DETECTION_HEIGHT = 1080
MAX_FACES_ATTEMPTS_AFTER_REC = 6
MAX_FACE_ATTEMPTS = 12
FACE_METRICS_LOG_INTERVAL = 30


class FaceRealTimeProcessor(RealTimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics)
        self.face_config = config.face_recognition
        self.requestor = requestor
        self.sub_label_publisher = sub_label_publisher
        self.face_detector: cv2.FaceDetectorYN | None = None
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.face_tracks: dict[tuple[str, str], FaceTrackState] = {}
        self.face_counters: Counter[str] = Counter()
        self.last_face_metrics_log = time.monotonic()
        removed = reap_stale_staging(FACE_DIR)
        if removed:
            logger.info("Removed %d stale face staging artifacts", removed)
        self.face_snapshot_worker = LatestPerObjectWorker(
            lambda job: write_face_snapshot_artifact(job, FACE_DIR),
            max_objects=4,
        )
        self.face_attempt_worker = LatestPerObjectWorker(
            write_face_attempt,
            max_objects=4,
            name="face_attempt_worker",
        )
        self.recognizer: FaceRecognizer
        self.faces_per_second = EventsPerSecond()
        self.inference_speed = InferenceSpeed(self.metrics.face_rec_speed)

        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")

        download_path = os.path.join(MODEL_CACHE_DIR, "facedet")
        self.model_files = {
            "facedet.onnx": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx",
            "landmarkdet.yaml": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/landmarkdet.yaml",
        }

        if not all(
            os.path.exists(os.path.join(download_path, n))
            for n in self.model_files.keys()
        ):
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            self.downloader = ModelDownloader(
                model_name="facedet",
                download_path=download_path,
                file_names=list(self.model_files.keys()),
                download_func=self.__download_models,
                complete_func=self.__build_detector,
            )
            self.downloader.ensure_model_files()
        else:
            self.__build_detector()

        self.label_map: dict[int, str] = {}

        if self.face_config.model_size == "small":
            self.recognizer = FaceNetRecognizer(self.config)
        else:
            self.recognizer = ArcFaceRecognizer(self.config)

        self.recognizer.build()
        identities, training_images = self.__face_library_stats()
        logger.info(
            "Face recognition initialized model=%s device=%s "
            "library_identities=%d training_images=%d face_dir=%s",
            self.face_config.model_size,
            self.face_config.device,
            identities,
            training_images,
            FACE_DIR,
        )
        if identities == 0 or training_images == 0:
            logger.warning(
                "Face recognition library is empty; detected faces cannot be "
                "matched until training images are added under %s/<identity>",
                FACE_DIR,
            )

    CONFIG_UPDATE_TOPIC = "config/face_recognition"

    def update_config(self, topic: str, payload: Any) -> None:
        """Update face recognition config at runtime."""
        if topic != self.CONFIG_UPDATE_TOPIC:
            return

        previous_min_area = self.config.face_recognition.min_area
        self.config.face_recognition = payload
        self.face_config = payload

        for camera_config in self.config.cameras.values():
            if camera_config.face_recognition.min_area == previous_min_area:
                camera_config.face_recognition.min_area = payload.min_area

        logger.debug("Face recognition config updated dynamically")

    def __download_models(self, path: str) -> None:
        try:
            file_name = os.path.basename(path)
            # conditionally import ModelDownloader
            from frigate.util.downloader import ModelDownloader

            ModelDownloader.download_from_url(self.model_files[file_name], path)
        except Exception as e:
            logger.error(f"Failed to download {path}: {e}")

    def __build_detector(self) -> None:
        self.face_detector = cv2.FaceDetectorYN.create(
            os.path.join(MODEL_CACHE_DIR, "facedet/facedet.onnx"),
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
        )
        self.faces_per_second.start()

    def __detect_face(
        self, input: np.ndarray, threshold: float
    ) -> tuple[int, int, int, int] | None:
        """Detect faces in input image."""
        if not self.face_detector:
            return None

        # YN face detector fails at extreme definitions
        # this rescales to a size that can properly detect faces
        # still retaining plenty of detail
        if input.shape[0] > MAX_DETECTION_HEIGHT:
            scale_factor = MAX_DETECTION_HEIGHT / input.shape[0]
            new_width = int(scale_factor * input.shape[1])
            input = cv2.resize(input, (new_width, MAX_DETECTION_HEIGHT))
        else:
            scale_factor = 1

        self.face_detector.setInputSize((input.shape[1], input.shape[0]))
        faces = self.face_detector.detect(input)

        if faces is None or faces[1] is None:
            return None  # type: ignore[unreachable]

        face = None

        for _, potential_face in enumerate(faces[1]):
            if potential_face[-1] < threshold:
                continue

            raw_bbox = potential_face[0:4].astype(np.uint16)
            x: int = int(max(raw_bbox[0], 0) / scale_factor)
            y: int = int(max(raw_bbox[1], 0) / scale_factor)
            w: int = int(raw_bbox[2] / scale_factor)
            h: int = int(raw_bbox[3] / scale_factor)
            bbox = (x, y, x + w, y + h)

            if face is None or area(bbox) > area(face):  # type: ignore[unreachable]
                face = bbox

        return face

    def __update_metrics(self, duration: float) -> None:
        self.faces_per_second.update()
        self.inference_speed.update(duration)

    def __face_library_stats(self) -> tuple[int, int]:
        identities = 0
        training_images = 0
        try:
            for name in os.listdir(FACE_DIR):
                identity_dir = os.path.join(FACE_DIR, name)
                if name == "train" or not os.path.isdir(identity_dir):
                    continue
                images = sum(
                    os.path.isfile(os.path.join(identity_dir, file_name))
                    for file_name in os.listdir(identity_dir)
                )
                if images:
                    identities += 1
                    training_images += images
        except OSError as error:
            logger.warning("Unable to inspect face recognition library: %s", error)
        return identities, training_images

    def __log_pipeline_metrics(self) -> None:
        now = time.monotonic()
        if now - self.last_face_metrics_log < FACE_METRICS_LOG_INTERVAL:
            return
        identities, training_images = self.__face_library_stats()
        logger.info(
            "Face recognition pipeline frames=%d no_face=%d too_small=%d "
            "empty_crop=%d classifier_unavailable=%d classified=%d unknown=%d "
            "matched=%d vote_pending=%d snapshot_queued=%d snapshot_rejected=%d "
            "stale=%d discontinuity=%d active_tracks=%d "
            "library_identities=%d training_images=%d",
            self.face_counters["frames"],
            self.face_counters["no_face"],
            self.face_counters["too_small"],
            self.face_counters["empty_crop"],
            self.face_counters["classifier_unavailable"],
            self.face_counters["classified"],
            self.face_counters["unknown"],
            self.face_counters["matched"],
            self.face_counters["vote_pending"],
            self.face_counters["snapshot_queued"],
            self.face_counters["face_snapshot_rejected"],
            self.face_counters["stale_result"],
            self.face_counters["face_track_discontinuity"],
            len(self.face_tracks),
            identities,
            training_images,
        )
        self.last_face_metrics_log = now

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Look for faces in image."""
        self.face_counters["frames"] += 1
        self.__log_pipeline_metrics()
        self.metrics.face_rec_fps.value = self.faces_per_second.eps()
        camera = obj_data["camera"]

        if not self.config.cameras[camera].face_recognition.enabled:
            logger.debug(f"Face recognition disabled for camera {camera}, skipping")
            return

        start = datetime.datetime.now().timestamp()
        id = obj_data["id"]
        frame_time = float(obj_data["frame_time"])
        key = (camera, id)

        # don't run for non person objects
        if obj_data.get("label") != "person":
            logger.debug("Not processing face for a non person object.")
            return

        # A face result is meaningful only for a continuous person track. If
        # the tracker jumps to a different person, discard the old voting
        # history before processing the new crop.
        person_box = tuple(int(v) for v in obj_data["box"])
        track_state = self.face_tracks.get(key)
        if track_state is not None:
            if frame_time <= track_state.last_frame_time:
                self.face_counters["stale_result"] += 1
                return
            if is_track_discontinuity(
                track_state.last_box,
                person_box,
                frame_time - track_state.last_frame_time,
            ):
                self.face_counters["face_track_discontinuity"] += 1
                logger.info(
                    "Resetting face history after track discontinuity for %s/%s",
                    camera,
                    id,
                )
                track_state = FaceTrackState(
                    last_frame_time=frame_time,
                    last_box=person_box,
                    last_snapshot_time=0,
                    votes=[],
                )
                self.face_tracks[key] = track_state
                # Clear the previous face-derived identity before evaluating
                # the new track segment. The current obj_data is a local event
                # copy, so clearing it also prevents the non-face guard below
                # from preserving the old person on this frame.
                obj_data["sub_label"] = None
                self.sub_label_publisher.publish(
                    (id, None, None),
                    EventMetadataTypeEnum.sub_label.value,
                )
                self.requestor.send_data(
                    "tracked_object_update",
                    json.dumps(
                        {
                            "type": TrackedObjectUpdateTypesEnum.face,
                            "name": None,
                            "score": 0.0,
                            "id": id,
                            "camera": camera,
                            "timestamp": frame_time,
                            "source_frame_time": frame_time,
                        }
                    ),
                )
            else:
                track_state.last_frame_time = frame_time
                track_state.last_box = person_box
        else:
            track_state = FaceTrackState(
                last_frame_time=frame_time,
                last_box=person_box,
                last_snapshot_time=0,
                votes=[],
            )
            self.face_tracks[key] = track_state

        # don't overwrite sub label for objects that have a sub label
        # that is not a face
        if obj_data.get("sub_label") and not track_state.votes:
            logger.debug(
                f"Not processing face due to existing sub label: {obj_data.get('sub_label')}."
            )
            return

        # check if we have hit limits
        if len(track_state.votes) >= MAX_FACES_ATTEMPTS_AFTER_REC:
            # if we are at max attempts after rec and we have a rec
            if obj_data.get("sub_label"):
                logger.debug(
                    "Not processing due to hitting max attempts after true recognition."
                )
                return

            # if we don't have a rec and are at max attempts
            if len(track_state.votes) >= MAX_FACE_ATTEMPTS:
                logger.debug("Not processing due to hitting max rec attempts.")
                return

        face: dict[str, Any] | None = None

        if self.requires_face_detection:
            logger.debug("Running manual face detection.")
            person_box = obj_data.get("box")

            if not person_box:
                logger.debug(f"No person box available for {id}")
                return

            # YuNet (cv2.FaceDetectorYN) is trained on BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            left, top, right, bottom = person_box
            person = bgr[top:bottom, left:right]
            face_box = self.__detect_face(person, self.face_config.detection_threshold)

            if not face_box:
                self.face_counters["no_face"] += 1
                logger.debug("Detected no faces for person object.")
                return

            face_box = (
                face_box[0] + person_box[0],
                face_box[1] + person_box[1],
                face_box[2] + person_box[0],
                face_box[3] + person_box[1],
            )

            face_frame = bgr[
                max(0, face_box[1]) : min(bgr.shape[0], face_box[3]),
                max(0, face_box[0]) : min(bgr.shape[1], face_box[2]),
            ]

            # check that face is correct size
            if area(face_box) < self.config.cameras[camera].face_recognition.min_area:
                self.face_counters["too_small"] += 1
                logger.debug(
                    f"Detected face that is smaller than the min_area {face} < {self.config.cameras[camera].face_recognition.min_area}"
                )
                return

        else:
            # don't run for object without attributes
            if not obj_data.get("current_attributes"):
                logger.debug("No attributes to parse.")
                return

            attributes: list[dict[str, Any]] = obj_data.get("current_attributes", [])
            for attr in attributes:
                if attr.get("label") != "face":
                    continue

                if face is None or attr.get("score", 0.0) > face.get("score", 0.0):
                    face = attr

            # no faces detected in this frame
            if not face:
                logger.debug(f"No face attributes found for {id}")
                return

            face_box = face.get("box")

            # check that face is valid
            if (
                not face_box
                or area(face_box)
                < self.config.cameras[camera].face_recognition.min_area
            ):
                logger.debug(f"Invalid face box {face}")
                return

            face_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

            face_frame = face_frame[
                max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
            ]

        if face_frame.size == 0:
            self.face_counters["empty_crop"] += 1
            logger.debug(f"Empty face crop for {id}")
            return

        res = self.recognizer.classify(face_frame)

        if not res:
            self.face_counters["classifier_unavailable"] += 1
            logger.debug(f"Face recognizer returned no result for {id}")
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        sub_label, score = res
        self.face_counters["classified"] += 1

        if score <= self.face_config.unknown_score:
            sub_label = "unknown"
            self.face_counters["unknown"] += 1
        else:
            self.face_counters["matched"] += 1

        logger.debug(
            f"Detected best face for person as: {sub_label} with probability {score}"
        )

        self.queue_face_attempt(
            camera,
            face_frame,
            id,
            datetime.datetime.now().timestamp(),
            sub_label,
            score,
        )
        track_state.votes.append(
            FaceVote(sub_label, score, face_frame.shape[0] * face_frame.shape[1])
        )
        (weighted_sub_label, weighted_score) = self.weighted_average(track_state.votes)
        if weighted_sub_label is None:
            self.face_counters["vote_pending"] += 1

        self.requestor.send_data(
            "tracked_object_update",
            json.dumps(
                {
                    "type": TrackedObjectUpdateTypesEnum.face,
                    "name": weighted_sub_label,
                    "score": weighted_score,
                    "id": id,
                    "camera": camera,
                    "timestamp": frame_time,
                    "source_frame_time": frame_time,
                }
            ),
        )

        if weighted_score >= self.face_config.recognition_threshold:
            if (
                weighted_sub_label is not None
                and frame_time > track_state.last_snapshot_time
            ):
                queued = self.face_snapshot_worker.submit(
                    (camera, id),
                    FaceSnapshotJob(
                        camera=camera,
                        event_id=id,
                        frame_time=frame_time,
                        person_box=person_box,
                        face_box=tuple(int(value) for value in face_box),
                        sub_label=weighted_sub_label,
                        face_score=weighted_score,
                        frame=frame.copy(),
                    ),
                )
                if queued:
                    track_state.last_snapshot_time = frame_time
                    self.face_counters["snapshot_queued"] += 1
                else:
                    self.face_counters["face_snapshot_rejected"] += 1

        self.__update_metrics(datetime.datetime.now().timestamp() - start)

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == EmbeddingsRequestEnum.clear_face_classifier.value:
            self.recognizer.clear()
            return {"success": True, "message": "Face classifier cleared."}
        elif topic == EmbeddingsRequestEnum.recognize_face.value:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(request_data["image"]), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            # detect faces with lower confidence since we expect the face
            # to be visible in uploaded images
            face_box = self.__detect_face(img, 0.5)

            if not face_box:
                return {"message": "No face was detected.", "success": False}

            face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
            res = self.recognizer.classify(face)

            if not res:
                return {"success": False, "message": "No face was recognized."}

            sub_label, score = res

            if score <= self.face_config.unknown_score:
                sub_label = "unknown"

            return {"success": True, "score": score, "face_name": sub_label}
        elif topic == EmbeddingsRequestEnum.register_face.value:
            label = request_data["face_name"]

            if request_data.get("cropped"):
                thumbnail = request_data["image"]
            else:
                img = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(request_data["image"]), dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )

                # detect faces with lower confidence since we expect the face
                # to be visible in uploaded images
                face_box = self.__detect_face(img, 0.5)

                if not face_box:
                    return {
                        "message": "No face was detected.",
                        "success": False,
                    }

                face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
                _, thumbnail = cv2.imencode(
                    ".webp", face, [int(cv2.IMWRITE_WEBP_QUALITY), 100]
                )

            # write face to library
            folder = os.path.join(FACE_DIR, label)
            file = os.path.join(
                folder, f"{label}_{datetime.datetime.now().timestamp()}.webp"
            )
            os.makedirs(folder, exist_ok=True)

            # save face image
            with open(file, "wb") as output:
                output.write(thumbnail.tobytes())

            self.recognizer.clear()
            return {
                "message": "Successfully registered face.",
                "success": True,
            }
        elif topic == EmbeddingsRequestEnum.reprocess_face.value:
            current_file: str = request_data["image_file"]
            (id_time, id_rand, timestamp, _, _) = current_file.split("-")
            img = None
            id = f"{id_time}-{id_rand}"

            if current_file:
                img = cv2.imread(current_file)

            if img is None:
                return {  # type: ignore[unreachable]
                    "message": "Invalid image file.",
                    "success": False,
                }

            res = self.recognizer.classify(img)

            if not res:
                return {
                    "message": "Model is still training, please try again in a few moments.",
                    "success": False,
                }

            sub_label, score = res

            if score <= self.face_config.unknown_score:
                sub_label = "unknown"

            if "-" in sub_label:
                sub_label = sub_label.replace("-", "_")

            if self.config.face_recognition.save_attempts:
                # write face to library
                folder = os.path.join(FACE_DIR, "train")
                os.makedirs(folder, exist_ok=True)
                new_file = os.path.join(
                    folder, f"{id}-{timestamp}-{sub_label}-{score}.webp"
                )
                shutil.move(current_file, new_file)

            return {
                "message": f"Successfully reprocessed face. Result: {sub_label} (score: {score:.2f})",
                "success": True,
                "face_name": sub_label,
                "score": score,
            }

        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        self.face_tracks.pop((camera, object_id), None)

    def drain_results(self) -> list[dict[str, Any]]:
        """Return snapshot artifacts completed by the background worker."""
        payloads = []
        for result in self.face_snapshot_worker.drain_results():
            if not isinstance(result, FaceRecognitionResult):
                continue
            state = self.face_tracks.get(result.key)
            if state is not None and (
                state.candidate is None
                or result.frame_time > state.candidate.frame_time
            ):
                state.candidate = result
            payloads.append({"type": "face_snapshot", **result.as_payload()})
        return payloads

    def shutdown(self) -> None:
        """Stop snapshot encoding during embeddings shutdown."""
        self.face_snapshot_worker.stop()
        self.face_attempt_worker.stop()

    def weighted_average(
        self, results_list: list[FaceVote], max_weight: int = 4000
    ) -> tuple[str | None, float]:
        """
        Calculates a robust weighted average, capping the area weight and giving more weight to higher scores.

        Args:
            results_list: A list of tuples, where each tuple contains (name, score, face_area).
            max_weight: The maximum weight to apply based on face area.

        Returns:
            A tuple containing the prominent name and its weighted average score, or (None, 0.0) if the list is empty.
        """
        if not results_list:
            return None, 0.0

        counts: dict[str, int] = {}
        weighted_scores: dict[str, float] = {}
        total_weights: dict[str, float] = {}

        for vote in results_list:
            name = vote.sub_label
            score = vote.score
            face_area = vote.face_area
            if name == "unknown":
                continue

            if name not in weighted_scores:
                counts[name] = 0
                weighted_scores[name] = 0.0
                total_weights[name] = 0.0

            # increase count
            counts[name] += 1

            # Capped weight based on face area
            weight: float = min(face_area, max_weight)

            # Score-based weighting (higher scores get more weight)
            weight *= (score - self.face_config.unknown_score) * 10
            weighted_scores[name] += score * weight
            total_weights[name] += weight

        if not weighted_scores:
            return None, 0.0

        best_name = max(weighted_scores, key=lambda k: weighted_scores[k])

        # If the number of faces for this person < min_faces, we are not confident it is a correct result
        if counts[best_name] < self.face_config.min_faces:
            return None, 0.0

        # If the best name has the same number of results as another name, we are not confident it is a correct result
        for name, count in counts.items():
            if name != best_name and counts[best_name] == count:
                return None, 0.0

        weighted_average = weighted_scores[best_name] / total_weights[best_name]

        return best_name, weighted_average

    def queue_face_attempt(
        self,
        camera: str,
        frame: np.ndarray,
        event_id: str,
        timestamp: float,
        sub_label: str,
        score: float,
    ) -> None:
        if self.config.face_recognition.save_attempts:
            self.face_attempt_worker.submit(
                (camera, event_id),
                FaceAttemptJob(
                    frame=frame.copy(),
                    event_id=event_id,
                    timestamp=timestamp,
                    sub_label=sub_label,
                    score=score,
                    face_dir=FACE_DIR,
                    max_files=self.config.face_recognition.save_attempts,
                ),
            )
