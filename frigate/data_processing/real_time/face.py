"""Handle processing images for face detection and recognition."""

from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import shutil
from collections import Counter
from typing import Any

import cv2
import numpy as np
from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.recognition.adapters.frigate import (
    BorrowedEvidenceResolver,
    FrigateEventAdapter,
    FrigateRecognitionAdapter,
)
from frigate.recognition.contracts import RecognitionTask
from frigate.recognition.core import RecognitionCore
from frigate.recognition.face import FacePolicy, weighted_face_vote
from frigate.recognition.lpr import LprPolicy
from frigate.recognition.ports import RawRecognition
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.face_snapshot import (
    FaceAttemptJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    LatestPerObjectWorker,
    cleanup_legacy_face_events,
    is_face_identity_directory,
    reap_stale_staging,
    write_face_attempt,
    write_face_snapshot_artifact,
)
from frigate.util.image import area
from frigate.util.passage_trace import (
    canonical_trace_id,
    passage_evidence,
    passage_evidence_id,
    passage_trace,
)

from frigate.config import FrigateConfig

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


MAX_DETECTION_HEIGHT = 1080
def _box4(values: Any) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("face bbox must contain exactly four coordinates")
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


class FaceRealTimeProcessor(RealTimeProcessorApi):

    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        stream_epoch: str = "process",
    ):
        super().__init__(config, metrics)
        self.face_config = config.face_recognition
        self.requestor = requestor
        self.sub_label_publisher = sub_label_publisher
        self.stream_epoch = stream_epoch
        self._recognition_adapters: dict[str, FrigateRecognitionAdapter] = {}
        self._snapshot_published: set[tuple[str, str]] = set()
        self.face_detector: cv2.FaceDetectorYN | None = None
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.face_counters: Counter[str] = Counter()
        removed = reap_stale_staging()
        if removed:
            logger.info("Removed %d stale face staging artifacts", removed)
        legacy_removed = cleanup_legacy_face_events(FACE_DIR)
        if legacy_removed:
            logger.info("Removed %d obsolete FACE_DIR/events artifacts", legacy_removed)
        self.face_snapshot_worker = LatestPerObjectWorker(
            write_face_snapshot_artifact,
            max_objects=max(
                4,
                min(
                    32,
                    4
                    * sum(
                        camera.face_recognition.enabled
                        for camera in config.cameras.values()
                    ),
                ),
            ),
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

        for adapter in self._recognition_adapters.values():
            adapter.shutdown()
        self._recognition_adapters.clear()
        self._snapshot_published.clear()

        logger.debug("Face recognition config updated and sessions reset")

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
        """Return lightweight enrollment counts for startup observability."""
        identities = 0
        training_images = 0
        try:
            for name in os.listdir(FACE_DIR):
                identity_dir = os.path.join(FACE_DIR, name)
                if not is_face_identity_directory(name, identity_dir):
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

    def _recognition_adapter(self, camera: str) -> FrigateRecognitionAdapter:
        adapter = self._recognition_adapters.get(camera)
        if adapter is not None:
            return adapter

        evidence = BorrowedEvidenceResolver()
        event_adapter = FrigateEventAdapter(
            lambda payload: self.requestor.send_data(
                "tracked_object_update", json.dumps(payload)
            ),
            lambda kind, payload: self.sub_label_publisher.publish(payload, kind),
        )
        core = RecognitionCore(
            self,
            evidence,
            LprPolicy(
                detect_fps=self.config.cameras[camera].detect.fps,
                recognition_threshold=self.config.lpr.recognition_threshold,
            ),
            FacePolicy(
                unknown_score=self.face_config.unknown_score,
                recognition_threshold=self.face_config.recognition_threshold,
                min_faces=self.face_config.min_faces,
            ),
            event_adapter,
        )
        adapter = FrigateRecognitionAdapter(core, self.stream_epoch, evidence)
        self._recognition_adapters[camera] = adapter
        return adapter

    @property
    def recognition_stats(self) -> dict[str, int]:
        totals = {"sessions": 0, "in_flight": 0, "evidence_pinned": 0}
        for adapter in self._recognition_adapters.values():
            for name in totals:
                totals[name] += adapter.stats[name]
        return totals

    def recognize(
        self,
        task: RecognitionTask,
        observation,
        evidence: object,
    ) -> RawRecognition | None:
        """Run one Face model attempt for the standalone core."""
        if task is not RecognitionTask.FACE:
            return None
        yuv_frame, supplied_bgr = evidence
        frame = np.asarray(yuv_frame)
        bgr_frame = None if supplied_bgr is None else np.asarray(supplied_bgr)
        camera = observation.key.camera_id
        person_box = observation.object_bbox
        face_box: tuple[int, int, int, int] | None = None

        if self.requires_face_detection:
            bgr = (
                bgr_frame
                if bgr_frame is not None
                else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            )
            left, top, right, bottom = person_box
            person = bgr[top:bottom, left:right]
            detected = self.__detect_face(
                person, self.face_config.detection_threshold
            )
            if detected is None:
                self.face_counters["no_face"] += 1
                return None
            face_box = (
                detected[0] + left,
                detected[1] + top,
                detected[2] + left,
                detected[3] + top,
            )
        else:
            faces = [
                attr
                for attr in observation.attributes.get("current_attributes", ())
                if attr.get("label") == "face" and attr.get("box")
            ]
            if not faces:
                self.face_counters["no_face"] += 1
                return None
            best_face = max(faces, key=lambda attr: float(attr.get("score", 0.0)))
            face_box = _box4(best_face["box"])
            bgr = (
                bgr_frame
                if bgr_frame is not None
                else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            )

        if area(face_box) < self.config.cameras[camera].face_recognition.min_area:
            self.face_counters["too_small"] += 1
            return None
        face_frame = bgr[
            max(0, face_box[1]) : min(bgr.shape[0], face_box[3]),
            max(0, face_box[0]) : min(bgr.shape[1], face_box[2]),
        ]
        if face_frame.size == 0:
            self.face_counters["empty_crop"] += 1
            return None

        started = datetime.datetime.now().timestamp()
        result = self.recognizer.classify(face_frame)
        self.__update_metrics(datetime.datetime.now().timestamp() - started)
        if result is None:
            self.face_counters["classifier_unavailable"] += 1
            return None
        name, score = result
        self.face_counters["classified"] += 1
        if score <= self.face_config.unknown_score:
            name = "unknown"
            self.face_counters["unknown"] += 1
        else:
            self.face_counters["matched"] += 1
        return RawRecognition(
            name,
            float(score),
            detail_bbox=face_box,
            area=int(face_frame.shape[0] * face_frame.shape[1]),
        )

    def process_frame(
        self,
        obj_data: dict[str, Any],
        frame: np.ndarray,
        bgr_frame: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Recognize Face through the synchronous standalone core."""
        camera = str(obj_data["camera"])
        if (
            not self.config.cameras[camera].face_recognition.enabled
            or obj_data.get("label") != "person"
            or not obj_data.get("box")
        ):
            return

        frame_time = float(obj_data["frame_time"])
        track_id = str(obj_data["id"])
        person_box = _box4(obj_data["box"])
        evidence_ref = f"face:{camera}:{track_id}:{frame_time:.6f}"
        passage_trace(
            "track_seen",
            camera=camera,
            frame_time=frame_time,
            track_id=track_id,
            task="face",
            object_box=list(person_box),
        )
        updates = self._recognition_adapter(camera).observe(
            RecognitionTask.FACE,
            obj_data,
            frame_time,
            evidence_ref,
            evidence=(frame, bgr_frame),
            observed_in_frame=obj_data.get("observed_in_frame"),
            attributes={"current_attributes": obj_data.get("current_attributes", ())},
        )
        if not updates:
            return

        bgr = (
            bgr_frame
            if bgr_frame is not None
            else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        )
        for update in updates:
            face_box = update.detail_bbox
            if face_box is None:
                continue
            evidence_id = passage_evidence_id(
                camera, track_id, frame_time, int(frame_time * 1000)
            )
            passage_evidence(
                "recognition_attempt",
                evidence_id=evidence_id,
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                trace_id=canonical_trace_id("face", camera, track_id),
                pipeline="face",
                image=bgr,
                object_box=list(person_box),
                detail_box=list(face_box),
                raw_identity=update.raw_value,
                raw_score=update.raw_score,
            )
            face_crop = bgr[
                max(0, face_box[1]) : min(bgr.shape[0], face_box[3]),
                max(0, face_box[0]) : min(bgr.shape[1], face_box[2]),
            ]
            if face_crop.size:
                self.queue_face_attempt(
                    camera,
                    face_crop,
                    track_id,
                    frame_time,
                    update.raw_value or "unknown",
                    update.raw_score,
                )
            passage_trace(
                "first_qualified_face",
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                person_box=list(person_box),
                face_box=list(face_box),
            )
            passage_trace(
                "candidate_submitted",
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                person_box=list(person_box),
                face_box=list(face_box),
            )
            passage_trace(
                "first_attempt",
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                identity=update.raw_value,
                score=update.raw_score,
                person_box=list(person_box),
                face_box=list(face_box),
            )
            if not update.publish:
                continue
            passage_trace(
                "confirmed_result",
                camera=camera,
                frame_time=frame_time,
                track_id=track_id,
                identity=update.aggregate_value,
                score=update.aggregate_score,
                person_box=list(person_box),
                face_box=list(face_box),
            )
            snapshot_key = (camera, track_id)
            if snapshot_key in self._snapshot_published:
                continue
            if self.face_snapshot_worker.submit(
                snapshot_key,
                FaceSnapshotJob(
                    camera=camera,
                    event_id=track_id,
                    frame_time=frame_time,
                    person_box=person_box,
                    face_box=face_box,
                    sub_label=update.aggregate_value or "unknown",
                    face_score=update.aggregate_score,
                    frame=frame.copy(),
                ),
            ):
                self._snapshot_published.add(snapshot_key)

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == EmbeddingsRequestEnum.clear_face_classifier.value:
            self.recognizer.clear()
            return {"success": True, "message": "Face classifier cleared."}
        if topic == EmbeddingsRequestEnum.recognize_face.value:
            img = cv2.imdecode(
                np.frombuffer(base64.b64decode(request_data["image"]), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is None:
                return {"message": "Invalid face image.", "success": False}
            face_box = self.__detect_face(img, 0.5)
            if not face_box:
                return {"message": "No face was detected.", "success": False}
            result = self.recognizer.classify(
                img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
            )
            if result is None:
                return {"success": False, "message": "No face was recognized."}
            name, score = result
            if score <= self.face_config.unknown_score:
                name = "unknown"
            return {"success": True, "score": score, "face_name": name}
        if topic == EmbeddingsRequestEnum.register_face.value:
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
                if img is None:
                    return {"message": "Invalid face image.", "success": False}
                face_box = self.__detect_face(img, 0.5)
                if not face_box:
                    return {"message": "No face was detected.", "success": False}
                face = img[face_box[1] : face_box[3], face_box[0] : face_box[2]]
                _, thumbnail = cv2.imencode(
                    ".webp", face, [int(cv2.IMWRITE_WEBP_QUALITY), 100]
                )
            folder = os.path.join(FACE_DIR, label)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                folder, f"{label}_{datetime.datetime.now().timestamp()}.webp"
            )
            with open(path, "wb") as output:
                output.write(thumbnail.tobytes())
            self.recognizer.clear()
            return {"message": "Successfully registered face.", "success": True}
        if topic == EmbeddingsRequestEnum.reprocess_face.value:
            current_file = str(request_data["image_file"])
            parts = current_file.split("-")
            if len(parts) < 5:
                return {"message": "Invalid image file.", "success": False}
            id_time, id_rand, timestamp, _, _ = parts
            img = cv2.imread(current_file)
            if img is None:
                return {"message": "Invalid image file.", "success": False}
            result = self.recognizer.classify(img)
            if result is None:
                return {
                    "message": "Model is still training, please try again later.",
                    "success": False,
                }
            name, score = result
            if score <= self.face_config.unknown_score:
                name = "unknown"
            name = name.replace("-", "_")
            if self.config.face_recognition.save_attempts:
                folder = os.path.join(FACE_DIR, "train")
                os.makedirs(folder, exist_ok=True)
                shutil.move(
                    current_file,
                    os.path.join(
                        folder,
                        f"{id_time}-{id_rand}-{timestamp}-{name}-{score}.webp",
                    ),
                )
            return {
                "message": f"Successfully reprocessed face. Result: {name} (score: {score:.2f})",
                "success": True,
                "face_name": name,
                "score": score,
            }
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        adapter = self._recognition_adapters.get(camera)
        if adapter is not None:
            adapter.end_track(camera, object_id, "event_end")
        self._snapshot_published.discard((camera, object_id))

    def weighted_average(
        self, results_list: list[tuple[str, float, int]], max_weight: int = 4000
    ) -> tuple[str | None, float]:
        return weighted_face_vote(
            [
                RawRecognition(name, score, area=face_area)
                for name, score, face_area in results_list
            ],
            FacePolicy(
                unknown_score=self.face_config.unknown_score,
                recognition_threshold=self.face_config.recognition_threshold,
                min_faces=self.face_config.min_faces,
                area_cap=max_weight,
            ),
        )

    def drain_results(self) -> list[dict[str, Any]]:
        payloads = []
        for result in self.face_snapshot_worker.drain_results():
            if isinstance(result, FaceRecognitionResult):
                payloads.append({"type": "face_snapshot", **result.as_payload()})
        return payloads

    def shutdown(self) -> None:
        for adapter in self._recognition_adapters.values():
            adapter.shutdown()
        self._recognition_adapters.clear()
        self._snapshot_published.clear()
        self.face_snapshot_worker.stop()
        self.face_attempt_worker.stop()

    def queue_face_attempt(
        self,
        camera: str,
        frame: np.ndarray,
        event_id: str,
        timestamp: float,
        sub_label: str,
        score: float,
    ) -> None:
        if self.config.face_recognition.save_attempts and sub_label == "unknown":
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
