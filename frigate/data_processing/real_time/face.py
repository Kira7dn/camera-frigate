"""Handle processing images for face detection and recognition."""

import base64
import datetime
import json
import logging
import os
import shutil
import time
from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from frigate.comms.embeddings_updater import EmbeddingsRequestEnum
from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.data_processing.common.evidence import EvidenceRingBuffer, FrameRef
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.data_processing.common.face.pipeline import (
    FaceCaptureRequest,
    FaceRecognitionOutcome,
    FaceRecognitionPipeline,
)
from frigate.data_processing.common.quality import QualitySelector, QualityThresholds
from frigate.data_processing.common.recognition import (
    BestResultReducer,
    RecognitionKey,
    RecognitionLifecycle,
    RecognitionPolicy,
    RecognitionStatus,
    face_result_outcome,
)
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.face_snapshot import (
    FACE_PROCESS_INTERVAL,
    FaceAttemptJob,
    FaceRecognitionResult,
    FaceSnapshotJob,
    FaceTrackState,
    FaceVote,
    LatestPerObjectWorker,
    cleanup_legacy_face_events,
    is_face_identity_directory,
    is_track_discontinuity,
    reap_stale_staging,
    write_face_attempt,
    write_face_snapshot_artifact,
)
from frigate.util.image import area
from frigate.util.passage_trace import (
    canonical_trace_id,
    passage_evidence,
    passage_evidence_enabled,
    passage_evidence_should_capture,
    passage_evidence_id,
    passage_trace,
)

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


MAX_DETECTION_HEIGHT = 1080
MAX_FACES_ATTEMPTS_AFTER_REC = 6
MAX_FACE_ATTEMPTS = 12
FACE_METRICS_LOG_INTERVAL = 30


def _box4(values: Any) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("face bbox must contain exactly four coordinates")
    return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))


class FaceRealTimeProcessor(RealTimeProcessorApi):
    def _get_recognition_lifecycle(self) -> RecognitionLifecycle:
        lifecycle = getattr(self, "recognition_lifecycle", None)
        if lifecycle is None:
            lifecycle = RecognitionLifecycle()
            self.recognition_lifecycle = lifecycle
        return lifecycle

    def _face_vote_decision(self, top1_score: float, top2_score: float) -> str:
        if top1_score < self.face_config.recognition_threshold:
            return "unknown"
        if top1_score - top2_score < self.face_config.min_identity_margin:
            return "ambiguous_identity"
        return "vote_valid"

    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
        evidence_ring: EvidenceRingBuffer | None = None,
        quality_selector: QualitySelector | None = None,
        recognition_lifecycle: RecognitionLifecycle | None = None,
    ):
        super().__init__(config, metrics)
        self.face_config = config.face_recognition
        self.requestor = requestor
        self.sub_label_publisher = sub_label_publisher
        self.evidence_ring = evidence_ring
        self.quality_selector = quality_selector
        self.recognition_lifecycle = recognition_lifecycle or RecognitionLifecycle()
        self.face_detector: cv2.FaceDetectorYN | None = None
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.face_tracks: dict[tuple[str, str], FaceTrackState] = {}
        self._missing_detection_counts: Counter[tuple[str, str]] = Counter()
        self.face_generation_counter = 0
        self.face_counters: Counter[str] = Counter()
        self.face_latency_samples: dict[str, deque[float]] = {
            name: deque(maxlen=512)
            for name in (
                "batch_wait_ms",
                "embedding_ms",
                "end_to_end_ms",
                "first_attempt_ms",
                "confirmed_ms",
            )
        }
        self.last_face_metrics_log = time.monotonic()
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
        self.face_pipeline = FaceRecognitionPipeline(
            self.recognizer,
            quality_selector=self.quality_selector,
            recognition_lifecycle=self.recognition_lifecycle,
        )
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

    def __log_pipeline_metrics(self) -> None:
        now = time.monotonic()
        if now - self.last_face_metrics_log < FACE_METRICS_LOG_INTERVAL:
            return
        identities, training_images = self.__face_library_stats()
        pipeline = getattr(self, "face_pipeline", None)
        batch_candidates = self.face_counters["batch_candidates"]
        latency = {
            name: self.__latency_summary(samples)
            for name, samples in self.face_latency_samples.items()
        }
        structured = {
            "candidate_age_ms": round(
                self.face_counters["candidate_age_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "candidate_drops": {
                key.removeprefix("drop_"): value
                for key, value in (pipeline.metrics.items() if pipeline else [])
                if key.startswith("drop_")
            },
            "pending_count": pipeline.pending_count() if pipeline else 0,
            "batch_size": round(
                self.face_counters["batch_size_total"] / max(1, batch_candidates),
                2,
            ),
            "batch_wait_ms": round(
                self.face_counters["batch_wait_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "batch_wait_ms_p95": latency["batch_wait_ms"][1],
            "batch_wait_ms_max": latency["batch_wait_ms"][2],
            "yunet_ms": round(
                self.face_counters["yunet_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "alignment_ms": round(
                self.face_counters["alignment_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "embedding_ms": round(
                self.face_counters["embedding_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "embedding_ms_p95": latency["embedding_ms"][1],
            "embedding_ms_max": latency["embedding_ms"][2],
            "end_to_end_ms": round(
                self.face_counters["candidate_age_ms_total"] / max(1, batch_candidates),
                1,
            ),
            "end_to_end_ms_p95": latency["end_to_end_ms"][1],
            "end_to_end_ms_max": latency["end_to_end_ms"][2],
            "first_attempt_ms": round(
                self.face_counters["first_attempt_ms_total"]
                / max(1, self.face_counters["first_attempt_count"]),
                1,
            ),
            "first_attempt_ms_p95": latency["first_attempt_ms"][1],
            "first_attempt_ms_max": latency["first_attempt_ms"][2],
            "confirmed_ms": round(
                self.face_counters["confirmed_ms_total"]
                / max(1, self.face_counters["confirmed_count"]),
                1,
            ),
            "confirmed_ms_p95": latency["confirmed_ms"][1],
            "confirmed_ms_max": latency["confirmed_ms"][2],
            "commit_state": {
                "queued": self.face_counters["snapshot_queued"],
                "rejected": self.face_counters["face_snapshot_rejected"],
            },
        }
        logger.info("face_pipeline_metrics %s", json.dumps(structured, sort_keys=True))
        logger.info(
            "Face recognition pipeline frames=%d no_face=%d too_small=%d "
            "empty_crop=%d classifier_unavailable=%d classified=%d unknown=%d "
            "matched=%d vote_pending=%d snapshot_queued=%d snapshot_rejected=%d "
            "stale_results=%d stale_frames=%d discontinuity=%d active_tracks=%d "
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
            self.face_counters["stale_frame_ingest"],
            self.face_counters["face_track_discontinuity"],
            len(self.face_tracks),
            identities,
            training_images,
        )
        for key in (
            "batch_candidates",
            "batch_size_total",
            "batch_wait_ms_total",
            "yunet_ms_total",
            "alignment_ms_total",
            "embedding_ms_total",
            "candidate_age_ms_total",
            "first_attempt_count",
            "first_attempt_ms_total",
            "confirmed_count",
            "confirmed_ms_total",
        ):
            self.face_counters[key] = 0
        for samples in self.face_latency_samples.values():
            samples.clear()
        self.last_face_metrics_log = now

    @staticmethod
    def __latency_summary(samples: deque[float]) -> tuple[float, float, float]:
        if not samples:
            return (0.0, 0.0, 0.0)
        values = np.asarray(samples, dtype=np.float64)
        return (
            round(float(values.mean()), 1),
            round(float(np.percentile(values, 95)), 1),
            round(float(values.max()), 1),
        )

    def submit_frame(self, obj_data: dict[str, Any], frame_ref: FrameRef) -> bool:
        """Conflate one eligible person track into the bounded face pipeline."""
        self.face_counters["frames"] += 1
        self.__log_pipeline_metrics()
        self.metrics.face_rec_fps.value = self.faces_per_second.eps()
        camera = str(obj_data["camera"])
        if (
            not self.config.cameras[camera].face_recognition.enabled
            or obj_data.get("label") != "person"
            or not obj_data.get("box")
        ):
            return False
        if self.evidence_ring is None or self.quality_selector is None:
            self.face_counters["evidence_unavailable"] += 1
            return False
        evidence_lease = self.evidence_ring.acquire(frame_ref)
        if evidence_lease is None:
            self.quality_selector.record_reject("face", "frame_expired")
            self.face_counters["frame_expired"] += 1
            return False

        event_id = str(obj_data["id"])
        frame_time = float(obj_data["frame_time"])
        person_box = _box4(obj_data["box"])
        key = (camera, event_id)
        state = self.face_tracks.get(key)
        if state is not None:
            if frame_time <= state.last_frame_time:
                self.face_counters["stale_frame_ingest"] += 1
                evidence_lease.release()
                return False
            if is_track_discontinuity(
                state.last_box,
                person_box,
                frame_time - state.last_frame_time,
            ):
                self.face_counters["face_track_discontinuity"] += 1
                self.face_pipeline.expire(key)
                self._get_recognition_lifecycle().expire(
                    RecognitionKey("face", camera, event_id, state.generation),
                    "track_expired",
                )
                self.quality_selector.expire("face", camera, event_id, state.generation)
                self._release_face_votes(state.votes)
                state = FaceTrackState(
                    last_frame_time=frame_time,
                    last_box=person_box,
                    last_snapshot_time=0,
                    votes=[],
                    generation=self.__next_track_generation(),
                )
                self.face_tracks[key] = state
            else:
                state.last_frame_time = frame_time
                state.last_box = person_box
        else:
            state = FaceTrackState(
                last_frame_time=frame_time,
                last_box=person_box,
                last_snapshot_time=0,
                votes=[],
                generation=self.__next_track_generation(),
            )
            self.face_tracks[key] = state

        lifecycle_key = RecognitionKey("face", camera, event_id, state.generation)
        trace_id = canonical_trace_id("face", camera, event_id, state.generation)
        if self._get_recognition_lifecycle().is_terminal(lifecycle_key):
            self.face_counters["terminal_skip"] += 1
            evidence_lease.release()
            return False

        if frame_time - state.last_attempt_time <= FACE_PROCESS_INTERVAL + 1e-6:
            self.face_counters["rate_limited"] += 1
            evidence_lease.release()
            return False
        if state.result_emitted:
            if frame_time - state.last_snapshot_time < 10:
                evidence_lease.release()
                return False
            state.result_emitted = False
            self._release_face_votes(state.votes)
            state.votes.clear()
            state.first_attempt_monotonic = 0.0
            state.first_attempt_completed = False
            state.first_match_monotonic.clear()
            self.face_counters["snapshot_retry"] += 1
        if obj_data.get("sub_label") and not state.votes:
            evidence_lease.release()
            return False
        if len(state.votes) >= MAX_FACES_ATTEMPTS_AFTER_REC and (
            obj_data.get("sub_label") or len(state.votes) >= MAX_FACE_ATTEMPTS
        ):
            evidence_lease.release()
            return False

        if passage_evidence_enabled() and passage_evidence_should_capture(
            camera, event_id, frame_time
        ):
            passage_evidence(
                "runtime_frame",
                evidence_id=passage_evidence_id(
                    camera, event_id, frame_time, frame_ref.frame_id
                ),
                camera=camera,
                frame_time=frame_time,
                track_id=event_id,
                trace_id=trace_id,
                pipeline="face",
                image=evidence_lease.frame,
                object_box=list(person_box),
                frame_ref=f"ring:{frame_ref.frame_id}",
            )

        attribute_face_box: tuple[int, int, int, int] | None = None
        if not self.requires_face_detection:
            faces = [
                attr
                for attr in obj_data.get("current_attributes", [])
                if attr.get("label") == "face" and attr.get("box")
            ]
            if not faces:
                evidence_lease.release()
                return False
            best_face = max(faces, key=lambda attr: float(attr.get("score", 0.0)))
            attribute_face_box = _box4(best_face["box"])
            if (
                area(attribute_face_box)
                < self.config.cameras[camera].face_recognition.min_area
            ):
                self.face_counters["too_small"] += 1
                evidence_lease.release()
                return False

        quality_config = self.config.cameras[camera].quality
        task_quality = quality_config.face

        request = FaceCaptureRequest(
            camera=camera,
            event_id=event_id,
            frame_time=frame_time,
            generation=state.generation,
            person_box=person_box,
            evidence_lease=evidence_lease,
            detection_threshold=self.face_config.detection_threshold,
            min_area=self.config.cameras[camera].face_recognition.min_area,
            requires_face_detection=self.requires_face_detection,
            attribute_face_box=attribute_face_box,
            vote_count=len(state.votes),
            created_monotonic=time.monotonic(),
            quality_enabled=quality_config.enabled,
            quality_thresholds=QualityThresholds(
                task_quality.min_detail_width_px,
                task_quality.min_detail_height_px,
                task_quality.min_laplacian_variance,
                task_quality.max_dark_fraction,
                task_quality.max_bright_fraction,
            ),
            top_k=quality_config.top_k,
            # Deprecated compatibility key no longer controls runtime dispatch.
            candidate_collection_seconds=0.0,
            detector_score=(
                float(best_face.get("score", 0.0))
                if attribute_face_box is not None
                else None
            ),
            quality=float(obj_data.get("area", area(person_box))),
            lifecycle_policy=RecognitionPolicy(
                max_attempts=self.config.cameras[
                    camera
                ].recognition_lifecycle.max_attempts,
                min_candidate_interval_seconds=self.config.cameras[
                    camera
                ].recognition_lifecycle.min_candidate_interval_seconds,
                max_candidate_bbox_iou=self.config.cameras[
                    camera
                ].recognition_lifecycle.max_candidate_bbox_iou,
            ),
        )
        passage_trace(
            "first_qualified_face",
            camera=camera,
            frame_time=frame_time,
            track_id=str(event_id),
            generation=state.generation,
            person_box=list(person_box),
            face_box=list(attribute_face_box) if attribute_face_box else None,
        )
        state.last_attempt_time = frame_time
        if state.first_attempt_monotonic == 0:
            state.first_attempt_monotonic = request.created_monotonic
        state.pending_frame_times.add(frame_time)
        if len(state.pending_frame_times) > 4:
            state.pending_frame_times = set(sorted(state.pending_frame_times)[-4:])
        accepted = self.face_pipeline.submit(request)
        if not accepted:
            evidence_lease.release()
        if accepted:
            self.face_counters["candidate_submitted"] += 1
            passage_trace(
                "candidate_submitted",
                camera=camera,
                frame_time=frame_time,
                track_id=str(event_id),
                generation=state.generation,
                identity=obj_data.get("sub_label") or "unknown",
                person_box=list(person_box),
            )
        return accepted

    def __next_track_generation(self) -> int:
        """Return a process-unique token, including after tracker flicker."""
        self.face_generation_counter = getattr(self, "face_generation_counter", 0) + 1
        return self.face_generation_counter

    def process_frame(
        self,
        obj_data: dict[str, Any],
        frame: np.ndarray,
        bgr_frame: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Look for faces in image."""
        if bgr_frame is None:
            bgr_frame = getattr(self, "_shared_bgr_frame", None)
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
        person_box = _box4(obj_data["box"])
        track_state = self.face_tracks.get(key)
        if track_state is not None:
            if frame_time <= track_state.last_frame_time:
                self.face_counters["stale_frame_ingest"] += 1
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
                    generation=self.__next_track_generation(),
                )
                self.face_tracks[key] = track_state
            else:
                track_state.last_frame_time = frame_time
                track_state.last_box = person_box
        else:
            track_state = FaceTrackState(
                last_frame_time=frame_time,
                last_box=person_box,
                last_snapshot_time=0,
                votes=[],
                generation=self.__next_track_generation(),
            )
            self.face_tracks[key] = track_state

        if frame_time - track_state.last_attempt_time <= FACE_PROCESS_INTERVAL + 1e-6:
            self.face_counters["rate_limited"] += 1
            return
        track_state.last_attempt_time = frame_time

        if track_state.result_emitted:
            if frame_time - track_state.last_snapshot_time < 10:
                return
            track_state.result_emitted = False
            track_state.votes.clear()
            track_state.first_attempt_monotonic = 0.0
            track_state.first_attempt_completed = False
            track_state.first_match_monotonic.clear()
            self.face_counters["snapshot_retry"] += 1

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
            bgr = (
                bgr_frame
                if bgr_frame is not None
                else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            )
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

            bgr = (
                bgr_frame
                if bgr_frame is not None
                else cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            )

            face_frame = bgr[
                max(0, face_box[1]) : min(bgr.shape[0], face_box[3]),
                max(0, face_box[0]) : min(bgr.shape[1], face_box[2]),
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
            frame_time,
            sub_label,
            score,
        )
        track_state.votes.append(
            FaceVote(sub_label, score, face_frame.shape[0] * face_frame.shape[1])
        )
        # This compatibility API is not used by the production detection-frame
        # pipeline. It must not restore the removed vote-count decision.
        weighted_sub_label, weighted_score = sub_label, score
        if weighted_sub_label is None:
            self.face_counters["vote_pending"] += 1

        if (
            weighted_score >= self.face_config.recognition_threshold
            and weighted_sub_label is not None
            and not track_state.result_emitted
        ):
            queued = self.face_snapshot_worker.submit(
                (camera, id),
                FaceSnapshotJob(
                    camera=camera,
                    event_id=id,
                    frame_time=frame_time,
                    person_box=person_box,
                    face_box=_box4(face_box),
                    sub_label=weighted_sub_label,
                    face_score=weighted_score,
                    frame=frame.copy(),
                ),
            )
            if queued:
                track_state.last_snapshot_time = frame_time
                track_state.result_emitted = True
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
            if img is None:
                return {"message": "Invalid face image.", "success": False}

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

    @staticmethod
    def _face_exhausted_reason(state: FaceTrackState) -> str:
        votes = getattr(state, "votes", [])
        if getattr(state, "ambiguous_identity_seen", False) or len(
            {vote.sub_label for vote in votes}
        ) > 1:
            return "ambiguous_identity"
        if getattr(state, "unknown_seen", False):
            return "unknown"
        return "insufficient_quality"

    def expire_object(self, object_id: str, camera: str) -> None:
        key = (camera, object_id)
        state = self.face_tracks.get(key)
        if state is not None:
            if state.closing:
                return
            state.closing = True
            state.close_deadline_monotonic = time.monotonic() + 2.0
            if hasattr(self, "face_pipeline"):
                self.face_pipeline.finalize(key)
                self._finalize_face_passage_if_ready(key, state)

    def expire_missing_objects(self, camera: str, active_ids: set[str]) -> None:
        """Finalize only after tracker-equivalent consecutive disappearance."""
        if not hasattr(self, "_missing_detection_counts"):
            self._missing_detection_counts = Counter()
        counts: Counter[tuple[str, str]] = self._missing_detection_counts
        active_ids = {str(value) for value in active_ids}
        limit = int(self.config.cameras[camera].detect.max_disappeared or 1)
        for key in list(self.face_tracks):
            if key[0] != camera:
                continue
            if key[1] in active_ids:
                counts.pop(key, None)
                continue
            counts[key] += 1
            if counts[key] >= limit:
                counts.pop(key, None)
                self.expire_object(key[1], key[0])

    def _apply_recognition_outcome(self, outcome: FaceRecognitionOutcome) -> None:
        """Apply a worker result only to the exact scheduled track generation."""
        candidate = outcome.candidate
        request = candidate.request
        state = self.face_tracks.get(request.key)
        if (
            state is None
            or state.generation != request.generation
            or request.frame_time <= state.last_result_frame_time
        ):
            self.face_counters["stale_result"] += 1
            candidate.evidence.release()
            return
        if (
            self.config.cameras[request.camera].quality.enabled
            and self.quality_selector is not None
            and not self.quality_selector.is_selected(candidate.evidence)
        ):
            self.face_counters["stale_quality_result"] += 1
            candidate.evidence.release()
            return
        state.pending_frame_times.discard(request.frame_time)
        state.last_result_frame_time = request.frame_time

        sub_label, score = outcome.sub_label, outcome.score
        margin = score - outcome.top2_score
        self.face_counters["classified"] += 1
        decision_reason = self._face_vote_decision(score, outcome.top2_score)
        vote_valid = decision_reason == "vote_valid"
        if decision_reason == "unknown":
            state.unknown_seen = True
            self.face_counters["unknown"] += 1
        elif decision_reason == "ambiguous_identity":
            state.ambiguous_identity_seen = True
            self.face_counters["ambiguous_identity"] += 1
        else:
            self.face_counters["matched"] += 1
            state.first_match_monotonic.setdefault(
                sub_label, outcome.completed_monotonic
            )
        passage_trace(
            "recognition_attempt",
            camera=request.camera,
            frame_time=request.frame_time,
            passage_id=request.event_id,
            recognition_passage_id=request.event_id,
            track_id=request.event_id,
            raw_track_lineage=[request.event_id],
            generation=request.generation,
            task="face",
            attempt_index=outcome.attempt.attempt_index,
            candidate_id=candidate.evidence.candidate_id,
            evidence_id=candidate.evidence.frame_ref.identity,
            frame_id=candidate.evidence.frame_ref.frame_id,
            bbox=list(candidate.face_box),
            quality_score=candidate.evidence.quality_score,
            top1_identity=sub_label,
            top1_score=score,
            top2_identity=outcome.top2_label,
            top2_score=outcome.top2_score,
            score_type="raw_match_score",
            identity_margin=margin,
            latency_ms=(outcome.completed_monotonic - outcome.attempt.started_monotonic)
            * 1000,
            decision_reason=decision_reason,
        )

        self.queue_face_attempt(
            request.camera,
            candidate.face_frame,
            request.event_id,
            request.frame_time,
            sub_label if vote_valid else decision_reason,
            score,
        )
        result_outcome = face_result_outcome(
            candidate_id=candidate.evidence.candidate_id,
            image_rank=candidate.evidence.image_rank,
            payload=outcome,
            top1_score=score,
            top2_score=outcome.top2_score,
            recognition_threshold=self.face_config.recognition_threshold,
            min_identity_margin=self.face_config.min_identity_margin,
            image_quality_valid=True,
            margin_scale=self.face_config.min_identity_margin,
            frame_id=candidate.evidence.frame_ref.frame_id,
            detail_bbox=candidate.face_box,
        )
        state.outcomes.append(result_outcome)
        if state.closing:
            self.face_pipeline.retry(request.key)
            self._finalize_face_passage_if_ready(request.key, state)

        self.face_counters["batch_candidates"] += 1
        self.face_counters["batch_size_total"] += outcome.batch_size
        self.face_counters["batch_wait_ms_total"] += int(outcome.batch_wait_ms)
        self.face_counters["yunet_ms_total"] += int(candidate.capture_ms)
        self.face_counters["alignment_ms_total"] += int(outcome.alignment_ms)
        self.face_counters["embedding_ms_total"] += int(outcome.embedding_ms)
        end_to_end_ms = (outcome.completed_monotonic - request.created_monotonic) * 1000
        self.face_counters["candidate_age_ms_total"] += int(end_to_end_ms)
        self.face_latency_samples["batch_wait_ms"].append(outcome.batch_wait_ms)
        self.face_latency_samples["embedding_ms"].append(outcome.embedding_ms)
        self.face_latency_samples["end_to_end_ms"].append(end_to_end_ms)
        if not state.first_attempt_completed:
            passage_trace(
                "first_attempt",
                camera=request.camera,
                frame_time=request.frame_time,
                track_id=str(request.event_id),
                generation=request.generation,
                identity=sub_label,
                score=score,
                person_box=list(request.person_box),
                face_box=list(candidate.face_box),
                candidate_id=candidate.evidence.candidate_id,
                source_role=candidate.evidence.source_role.value,
                quality_score=candidate.evidence.quality_score,
                quality_components=candidate.evidence.quality_components,
                quality_unavailable=candidate.evidence.unavailable_metrics,
                first_attempt_ms=end_to_end_ms,
                embedding_ms=outcome.embedding_ms,
            )
            state.first_attempt_completed = True
            self.face_counters["first_attempt_count"] += 1
            self.face_counters["first_attempt_ms_total"] += int(end_to_end_ms)
            self.face_latency_samples["first_attempt_ms"].append(end_to_end_ms)
        processing_seconds = (
            candidate.capture_ms + outcome.alignment_ms + outcome.embedding_ms
        ) / 1000
        self.__update_metrics(processing_seconds)

    def _finalize_face_passage_if_ready(
        self, key: tuple[str, str], state: FaceTrackState, *, force: bool = False
    ) -> bool:
        lifecycle_key = RecognitionKey("face", key[0], key[1], state.generation)
        attempts = self._get_recognition_lifecycle().attempts(lifecycle_key)
        pending = self.face_pipeline.has_pending(key)
        drained = (
            not pending
            and all(attempt.completed_monotonic is not None for attempt in attempts)
        )
        if not force and (not drained or len(state.outcomes) < len(attempts)):
            return False

        reducer = BestResultReducer("face")
        for result in state.outcomes[:3]:
            reducer.add(result)
        inference_failed = len(state.outcomes) < len(attempts) or (
            force and pending
        )
        winner = None if inference_failed else reducer.winner()
        reason = "inference_timeout" if inference_failed else reducer.exhausted_reason()
        status = RecognitionStatus.EXHAUSTED
        winner_identity = None
        winner_rank = None
        if winner is not None:
            face_outcome = winner.payload
            winning_candidate = face_outcome.candidate
            winning_request = winning_candidate.request
            queued = self.face_snapshot_worker.submit(
                key,
                FaceSnapshotJob(
                    camera=winning_request.camera,
                    event_id=winning_request.event_id,
                    frame_time=winning_request.frame_time,
                    person_box=winning_request.person_box,
                    face_box=winning_candidate.face_box,
                    sub_label=face_outcome.sub_label,
                    face_score=face_outcome.score,
                    frame=winning_candidate.evidence.frame.copy(),
                    candidate_id=winner.candidate_id,
                    quality_score=winning_candidate.evidence.quality_score,
                    quality_components=winning_candidate.evidence.quality_components,
                    source_role=winning_candidate.evidence.source_role.value,
                ),
            )
            if queued:
                status = RecognitionStatus.ACCEPTED
                reason = "best_valid_result"
                winner_identity = face_outcome.sub_label
                winner_rank = winner.result_rank
                state.result_emitted = True
                self.face_counters["snapshot_queued"] += 1
                passage_trace(
                    "confirmed_result",
                    camera=winning_request.camera,
                    frame_time=winning_request.frame_time,
                    passage_id=winning_request.event_id,
                    recognition_passage_id=winning_request.event_id,
                    track_id=winning_request.event_id,
                    generation=state.generation,
                    identity=face_outcome.sub_label,
                    score=face_outcome.score,
                    bbox=list(winning_candidate.face_box),
                    face_box=list(winning_candidate.face_box),
                    person_box=list(winning_request.person_box),
                    candidate_id=winner.candidate_id,
                    quality_score=winning_candidate.evidence.quality_score,
                    quality_components=winning_candidate.evidence.quality_components,
                    quality_unavailable=winning_candidate.evidence.unavailable_metrics,
                    source_role=winning_candidate.evidence.source_role.value,
                    image_rank=winner.image_rank,
                    result_rank=winner.result_rank,
                )
            else:
                reason = "snapshot_enqueue_failed"
                self.face_counters["face_snapshot_rejected"] += 1

        self._get_recognition_lifecycle().terminal(lifecycle_key, status, reason)
        passage_trace(
            "recognition_terminal",
            camera=key[0],
            passage_id=key[1],
            recognition_passage_id=key[1],
            track_id=key[1],
            generation=state.generation,
            task="face",
            status=status.value,
            reason=reason,
            winner=winner_identity,
            winner_rank=winner_rank,
            inferred_candidate_ids=[item.candidate_id for item in state.outcomes],
            best_effort=False,
        )
        for result in state.outcomes:
            result.payload.candidate.evidence.release()
        state.outcomes.clear()
        if self.quality_selector is not None:
            frozen = self.quality_selector.freeze(
                "face", key[0], key[1], state.generation
            )
            for candidate in frozen:
                candidate.release()
        self.face_pipeline.expire(key)
        if self.quality_selector is not None:
            self.quality_selector.expire("face", key[0], key[1], state.generation)
        self._get_recognition_lifecycle().expire(lifecycle_key, reason)
        self.face_tracks.pop(key, None)
        return True

    def drain_results(self) -> list[dict[str, Any]]:
        """Return snapshot artifacts completed by the background worker."""
        payloads = []
        if hasattr(self, "face_pipeline"):
            for outcome in self.face_pipeline.drain_results():
                self._apply_recognition_outcome(outcome)
        for key, state in list(self.face_tracks.items()):
            if state.closing and time.monotonic() >= state.close_deadline_monotonic:
                self._finalize_face_passage_if_ready(key, state, force=True)
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
        if hasattr(self, "face_pipeline"):
            self.face_pipeline.stop()
        for state in self.face_tracks.values():
            self._release_face_votes(state.votes)
            for outcome in state.outcomes:
                outcome.payload.candidate.evidence.release()
            state.outcomes.clear()
        for (camera, event_id), state in self.face_tracks.items():
            self._get_recognition_lifecycle().expire(
                RecognitionKey("face", camera, event_id, state.generation), "shutdown"
            )
        self.face_tracks.clear()
        self.face_snapshot_worker.stop()
        self.face_attempt_worker.stop()

    @staticmethod
    def _release_face_votes(votes: list[FaceVote]) -> None:
        for vote in votes:
            if vote.candidate is not None:
                vote.candidate.evidence.release()

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
