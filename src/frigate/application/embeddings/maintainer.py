"""Maintain embeddings in SQLite-vec."""

import base64
import json
import logging
import os
import queue
import threading
import uuid
from multiprocessing import Queue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, cast

from peewee import DoesNotExist

from frigate.infrastructure.comms.config_updater import ConfigSubscriber
from frigate.infrastructure.comms.detections_updater import DetectionSubscriber, DetectionTypeEnum
from frigate.infrastructure.comms.embeddings_updater import (
    EmbeddingsRequestEnum,
    EmbeddingsResponder,
)
from frigate.infrastructure.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataSubscriber,
    EventMetadataTypeEnum,
)
from frigate.infrastructure.comms.events_updater import EventEndSubscriber, EventUpdateSubscriber
from frigate.infrastructure.comms.inter_process import InterProcessRequestor
from frigate.infrastructure.comms.recordings_updater import (
    RecordingsDataSubscriber,
    RecordingsDataTypeEnum,
)
from frigate.infrastructure.comms.review_updater import ReviewDataSubscriber
from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)
from frigate.infrastructure.config.classification import ObjectClassificationType
from frigate.infrastructure.data_processing.common.license_plate.model import (
    LicensePlateModelRunner,
)
from frigate.infrastructure.data_processing.post.api import PostProcessorApi
from frigate.infrastructure.data_processing.post.audio_transcription import (
    AudioTranscriptionPostProcessor,
)
from frigate.infrastructure.data_processing.post.object_descriptions import ObjectDescriptionProcessor
from frigate.infrastructure.data_processing.post.review_descriptions import ReviewDescriptionProcessor
from frigate.infrastructure.data_processing.post.semantic_trigger import SemanticTriggerProcessor
from frigate.infrastructure.data_processing.real_time.api import RealTimeProcessorApi
from frigate.infrastructure.data_processing.real_time.external_recognition import (
    ExternalRecognitionProcessor,
)
from frigate.infrastructure.data_processing.real_time.face import FaceRealTimeProcessor
from frigate.infrastructure.data_processing.real_time.license_plate import (
    LicensePlateRealTimeProcessor,
)
from frigate.infrastructure.data_processing.types import DataProcessorMetrics, PostProcessDataEnum
from frigate.infrastructure.db.sqlitevecq import SqliteVecQueueDatabase
from frigate.application.events.types import (
    EventStateEnum,
    EventTypeEnum,
    RegenerateDescriptionEnum,
)
from frigate.application.genai import GenAIClientManager
from frigate.models import Event, Recordings, ReviewSegment, Trigger
from camera_platform.topology.compiler import compile_topology
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import serialize
from frigate.util.file import get_event_thumbnail_bytes
from frigate.util.image import SharedMemoryFrameManager
from frigate.util.passage_trace import passage_writer_stats, shutdown_passage_writers

from .embeddings import Embeddings

logger = logging.getLogger(__name__)

MAX_THUMBNAILS = 10


class EmbeddingMaintainer(threading.Thread):
    """Handle embedding queue and post event updates."""

    def __init__(
        self,
        config: FrigateConfig,
        metrics: DataProcessorMetrics,
        stop_event: MpEvent,
        face_result_queue: Queue,
    ) -> None:
        super().__init__(name="embeddings_maintainer")
        self.config = config
        self.metrics = metrics
        self.face_result_queue = face_result_queue
        self.embeddings: Embeddings | None = None
        self.config_updater = CameraConfigUpdateSubscriber(
            self.config,
            self.config.cameras,
            [
                CameraConfigUpdateEnum.add,
                CameraConfigUpdateEnum.remove,
                CameraConfigUpdateEnum.detect,
                CameraConfigUpdateEnum.face_recognition,
                CameraConfigUpdateEnum.ffmpeg,
                CameraConfigUpdateEnum.lpr,
                CameraConfigUpdateEnum.motion,
                CameraConfigUpdateEnum.objects,
                CameraConfigUpdateEnum.object_genai,
                CameraConfigUpdateEnum.review,
                CameraConfigUpdateEnum.review_genai,
                CameraConfigUpdateEnum.semantic_search,
                CameraConfigUpdateEnum.zones,
            ],
        )
        self.enrichment_config_subscriber = ConfigSubscriber("config/")

        # Configure Frigate DB
        db = SqliteVecQueueDatabase(
            config.database.path,
            pragmas={
                "auto_vacuum": "FULL",  # Does not defragment database
                "cache_size": -512 * 1000,  # 512MB of cache
                "synchronous": "NORMAL",  # Safe when using WAL https://www.sqlite.org/pragma.html#pragma_synchronous
            },
            timeout=max(
                60, 10 * len([c for c in config.cameras.values() if c.enabled])
            ),
            load_vec_extension=True,
        )
        models = [Event, Recordings, ReviewSegment, Trigger]
        db.bind(models)

        self.genai_manager = GenAIClientManager(config)

        needs_embeddings = (
            config.semantic_search.enabled
            or any(
                camera.audio_transcription.enabled
                or camera.objects.genai.enabled_in_config
                for camera in config.cameras.values()
            )
        )
        if needs_embeddings:
            self.embeddings = Embeddings(config, db, metrics, self.genai_manager)

            # Check if we need to re-index events
            if config.semantic_search.enabled and config.semantic_search.reindex:
                self.embeddings.reindex()

            # Sync semantic search triggers in db with config
            if config.semantic_search.enabled:
                self.embeddings.sync_triggers()

        # create communication for updating event descriptions
        self.requestor = InterProcessRequestor()

        self.event_subscriber = EventUpdateSubscriber()
        self.event_end_subscriber = EventEndSubscriber()
        self.event_metadata_publisher = EventMetadataPublisher()
        self.event_metadata_subscriber = EventMetadataSubscriber(
            EventMetadataTypeEnum.regenerate_description
        )
        self.recordings_subscriber = RecordingsDataSubscriber(
            RecordingsDataTypeEnum.saved
        )
        self.review_subscriber = ReviewDataSubscriber("")
        self.detection_subscriber = DetectionSubscriber(DetectionTypeEnum.video.value)
        self._latest_detection_lock = threading.Lock()
        self._latest_detections: dict[str, Any] = {}
        self._detection_frames_overwritten = 0
        self._camera_cursor = 0
        self._detection_reader = threading.Thread(
            target=self._read_detection_updates,
            daemon=True,
            name="embeddings_detection_ingest",
        )
        self.embeddings_responder = EmbeddingsResponder()
        self.frame_manager = SharedMemoryFrameManager()
        self._recognition_stream_epoch = uuid.uuid4().hex
        self._custom_processor_types: tuple[type[Any], type[Any]] | None = None

        external_recognition = compile_topology(self.config).recognition_external

        # model runners to share between realtime and post processors
        if self.config.lpr.enabled and not external_recognition:
            lpr_model_runner = LicensePlateModelRunner(
                self.requestor,
                device=self.config.lpr.device or "CPU",
                model_size=self.config.lpr.model_size,
            )

        # realtime processors
        self.realtime_processors: list[RealTimeProcessorApi] = []

        if external_recognition and (
            self.config.face_recognition.enabled or self.config.lpr.enabled
        ):
            logger.info(
                "External recognition enabled; local Face and LPR models are disabled"
            )
            self.realtime_processors.append(
                ExternalRecognitionProcessor(
                    self.config,
                    self.requestor,
                    self.event_metadata_publisher,
                    metrics,
                    self._recognition_stream_epoch,
                )
            )
        elif self.config.face_recognition.enabled:
            logger.debug("Face recognition enabled, initializing FaceRealTimeProcessor")
            self.realtime_processors.append(
                FaceRealTimeProcessor(
                    self.config,
                    self.requestor,
                    self.event_metadata_publisher,
                    metrics,
                    self._recognition_stream_epoch,
                )
            )
            logger.debug("FaceRealTimeProcessor initialized successfully")

        if self.config.classification.bird.enabled:
            from frigate.infrastructure.data_processing.real_time.bird import BirdRealTimeProcessor

            self.realtime_processors.append(
                BirdRealTimeProcessor(
                    self.config, self.event_metadata_publisher, metrics
                )
            )

        if self.config.lpr.enabled and not external_recognition:
            self.realtime_processors.append(
                LicensePlateRealTimeProcessor(
                    self.config,
                    self.requestor,
                    self.event_metadata_publisher,
                    metrics,
                    lpr_model_runner,
                    self._recognition_stream_epoch,
                )
            )

        for model_config in self.config.classification.custom.values():
            if not model_config.enabled:
                continue

            state_processor_type, object_processor_type = (
                self._get_custom_processor_types()
            )

            self.realtime_processors.append(
                state_processor_type(
                    self.config, model_config, self.requestor, self.metrics
                )
                if model_config.state_config != None
                else object_processor_type(
                    self.config,
                    model_config,
                    self.event_metadata_publisher,
                    self.requestor,
                    self.metrics,
                )
            )

        # post processors
        self.post_processors: list[PostProcessorApi] = []

        if any(c.review.genai.enabled_in_config for c in self.config.cameras.values()):
            self.post_processors.append(
                ReviewDescriptionProcessor(
                    self.config,
                    self.requestor,
                    self.metrics,
                    self.genai_manager,
                )
            )

        if any(
            c.enabled_in_config and c.audio_transcription.enabled
            for c in self.config.cameras.values()
        ):
            if self.embeddings is None:
                raise RuntimeError("Audio transcription requires embeddings")
            self.post_processors.append(
                AudioTranscriptionPostProcessor(
                    self.config, self.requestor, self.embeddings, metrics
                )
            )

        semantic_trigger_processor: SemanticTriggerProcessor | None = None
        if self.config.semantic_search.enabled:
            if self.embeddings is None:
                raise RuntimeError("Semantic search requires embeddings")
            semantic_trigger_processor = SemanticTriggerProcessor(
                db,
                self.config,
                self.requestor,
                self.event_metadata_publisher,
                metrics,
                self.embeddings,
            )
            self.post_processors.append(semantic_trigger_processor)

        if any(c.objects.genai.enabled_in_config for c in self.config.cameras.values()):
            if self.embeddings is None:
                raise RuntimeError("Object descriptions require embeddings")
            self.post_processors.append(
                ObjectDescriptionProcessor(
                    self.config,
                    self.embeddings,
                    self.requestor,
                    self.metrics,
                    self.genai_manager,
                    semantic_trigger_processor,
                )
            )

        # Recordings availability is process-owned mutable state and must exist
        # before the maintainer thread starts consuming updates.
        self.recordings_available_through: dict[str, float] = {}
        self.stop_event = stop_event

    def _get_custom_processor_types(self) -> tuple[type[Any], type[Any]]:
        if self._custom_processor_types is None:
            from frigate.infrastructure.data_processing.real_time.custom_classification import (
                CustomObjectClassificationProcessor,
                CustomStateClassificationProcessor,
            )

            self._custom_processor_types = (
                CustomStateClassificationProcessor,
                CustomObjectClassificationProcessor,
            )
        return self._custom_processor_types

    def run(self) -> None:
        """Maintain a SQLite-vec database for semantic search."""
        try:
            self._run_loop()
        except BaseException:
            logger.exception("Embeddings maintainer failed")
            raise

    def _run_loop(self) -> None:
        """Run the maintainer loop and surface failures to the process owner."""
        self._detection_reader.start()
        while not self.stop_event.is_set():
            self.config_updater.check_for_updates()
            self._check_enrichment_config_updates()
            self._process_requests()
            self._process_updates()
            self._process_recordings_updates()
            self._process_review_updates()
            self._process_frame_updates()
            self._process_deferred_results()
            self._sync_recognition_metrics()
            self._process_finalized()
            self._process_event_metadata()

        # Shutdown deferred processors
        for processor in self.realtime_processors:
            processor.shutdown()
        if not shutdown_passage_writers(2.0):
            logger.warning("Recognition trace writer did not flush before deadline")

        self.config_updater.stop()
        self.enrichment_config_subscriber.stop()
        self.event_subscriber.stop()
        self.event_end_subscriber.stop()
        self.recordings_subscriber.stop()
        self._detection_reader.join(timeout=2)
        self.detection_subscriber.stop()
        self.event_metadata_publisher.stop()
        self.event_metadata_subscriber.stop()
        self.embeddings_responder.stop()
        self.requestor.stop()
        logger.info("Exiting embeddings maintenance...")

    def _sync_recognition_metrics(self) -> None:
        recognition = {
            "sessions": 0,
            "in_flight": 0,
            "evidence_pinned": 0,
            "queue_depth": 0,
            "outcome_depth": 0,
            "rejected": 0,
            "service_healthy": 0,
        }
        for processor in self.realtime_processors:
            stats = getattr(processor, "recognition_stats", None)
            if stats is None:
                continue
            for name in recognition:
                recognition[name] += int(stats.get(name, 0))
        writer = passage_writer_stats()
        self.metrics.recognition_sessions.value = float(recognition["sessions"])
        self.metrics.recognition_in_flight.value = float(recognition["in_flight"])
        self.metrics.recognition_evidence_pinned.value = float(
            recognition["evidence_pinned"]
        )
        self.metrics.recognition_queue_depth.value = float(recognition["queue_depth"])
        self.metrics.recognition_outcome_depth.value = float(
            recognition["outcome_depth"]
        )
        self.metrics.recognition_rejected.value = float(recognition["rejected"])
        self.metrics.recognition_service_healthy.value = float(
            recognition["service_healthy"]
        )
        self.metrics.recognition_writer_depth.value = float(writer["depth"])
        self.metrics.recognition_writer_drops.value = float(writer["drops"])
        self.metrics.recognition_writer_errors.value = float(writer["errors"])

    def _check_enrichment_config_updates(self) -> None:
        """Check for enrichment config updates and delegate to processors."""
        topic, payload = self.enrichment_config_subscriber.check_for_update()

        if topic is None:
            return

        # Custom classification add/remove requires managing the processor list
        if topic.startswith("config/classification/custom/"):
            self._handle_custom_classification_update(topic, payload)
            return

        if topic == "config/genai":
            if not isinstance(payload, dict):
                logger.warning("Ignoring invalid GenAI configuration update")
                return
            self.config.genai = cast(Any, payload)
            self.genai_manager.update_config(self.config)

        # Broadcast to all processors — each decides if the topic is relevant
        for processor in self.realtime_processors:
            processor.update_config(topic, payload)

        for processor in self.post_processors:
            processor.update_config(topic, payload)

    def _remove_custom_classification_processor(self, model_name: str) -> None:
        """Shut down and drop any running processor for a custom model."""
        custom_processor_types = self._get_custom_processor_types()
        remaining = []
        for processor in self.realtime_processors:
            if (
                isinstance(processor, custom_processor_types)
                and processor.model_config.name == model_name
            ):
                processor.shutdown()
            else:
                remaining.append(processor)
        self.realtime_processors = remaining

    def _handle_custom_classification_update(
        self, topic: str, model_config: Any
    ) -> None:
        """Handle add/remove of custom classification processors."""
        model_name = topic.split("/")[-1]

        if model_config is None:
            self._remove_custom_classification_processor(model_name)
            logger.info(
                f"Successfully removed classification processor for model: {model_name}"
            )
            return

        self.config.classification.custom[model_name] = model_config
        state_processor_type, object_processor_type = self._get_custom_processor_types()

        # A disabled model must not run; tear down any existing processor and
        # do not register a new one.
        if not model_config.enabled:
            self._remove_custom_classification_processor(model_name)
            logger.info(f"Disabled classification processor for model: {model_name}")
            return

        for processor in self.realtime_processors:
            if (
                isinstance(processor, state_processor_type | object_processor_type)
                and processor.model_config.name == model_name
            ):
                processor.model_config = model_config
                logger.debug(
                    f"Updated config for classification processor: {model_name}"
                )
                return

        if model_config.state_config is not None:
            processor = state_processor_type(
                self.config, model_config, self.requestor, self.metrics
            )
        else:
            processor = object_processor_type(
                self.config,
                model_config,
                self.event_metadata_publisher,
                self.requestor,
                self.metrics,
            )

        self.realtime_processors.append(processor)
        logger.info(
            f"Added classification processor for model: {model_name} (type: {type(processor).__name__})"
        )

    def _process_requests(self) -> None:
        """Process embeddings requests"""

        def _handle_request(topic: str, data: dict[str, Any]) -> Any:
            try:
                # First handle the embedding-specific topics when semantic search is enabled
                if self.config.semantic_search.enabled:
                    if self.embeddings is None:
                        raise RuntimeError("Semantic search embeddings are unavailable")
                    if topic == EmbeddingsRequestEnum.embed_description.value:
                        return serialize(
                            self.embeddings.embed_description(
                                data["id"], data["description"]
                            ),
                            pack=False,
                        )
                    elif topic == EmbeddingsRequestEnum.embed_thumbnail.value:
                        thumbnail = base64.b64decode(data["thumbnail"])
                        return serialize(
                            self.embeddings.embed_thumbnail(data["id"], thumbnail),
                            pack=False,
                        )
                    elif topic == EmbeddingsRequestEnum.generate_search.value:
                        return serialize(
                            self.embeddings.embed_description(
                                "", str(data.get("description", "")), upsert=False
                            ),
                            pack=False,
                        )
                    elif topic == EmbeddingsRequestEnum.reindex.value:
                        response = self.embeddings.start_reindex()
                        return "started" if response else "in_progress"

                processors = [self.realtime_processors, self.post_processors]
                for processor_list in processors:
                    for processor in processor_list:
                        resp = processor.handle_request(topic, data)
                        if resp is not None:
                            return resp

                logger.error(f"No processor handled the topic {topic}")
                return None
            except Exception as e:
                logger.exception(f"Unable to handle embeddings request {e}")
                return None

        self.embeddings_responder.check_for_request(_handle_request)

    def _process_updates(self) -> None:
        """Process event updates"""
        update = self.event_subscriber.check_for_update()

        if update is None:
            return

        source_type, event_type, camera, frame_name, data = update
        owned_recognition_evidence = bool(
            data.pop("_recognition_evidence_owned", False)
        )

        logger.debug(
            f"Received update - source_type: {source_type}, camera: {camera}, data label: {data.get('label') if data else 'None'}"
        )

        if not camera or source_type != EventTypeEnum.tracked_object:
            logger.debug(
                f"Skipping update - camera: {camera}, source_type: {source_type}"
            )
            if owned_recognition_evidence:
                self.frame_manager.delete(frame_name)
            return

        if self.config.semantic_search.enabled and self.embeddings is not None:
            self.embeddings.update_stats()

        camera_config = self.config.cameras.get(camera)
        if camera_config is None:
            if owned_recognition_evidence:
                self.frame_manager.delete(frame_name)
            return

        # no need to process updated objects if no processors are active
        if len(self.realtime_processors) == 0 and len(self.post_processors) == 0:
            logger.debug(
                f"No processors active - realtime: {len(self.realtime_processors)}, post: {len(self.post_processors)}"
            )
            if owned_recognition_evidence:
                self.frame_manager.delete(frame_name)
            return

        if event_type == EventStateEnum.end:
            object_id = str(data.get("id", ""))
            if object_id:
                for processor in self.realtime_processors:
                    if isinstance(
                        processor,
                        FaceRealTimeProcessor
                        | LicensePlateRealTimeProcessor
                        | ExternalRecognitionProcessor,
                    ):
                        # The canonical tracked-object end owns recognition
                        # cleanup. The finalized-event subscriber repeats this
                        # as an idempotent safety net for DB/media processing.
                        processor.expire_object(object_id, camera)

        # Create our own thumbnail based on the bounding box and the frame time
        yuv_frame = None
        try:
            yuv_frame = self.frame_manager.get(
                frame_name, camera_config.frame_shape_yuv
            )
        except FileNotFoundError:
            logger.debug(f"Frame {frame_name} not found for camera {camera}")

        if yuv_frame is None:
            logger.debug(
                "Unable to process object update because frame is unavailable."
            )
            if owned_recognition_evidence:
                self.frame_manager.delete(frame_name)
            return

        try:
            logger.debug(
                f"Processing {len(self.realtime_processors)} realtime processors for object {data.get('id')} (label: {data.get('label')})"
            )
            for processor in self.realtime_processors:
                if event_type == EventStateEnum.end and isinstance(
                    processor,
                    FaceRealTimeProcessor
                    | LicensePlateRealTimeProcessor
                    | ExternalRecognitionProcessor,
                ):
                    # End callbacks carry the current frame but the removed
                    # object's prior bbox. Cleanup already ran before frame lookup.
                    continue
                logger.debug(
                    f"Calling process_frame on {processor.__class__.__name__}"
                )
                processor.process_frame(data, yuv_frame)

            for processor in self.post_processors:
                if isinstance(processor, ObjectDescriptionProcessor):
                    # skip end events — _process_finalized handles them via event_end_subscriber.
                    # processing them here can re-create tracked_events entries after cleanup
                    # when the event_subscriber queue is backlogged behind event_end_subscriber.
                    if event_type == EventStateEnum.end:
                        continue

                    processor.process_data(
                        {
                            "camera": camera,
                            "data": data,
                            "state": "update",
                            "yuv_frame": yuv_frame,
                        },
                        PostProcessDataEnum.tracked_object,
                    )
        finally:
            if owned_recognition_evidence:
                self.frame_manager.delete(frame_name)
            else:
                self.frame_manager.close(frame_name)

    def _process_finalized(self) -> None:
        """Process the end of an event."""
        while True:
            ended = self.event_end_subscriber.check_for_update()

            if ended == None:
                break

            event_id, camera, updated_db = ended

            # expire in realtime processors
            for processor in self.realtime_processors:
                processor.expire_object(event_id, camera)

            thumbnail: bytes | None = None

            if updated_db:
                try:
                    event: Event = Event.get(Event.id == event_id)
                except DoesNotExist:
                    for processor in self.post_processors:
                        if isinstance(processor, ObjectDescriptionProcessor):
                            processor.cleanup_event(event_id)
                    continue

                # Skip the event if not an object
                event_data = event.data
                if not isinstance(event_data, dict) or event_data.get("type") != "object":
                    for processor in self.post_processors:
                        if isinstance(processor, ObjectDescriptionProcessor):
                            processor.cleanup_event(event_id)
                    continue

                # Extract valid thumbnail
                thumbnail = get_event_thumbnail_bytes(event)

                # Embed the thumbnail
                if thumbnail is not None:
                    self._embed_thumbnail(event_id, thumbnail)

            # call any defined post processors
            for processor in self.post_processors:
                if isinstance(processor, AudioTranscriptionPostProcessor):
                    continue
                elif isinstance(processor, SemanticTriggerProcessor):
                    processor.process_data(
                        {"event_id": event_id, "camera": camera, "type": "image"},
                        PostProcessDataEnum.tracked_object,
                    )
                elif isinstance(processor, ObjectDescriptionProcessor):
                    if not updated_db:
                        # Still need to cleanup tracked events even if not processing
                        processor.cleanup_event(event_id)
                        continue

                    processor.process_data(
                        {
                            "event": event,
                            "camera": camera,
                            "state": "finalize",
                            "thumbnail": thumbnail,
                        },
                        PostProcessDataEnum.tracked_object,
                    )
                else:
                    processor.process_data(
                        {"event_id": event_id, "camera": camera},
                        PostProcessDataEnum.tracked_object,
                    )

    def _process_recordings_updates(self) -> None:
        """Process recordings updates."""
        while True:
            update = self.recordings_subscriber.check_for_update()

            if not update:
                break

            (raw_topic, payload) = update

            if not raw_topic or not payload:
                break

            topic = str(raw_topic)

            if topic.endswith(RecordingsDataTypeEnum.saved.value):
                camera, recordings_available_through_timestamp, _ = payload

                self.recordings_available_through[camera] = (
                    recordings_available_through_timestamp
                )

                logger.debug(
                    f"{camera} now has recordings available through {recordings_available_through_timestamp}"
                )

    def _process_review_updates(self) -> None:
        """Process review updates."""
        while True:
            review_updates = self.review_subscriber.check_for_update()

            if review_updates == None:
                break

            for processor in self.post_processors:
                if isinstance(processor, ReviewDescriptionProcessor):
                    processor.process_data(review_updates, PostProcessDataEnum.review)

    def _process_event_metadata(self):
        # Check for regenerate description requests
        update = self.event_metadata_subscriber.check_for_update()
        if update is None:
            return
        topic, payload = update
        if topic is None or not isinstance(payload, tuple | list) or len(payload) != 3:
            return

        event_id, source, force = payload

        if event_id:
            for processor in self.post_processors:
                if isinstance(processor, ObjectDescriptionProcessor):
                    processor.handle_request(
                        "regenerate_description",
                        {
                            "event_id": event_id,
                            "source": RegenerateDescriptionEnum(source),
                            "force": force,
                        },
                    )

    def _read_detection_updates(self) -> None:
        """Continuously conflate detection updates into one slot per camera."""
        while not self.stop_event.is_set():
            update = self.detection_subscriber.check_for_update(timeout=0.1)
            if update is None:
                continue
            _, data = update
            if not data or not data[0]:
                continue
            camera = str(data[0])
            with self._latest_detection_lock:
                if camera in self._latest_detections:
                    self._detection_frames_overwritten += 1
                self._latest_detections[camera] = data

    def _process_frame_updates(self) -> None:
        """Process conflated camera frames in rotating camera order."""
        if not hasattr(self, "_latest_detection_lock"):
            latest_by_camera: dict[str, Any] = {}
            topic, data = self.detection_subscriber.check_for_update(timeout=0)
            for _ in range(256):
                if topic is None:
                    break
                if data and data[0]:
                    latest_by_camera[str(data[0])] = data
                topic, data = self.detection_subscriber.check_for_update(timeout=0)
            for data in latest_by_camera.values():
                self._process_latest_frame(data)
            return
        with self._latest_detection_lock:
            latest_by_camera = self._latest_detections
            self._latest_detections = {}

        cameras = sorted(latest_by_camera)
        if not cameras:
            return
        start = self._camera_cursor % len(cameras)
        ordered = cameras[start:] + cameras[:start]
        self._camera_cursor = (start + 1) % len(cameras)
        for camera in ordered:
            data = latest_by_camera[camera]
            self._process_latest_frame(data)

    def _process_latest_frame(self, data: Any) -> None:
        """Process one latest detection frame without retaining older work."""

        camera, frame_name, frame_time, tracked_objects, motion_boxes, _ = data

        if not camera or camera not in self.config.cameras:
            return

        camera_config = self.config.cameras.get(camera)
        if camera_config is None:
            return

        has_enabled_custom = any(
            c.enabled for c in self.config.classification.custom.values()
        )
        # Face recognition is driven only by the canonical tracked-object
        # callback above. Detection-frame decision processing is superseded
        # and must not run in parallel.
        if not has_enabled_custom:
            # no active features that use this data
            return

        yuv_frame = None
        try:
            yuv_frame = self.frame_manager.get(
                frame_name, camera_config.frame_shape_yuv
            )
        except FileNotFoundError:
            pass

        if yuv_frame is None:
            logger.debug(
                "Unable to process dedicated LPR update because frame is unavailable."
            )
            return

        for processor in self.realtime_processors:
            custom_processor_types = getattr(self, "_custom_processor_types", None)
            if (
                custom_processor_types is not None
                and isinstance(processor, custom_processor_types[0])
            ):
                processor.process_frame(
                    {"camera": camera, "motion": motion_boxes}, yuv_frame
                )

        self.frame_manager.close(frame_name)

    def _process_deferred_results(self) -> None:
        """Drain results from deferred processors and perform IPC side-effects."""
        for processor in self.realtime_processors:
            results = processor.drain_results()

            for result in results:
                if result.get("type") == "face_snapshot":
                    payload = {
                        key: result[key]
                        for key in (
                            "camera",
                            "event_id",
                            "frame_time",
                            "person_box",
                            "face_box",
                            "sub_label",
                            "face_score",
                            "artifact_path",
                            "transaction_id",
                        )
                    }
                    try:
                        self.face_result_queue.put_nowait(payload)
                    except queue.Full:
                        logger.warning(
                            "Dropping face snapshot because the result queue is full"
                        )
                        try:
                            os.unlink(str(payload["artifact_path"]))
                        except FileNotFoundError:
                            pass
                    continue

                if result.get("type") != "classification":
                    continue

                if result["processor"] == "state":
                    self.requestor.send_data(
                        f"{result['camera']}/classification/{result['model_name']}",
                        result["state"],
                    )
                elif result["processor"] == "object":
                    object_id = result["object_id"]
                    camera = result["camera"]
                    timestamp = result["timestamp"]
                    model_name = result["model_name"]
                    label = result["label"]
                    score = result["score"]
                    classification_type = result["classification_type"]

                    if classification_type == ObjectClassificationType.sub_label:
                        self.event_metadata_publisher.publish(
                            (object_id, label, score),
                            EventMetadataTypeEnum.sub_label,
                        )
                        self.requestor.send_data(
                            "tracked_object_update",
                            json.dumps(
                                {
                                    "type": TrackedObjectUpdateTypesEnum.classification,
                                    "id": object_id,
                                    "camera": camera,
                                    "timestamp": timestamp,
                                    "model": model_name,
                                    "sub_label": label,
                                    "score": score,
                                }
                            ),
                        )
                    elif classification_type == ObjectClassificationType.attribute:
                        self.event_metadata_publisher.publish(
                            (object_id, model_name, label, score),
                            EventMetadataTypeEnum.attribute.value,
                        )
                        self.requestor.send_data(
                            "tracked_object_update",
                            json.dumps(
                                {
                                    "type": TrackedObjectUpdateTypesEnum.classification,
                                    "id": object_id,
                                    "camera": camera,
                                    "timestamp": timestamp,
                                    "model": model_name,
                                    "attribute": label,
                                    "score": score,
                                }
                            ),
                        )

    def _embed_thumbnail(self, event_id: str, thumbnail: bytes) -> None:
        """Embed the thumbnail for an event."""
        if not self.config.semantic_search.enabled:
            return
        if self.embeddings is None:
            logger.error("Semantic search is enabled but embeddings are unavailable")
            return

        try:
            self.embeddings.embed_thumbnail(event_id, thumbnail)
        except ValueError:
            logger.warning(f"Failed to embed thumbnail for event {event_id}")
