import datetime
import logging
import multiprocessing as mp
import os
import secrets
import shutil
from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path

import psutil
import uvicorn
import multiprocessing.shared_memory
from peewee_migrate import Router
from playhouse.sqlite_ext import SqliteExtDatabase

from frigate.api.auth import hash_password
from frigate.api.fastapi_app import create_fastapi_app
from frigate.domain.camera import CameraMetrics, PTZMetrics
from frigate.domain.camera.maintainer import CameraMaintainer
from frigate.domain.camera.runtime import (
    CAMERA_RUNTIME_MODELS,
    camera_runtime_config,
    ensure_runtime_dirs,
    start_detector_runtime,
)
from frigate.infrastructure.comms.base_communicator import Communicator
from frigate.infrastructure.comms.dispatcher import Dispatcher
from frigate.infrastructure.comms.event_metadata_updater import EventMetadataPublisher
from frigate.infrastructure.comms.inter_process import InterProcessCommunicator
from frigate.infrastructure.comms.mqtt import MqttClient
from frigate.infrastructure.comms.object_detector_signaler import DetectorProxy
from frigate.infrastructure.comms.ws import WebSocketClient
from frigate.infrastructure.comms.zmq_proxy import ZmqProxy
from frigate.infrastructure.config.camera.updater import CameraConfigUpdatePublisher
from frigate.infrastructure.config.config import FrigateConfig
from frigate.infrastructure.config.holder import ConfigHolder
from frigate.infrastructure.config.profile_manager import ProfileManager
from frigate.const import (
    CONFIG_DIR,
)
from frigate.infrastructure.data_processing.types import DataProcessorMetrics
from frigate.infrastructure.db.sqlitevecq import SqliteVecQueueDatabase
from frigate.debug_replay import (
    DebugReplayManager,
    cleanup_replay_cameras,
)
from frigate.application.embeddings import EmbeddingProcess, EmbeddingsContext
from frigate.application.events.audio import AudioProcessor
from frigate.application.events.cleanup import EventCleanup
from frigate.application.events.maintainer import EventProcessor
from frigate.application.jobs.export import reap_stale_exports
from frigate.application.jobs.motion_search import stop_all_motion_search_jobs
from frigate.log import _stop_logging
from frigate.models import (
    EdgeMediaManifest,
    Event,
    EventEvidence,
    EventObservation,
    Export,
    MediaArtifact,
    NotificationDelivery,
    NotificationIntent,
    NotificationRuleState,
    Previews,
    RecordingsToDelete,
    ReviewSegment,
    Timeline,
    TrackerJournalEntry,
    Trigger,
    User,
)
from frigate.application.notifications.client import NotificationClient
from frigate.domain.object_detection.base import ObjectDetectProcess
from frigate.infrastructure.output.output import OutputProcess
from frigate.domain.ptz.autotrack import PtzAutoTrackerThread
from frigate.domain.ptz.onvif import OnvifController
from frigate.domain.record.cleanup import RecordingCleanup
from frigate.domain.record.export import migrate_exports
from frigate.domain.record.record import RecordProcess
from frigate.application.review.review import ReviewProcess
from extension.topology.compiler import compile_topology
from frigate.application.stats.emitter import StatsEmitter
from frigate.application.stats.util import stats_init
from frigate.storage import StorageMaintainer
from frigate.timeline import TimelineProcessor
from frigate.domain.track.object_processing import TrackedObjectProcessor
from extension.tracker.adapters.frigate import TrackerMaintainer
from frigate.util.builtin import empty_and_close_queue
from frigate.util.process import FrigateProcess
from frigate.util.services import set_file_limit
from frigate.version import VERSION
from frigate.watchdog import FrigateWatchdog

logger = logging.getLogger(__name__)


class FrigateApp:
    def __init__(
        self, config: FrigateConfig, manager: SyncManager, stop_event: MpEvent
    ) -> None:
        self.metrics_manager = manager
        self.audio_process: mp.Process | None = None
        self.stop_event = stop_event
        camera_count = max(1, len(config.cameras))
        self.detection_queue: Queue = mp.Queue(maxsize=camera_count * 4)
        self.detectors: dict[str, ObjectDetectProcess] = {}
        self.detection_shms: list[mp.shared_memory.SharedMemory] = []
        self.log_queue: Queue = mp.Queue(maxsize=10000)
        self.camera_metrics: DictProxy = self.metrics_manager.dict()
        # The embeddings maintainer always runs, even when all enrichment
        # features are disabled, and owns shared quality/lifecycle metrics.
        # Keep its metrics contract total instead of passing None into every
        # processor and crashing during the maintenance loop.
        self.embeddings_metrics = DataProcessorMetrics(
            self.metrics_manager, list(config.classification.custom.keys())
        )
        self.ptz_metrics: dict[str, PTZMetrics] = {}
        self.processes: dict[str, int] = {}
        self.embeddings: EmbeddingsContext | None = None
        self.profile_manager: ProfileManager | None = None
        self.config_holder = ConfigHolder(config)
        self.topology_plan = compile_topology(config)

    @property
    def config(self) -> FrigateConfig:
        """The current config, not the one Frigate booted with.

        Read through the holder so the deferred watchdog factories below build
        a replacement process from the config as it is now. There is no setter
        on purpose: a plain attribute would let a caller pin this back to a
        single object and reintroduce the staleness.
        """
        return self.config_holder.config

    @property
    def camera_runtime_config(self) -> FrigateConfig:
        """Config view containing only cameras locally owned by Frigate main."""
        return camera_runtime_config(self.config, topology=self.topology_plan)

    def ensure_dirs(self) -> None:
        ensure_runtime_dirs(self.config)

    def init_debug_replay_manager(self) -> None:
        self.replay_manager = DebugReplayManager()

    def init_camera_metrics(self) -> None:
        # create camera_metrics
        for camera_name in self.config.cameras.keys():
            self.camera_metrics[camera_name] = CameraMetrics(self.metrics_manager)
            self.ptz_metrics[camera_name] = PTZMetrics(
                autotracker_enabled=self.config.cameras[
                    camera_name
                ].onvif.autotracking.enabled
            )

    def init_queues(self) -> None:
        # Queue for cameras to push tracked objects to
        # leaving room for 2 extra cameras to be added
        self.detected_frames_queue: Queue = mp.Queue(
            maxsize=(
                sum(
                    camera.enabled_in_config == True
                    for camera in self.config.cameras.values()
                )
                + 2
            )
            * 2
        )

        # Queue for timeline events
        self.timeline_queue: Queue = mp.Queue(
            maxsize=max(128, len(self.config.cameras) * 32)
        )
        face_queue_size = max(
            4,
            min(
                32,
                4
                * sum(
                    camera.enabled_in_config
                    for camera in self.config.cameras.values()
                ),
            ),
        )
        self.face_result_queue: Queue = mp.Queue(maxsize=face_queue_size)
        self.face_commit_queue: Queue = mp.Queue(maxsize=face_queue_size)
        self.face_completion_queue: Queue = mp.Queue(maxsize=face_queue_size * 2)
        self.event_update_queue: Queue = mp.Queue(
            maxsize=max(256, len(self.config.cameras) * 64)
        )
        self.tracker_event_commit_queue: Queue = mp.Queue(
            maxsize=max(64, len(self.config.cameras) * 16)
        )

    def init_database(self) -> None:
        def vacuum_db(db: SqliteExtDatabase) -> None:
            logger.info("Running database vacuum")
            db.execute_sql("VACUUM;")

            try:
                with open(f"{CONFIG_DIR}/.vacuum", "w") as f:
                    f.write(str(datetime.datetime.now().timestamp()))
            except PermissionError:
                logger.error("Unable to write to /config to save DB state")

        # Migrate DB schema
        migrate_db = SqliteExtDatabase(self.config.database.path)

        # Run migrations
        del logging.getLogger("peewee_migrate").handlers[:]
        router = Router(migrate_db)

        if len(router.diff) > 0:
            logger.info("Making backup of DB before migrations...")
            shutil.copyfile(
                self.config.database.path,
                self.config.database.path.replace("frigate.infrastructure.db", "backup.db"),
            )

        router.run()

        # check if vacuum needs to be run
        if os.path.exists(f"{CONFIG_DIR}/.vacuum"):
            with open(f"{CONFIG_DIR}/.vacuum") as f:
                try:
                    timestamp = round(float(f.readline()))
                except Exception:
                    timestamp = 0

                if (
                    timestamp
                    < (
                        datetime.datetime.now() - datetime.timedelta(weeks=2)
                    ).timestamp()
                ):
                    vacuum_db(migrate_db)
        else:
            vacuum_db(migrate_db)

        migrate_db.close()

    def init_go2rtc(self) -> None:
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == "go2rtc":
                logger.info(f"go2rtc process pid: {proc.info['pid']}")
                self.processes["go2rtc"] = proc.info["pid"]

    def init_recording_manager(self) -> None:
        if not self.camera_runtime_config.cameras:
            self.recording_process = None
            logger.info("No embedded cameras assigned; local recorder is disabled")
            return
        recording_process = RecordProcess(self.camera_runtime_config, self.stop_event)
        self.recording_process = recording_process
        recording_process.start()
        self.processes["recording"] = recording_process.pid or 0
        logger.info(f"Recording process started: {recording_process.pid}")

    def init_review_segment_manager(self) -> None:
        review_segment_process = ReviewProcess(self.config, self.stop_event)
        self.review_segment_process = review_segment_process
        review_segment_process.start()
        self.processes["review_segment"] = review_segment_process.pid or 0
        logger.info(f"Review process started: {review_segment_process.pid}")

    def init_embeddings_manager(self) -> None:
        # always start the embeddings process
        embedding_process = EmbeddingProcess(
            self.config,
            self.embeddings_metrics,
            self.stop_event,
            self.face_result_queue,
        )
        self.embedding_process = embedding_process
        embedding_process.start()
        self.processes["embeddings"] = embedding_process.pid or 0
        logger.info(f"Embedding process started: {embedding_process.pid}")

    def bind_database(self) -> None:
        """Bind db to the main process."""
        # NOTE: all db accessing processes need to be created before the db can be bound to the main process
        self.db = SqliteVecQueueDatabase(
            self.config.database.path,
            pragmas={
                "auto_vacuum": "FULL",  # Does not defragment database
                "cache_size": -512 * 1000,  # 512MB of cache,
                "synchronous": "NORMAL",  # Safe when using WAL https://www.sqlite.org/pragma.html#pragma_synchronous
            },
            timeout=max(
                60,
                10
                * len([c for c in self.config.cameras.values() if c.enabled_in_config]),
            ),
            load_vec_extension=True,
        )
        models = [
            Event,
            EventEvidence,
            EventObservation,
            EdgeMediaManifest,
            Export,
            Previews,
            *CAMERA_RUNTIME_MODELS,
            RecordingsToDelete,
            Timeline,
            TrackerJournalEntry,
            User,
            Trigger,
            NotificationDelivery,
            NotificationIntent,
            NotificationRuleState,
            MediaArtifact,
        ]
        self.db.bind(models)

    def check_db_data_migrations(self) -> None:
        # check if vacuum needs to be run
        if not os.path.exists(f"{CONFIG_DIR}/.exports"):
            try:
                with open(f"{CONFIG_DIR}/.exports", "w") as f:
                    f.write(str(datetime.datetime.now().timestamp()))
            except PermissionError:
                logger.error("Unable to write to /config to save export state")

            migrate_exports(self.config.ffmpeg, list(self.config.cameras.keys()))

    def init_embeddings_client(self) -> None:
        # Create a client for other processes to use
        self.embeddings = EmbeddingsContext(self.db)

    def init_inter_process_communicator(self) -> None:
        self.inter_process_communicator = InterProcessCommunicator()
        self.inter_config_updater = CameraConfigUpdatePublisher()
        self.event_metadata_updater = EventMetadataPublisher()
        self.inter_zmq_proxy = ZmqProxy()
        self.detection_proxy = DetectorProxy()

    def init_onvif(self) -> None:
        self.onvif_controller = OnvifController(
            self.camera_runtime_config, self.ptz_metrics
        )

    def init_dispatcher(self) -> None:
        comms: list[Communicator] = []

        if self.config.mqtt.enabled:
            comms.append(MqttClient(self.config))

        # The shared client always exists so provider/config changes take effect
        # on hot reload without restarting Frigate.
        comms.append(NotificationClient(self.config, self.stop_event))

        comms.append(WebSocketClient(self.config))
        comms.append(self.inter_process_communicator)

        self.dispatcher = Dispatcher(
            self.config,
            self.inter_config_updater,
            self.onvif_controller,
            self.ptz_metrics,
            comms,
            edge_control=lambda camera, operation, payload: (
                hasattr(self, "tracker_maintainer")
                and self.tracker_maintainer.control_camera(
                    camera, operation, payload
                )
            ),
            edge_cameras=frozenset(self.topology_plan.camera_owners),
        )

    def init_profile_manager(self) -> None:
        self.profile_manager = ProfileManager(
            self.config, self.inter_config_updater, self.dispatcher
        )
        self.dispatcher.profile_manager = self.profile_manager

    def restore_active_profile(self) -> None:
        """Re-activate the persisted profile after subscribers are connected.

        ZMQ PUB/SUB drops messages with no subscribers, so activation must
        run after every config_updater subscriber is up.
        """
        if self.profile_manager is None:
            return

        persisted = ProfileManager.load_persisted_profile()
        if persisted and any(
            persisted in cam.profiles for cam in self.config.cameras.values()
        ):
            logger.info("Restoring persisted profile '%s'", persisted)
            # runtime overrides are layered on top via restore_runtime_state()
            self.profile_manager.activate_profile(
                persisted, clear_runtime_overrides=False
            )

    def start_detectors(self) -> None:
        runtime_config = self.camera_runtime_config
        if not runtime_config.cameras:
            logger.info("No embedded cameras assigned; local detectors are disabled")
            return
        detector_runtime = start_detector_runtime(
            runtime_config,
            self.detection_queue,
            self.stop_event,
        )
        self.detectors = detector_runtime.processes
        self.detection_shms.extend(detector_runtime.shared_memory)

    def start_ptz_autotracker(self) -> None:
        self.ptz_autotracker_thread = PtzAutoTrackerThread(
            self.camera_runtime_config,
            self.onvif_controller,
            self.ptz_metrics,
            self.dispatcher,
            self.stop_event,
        )
        if self.camera_runtime_config.cameras:
            self.ptz_autotracker_thread.start()

    def start_detected_frames_processor(self) -> None:
        self.detected_frames_processor = TrackedObjectProcessor(
            self.camera_runtime_config,
            self.dispatcher,
            self.detected_frames_queue,
            self.ptz_autotracker_thread,
            self.stop_event,
            self.face_result_queue,
            self.face_commit_queue,
            self.face_completion_queue,
            self.event_update_queue,
        )
        if self.camera_runtime_config.cameras:
            self.detected_frames_processor.start()

    def start_video_output_processor(self) -> None:
        if not self.camera_runtime_config.cameras:
            self.output_processor = None
            logger.info("No embedded cameras assigned; local video output is disabled")
            return
        output_processor = OutputProcess(self.camera_runtime_config, self.stop_event)
        self.output_processor = output_processor
        output_processor.start()
        logger.info(f"Output process started: {output_processor.pid}")

    def start_camera_processor(self) -> None:
        self.camera_maintainer = CameraMaintainer(
            self.camera_runtime_config,
            self.detection_queue,
            self.detected_frames_queue,
            self.camera_metrics,
            self.ptz_metrics,
            self.stop_event,
            self.metrics_manager,
        )
        self.camera_maintainer.start()

    def start_audio_processor(self) -> None:
        if not self.camera_runtime_config.cameras:
            self.audio_process = None
            logger.info("No embedded cameras assigned; local audio processor is disabled")
            return
        self.audio_process = AudioProcessor(
            self.camera_runtime_config, self.camera_metrics, self.stop_event
        )
        self.audio_process.start()
        self.processes["audio_detector"] = self.audio_process.pid or 0

    def start_timeline_processor(self) -> None:
        self.timeline_processor = TimelineProcessor(
            self.config, self.timeline_queue, self.stop_event
        )
        self.timeline_processor.start()

    def start_event_processor(self) -> None:
        self.event_processor = EventProcessor(
            self.config,
            self.timeline_queue,
            self.stop_event,
            self.face_commit_queue,
            self.face_completion_queue,
            self.event_update_queue,
            self.tracker_event_commit_queue,
        )
        self.event_processor.start()

    def start_tracker_maintainer(self) -> None:
        self.tracker_maintainer = TrackerMaintainer(
            self.config,
            self.topology_plan,
            self.db,
            self.event_update_queue,
            self.tracker_event_commit_queue,
            self.stop_event,
        )
        self.tracker_maintainer.start()

    def start_event_cleanup(self) -> None:
        self.event_cleanup = EventCleanup(self.config, self.stop_event, self.db)
        self.event_cleanup.start()

    def start_record_cleanup(self) -> None:
        self.record_cleanup = RecordingCleanup(self.config, self.stop_event)
        self.record_cleanup.start()

    def start_storage_maintainer(self) -> None:
        self.storage_maintainer = StorageMaintainer(self.config, self.stop_event)
        self.storage_maintainer.start()

    def start_stats_emitter(self) -> None:
        self.stats_emitter = StatsEmitter(
            self.config,
            stats_init(
                self.config,
                self.camera_metrics,
                self.embeddings_metrics,
                self.detectors,
                self.processes,
            ),
            self.stop_event,
        )
        self.stats_emitter.start()

    def start_watchdog(self) -> None:
        self.frigate_watchdog = FrigateWatchdog(self.detectors, self.stop_event)

        # (attribute on self, key in self.processes, factory)
        specs: list[tuple[str, str, Callable[[], FrigateProcess]]] = [
            (
                "embedding_process",
                "embeddings",
                lambda: EmbeddingProcess(
                    self.config,
                    self.embeddings_metrics,
                    self.stop_event,
                    self.face_result_queue,
                ),
            ),
            (
                "recording_process",
                "recording",
                lambda: RecordProcess(self.camera_runtime_config, self.stop_event),
            ),
            (
                "review_segment_process",
                "review_segment",
                lambda: ReviewProcess(self.config, self.stop_event),
            ),
            (
                "output_processor",
                "output",
                lambda: OutputProcess(self.camera_runtime_config, self.stop_event),
            ),
        ]

        for attr, key, factory in specs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                continue

            def on_restart(
                proc: FrigateProcess, _attr: str = attr, _key: str = key
            ) -> None:
                setattr(self, _attr, proc)
                self.processes[_key] = proc.pid or 0

            self.frigate_watchdog.register(
                key, getattr(self, attr), factory, on_restart
            )

        self.frigate_watchdog.start()

    def init_auth(self) -> None:
        if self.config.auth.enabled:
            if User.select().count() == 0:
                password = secrets.token_hex(16)
                password_hash = hash_password(
                    password, iterations=self.config.auth.hash_iterations
                )
                User.insert(
                    {
                        User.username: "admin",
                        User.role: "admin",
                        User.password_hash: password_hash,
                        User.notification_tokens: [],
                    }
                ).execute()

                self.config.auth.admin_first_time_login = True

                logger.info("********************************************************")
                logger.info("********************************************************")
                logger.info("***    Auth is enabled, but no users exist.          ***")
                logger.info("***    Created a default user:                       ***")
                logger.info("***    User: admin                                   ***")
                logger.info(f"***    Password: {password}   ***")
                logger.info("********************************************************")
                logger.info("********************************************************")
            elif self.config.auth.reset_admin_password:
                password = secrets.token_hex(16)
                password_hash = hash_password(
                    password, iterations=self.config.auth.hash_iterations
                )
                User.replace(
                    username="admin",
                    role="admin",
                    password_hash=password_hash,
                    notification_tokens=[],
                ).execute()

                logger.info("********************************************************")
                logger.info("********************************************************")
                logger.info("***    Reset admin password set in the config.       ***")
                logger.info(f"***    Password: {password}   ***")
                logger.info("********************************************************")
                logger.info("********************************************************")

    def start(self) -> None:
        logger.info(f"Starting Frigate ({VERSION})")

        # Ensure global state.
        self.ensure_dirs()

        # Set soft file limits.
        set_file_limit()

        # Start frigate services.
        self.init_debug_replay_manager()
        self.init_camera_metrics()
        self.init_queues()
        self.init_database()
        self.init_onvif()
        self.init_recording_manager()
        self.init_review_segment_manager()
        self.init_go2rtc()
        self.init_embeddings_manager()
        self.bind_database()
        self.check_db_data_migrations()

        # Clean up any stale replay camera artifacts (filesystem + DB)
        cleanup_replay_cameras()

        # Reap any Export rows still marked in_progress from a previous
        # session (crash, kill, broken migration). Runs synchronously before
        # uvicorn binds so no API request can observe a stale row.
        reap_stale_exports()

        self.init_inter_process_communicator()
        self.start_detectors()
        self.init_dispatcher()
        self.init_profile_manager()
        self.init_embeddings_client()
        self.start_video_output_processor()
        self.start_ptz_autotracker()
        self.start_detected_frames_processor()
        self.start_camera_processor()
        self.start_audio_processor()
        self.start_storage_maintainer()
        self.start_stats_emitter()
        self.start_timeline_processor()
        self.start_event_processor()
        self.start_tracker_maintainer()
        self.start_event_cleanup()
        self.start_record_cleanup()
        self.start_watchdog()

        # restore persisted runtime overrides on top of config
        self.restore_active_profile()
        self.dispatcher.restore_runtime_state()

        self.init_auth()

        try:
            uvicorn.run(
                create_fastapi_app(
                    self.config,
                    self.db,
                    self.embeddings,
                    self.detected_frames_processor,
                    self.storage_maintainer,
                    self.onvif_controller,
                    self.stats_emitter,
                    self.event_metadata_updater,
                    self.inter_config_updater,
                    self.replay_manager,
                    self.dispatcher,
                    self.profile_manager,
                    config_holder=self.config_holder,
                    tracker_maintainer=self.tracker_maintainer,
                ),
                host="127.0.0.1",
                port=5001,
                log_level="error",
            )
        finally:
            self.stop()

    def stop(self) -> None:
        logger.info("Stopping...")

        # used by the docker healthcheck
        Path("/dev/shm/.frigate-is-stopping").touch()

        # Cancel any running motion search jobs before setting stop_event
        stop_all_motion_search_jobs()

        if hasattr(self, "tracker_maintainer"):
            self.tracker_maintainer.request_stop()
            self.tracker_maintainer.join(timeout=15)
            self.tracker_maintainer.close()

        self.stop_event.set()

        # set an end_time on entries without an end_time before exiting
        Event.update(
            end_time=datetime.datetime.now().timestamp(), has_snapshot=False
        ).where(Event.end_time == None).execute()
        ReviewSegment.update(end_time=datetime.datetime.now().timestamp()).where(
            ReviewSegment.end_time == None
        ).execute()

        # stop the audio process
        if self.audio_process:
            self.audio_process.terminate()
            self.audio_process.join()

        # stop the onvif controller
        if self.onvif_controller:
            self.onvif_controller.close()

        # ensure the detectors are done
        for detector in self.detectors.values():
            detector.stop()

        empty_and_close_queue(self.detection_queue)
        logger.info("Detection queue closed")

        if self.detected_frames_processor.is_alive():
            self.detected_frames_processor.join()
        empty_and_close_queue(self.detected_frames_queue)
        logger.info("Detected frames queue closed")

        self.timeline_processor.join()
        self.event_processor.join()
        empty_and_close_queue(self.timeline_queue)
        logger.info("Timeline queue closed")

        if self.output_processor is not None:
            self.output_processor.terminate()
            self.output_processor.join()

        if self.recording_process is not None:
            self.recording_process.terminate()
            self.recording_process.join()

        self.review_segment_process.terminate()
        self.review_segment_process.join()

        self.dispatcher.stop()
        if self.ptz_autotracker_thread.is_alive():
            self.ptz_autotracker_thread.join()

        self.event_cleanup.join()
        self.record_cleanup.join()
        self.stats_emitter.join()
        self.frigate_watchdog.join()
        self.camera_maintainer.join()
        self.db.stop()

        # Save embeddings stats to disk
        if self.embeddings:
            self.embeddings.stop()

        # Stop Communicators
        self.inter_process_communicator.stop()
        self.inter_config_updater.stop()
        self.event_metadata_updater.stop()
        self.inter_zmq_proxy.stop()
        self.detection_proxy.stop()

        while len(self.detection_shms) > 0:
            shm = self.detection_shms.pop()
            shm.close()
            shm.unlink()

        _stop_logging()
        self.metrics_manager.shutdown()
