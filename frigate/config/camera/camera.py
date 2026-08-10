import os
from enum import Enum

from pydantic import Field, PrivateAttr, model_validator

from frigate.const import CACHE_DIR, CACHE_SEGMENT_FORMAT, REGEX_CAMERA_NAME
from frigate.ffmpeg_presets import (
    parse_preset_hardware_acceleration_decode,
    parse_preset_hardware_acceleration_scale,
    parse_preset_input,
    parse_preset_output_record,
)
from frigate.util.builtin import (
    escape_special_characters,
    generate_color_palette,
    get_ffmpeg_arg_list,
)

from ..base import FrigateBaseModel
from ..classification import (
    CameraAudioTranscriptionConfig,
    CameraFaceRecognitionConfig,
    CameraLicensePlateRecognitionConfig,
    CameraSemanticSearchConfig,
)
from .audio import AudioConfig
from .birdseye import BirdseyeCameraConfig
from .detect import DetectConfig
from .ffmpeg import CameraFfmpegConfig, CameraInput
from .live import CameraLiveConfig
from .motion import MotionConfig
from .mqtt import CameraMqttConfig
from .notification import CameraNotificationConfig
from .objects import ObjectConfig
from .onvif import OnvifConfig
from .profile import CameraProfileConfig
from .quality import CameraQualityConfig, RecognitionLifecycleConfig
from .record import RecordConfig
from .review import ReviewConfig
from .snapshots import SnapshotsConfig
from .timestamp import TimestampStyleConfig
from .ui import CameraUiConfig
from .zone import ZoneConfig

__all__ = ["CameraConfig"]


class CameraTypeEnum(str, Enum):
    generic = "generic"
    lpr = "lpr"


class CameraConfig(FrigateBaseModel):
    name: str | None = Field(
        None,
        title="Camera name",
        description="Camera name is required",
        pattern=REGEX_CAMERA_NAME,
    )

    friendly_name: str | None = Field(
        None,
        title="Friendly name",
        description="Camera friendly name used in the Frigate UI",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_friendly_name(cls, values):
        if isinstance(values, dict) and "friendly_name" in values:
            pass
        return values

    enabled: bool = Field(default=True, title="Enabled", description="Enabled")

    # Options with global fallback
    audio: AudioConfig = Field(
        default_factory=AudioConfig,
        title="Audio detection",
        description="Settings for audio-based event detection for this camera.",
    )
    audio_transcription: CameraAudioTranscriptionConfig = Field(
        default_factory=CameraAudioTranscriptionConfig,
        title="Audio transcription",
        description="Settings for live and speech audio transcription used for events and live captions.",
    )
    birdseye: BirdseyeCameraConfig = Field(
        default_factory=BirdseyeCameraConfig,
        title="Birdseye",
        description="Settings for the Birdseye composite view that composes multiple camera feeds into a single layout.",
    )
    detect: DetectConfig = Field(
        default_factory=DetectConfig,
        title="Object Detection",
        description="Settings for the detection/detect role used to run object detection and initialize trackers.",
    )
    face_recognition: CameraFaceRecognitionConfig = Field(
        default_factory=CameraFaceRecognitionConfig,
        title="Face recognition",
        description="Settings for face detection and recognition for this camera.",
    )
    ffmpeg: CameraFfmpegConfig = Field(
        title="Streams (FFmpeg)",
        description="Camera stream inputs and FFmpeg options, including binary path, args, hwaccel, and per-role output args.",
    )
    live: CameraLiveConfig = Field(
        default_factory=CameraLiveConfig,
        title="Live playback",
        description="Settings used by the Web UI to control live stream selection, resolution and quality.",
    )
    lpr: CameraLicensePlateRecognitionConfig = Field(
        default_factory=CameraLicensePlateRecognitionConfig,
        title="License Plate Recognition",
        description="License plate recognition settings including detection thresholds, formatting, and known plates.",
    )
    motion: MotionConfig = Field(
        default_factory=MotionConfig,
        title="Motion detection",
        description="Default motion detection settings for this camera.",
    )
    quality: CameraQualityConfig = Field(
        default_factory=CameraQualityConfig,
        title="Recognition input quality",
        description="Bounded detect-frame evidence and shared Face/LPR candidate quality settings.",
    )
    recognition_lifecycle: RecognitionLifecycleConfig = Field(
        default_factory=RecognitionLifecycleConfig,
        title="Recognition lifecycle",
        description="Shared bounded retry and candidate diversity policy for Face and LPR.",
    )
    objects: ObjectConfig = Field(
        default_factory=ObjectConfig,
        title="Objects",
        description="Object tracking defaults including which labels to track and per-object filters.",
    )
    record: RecordConfig = Field(
        default_factory=RecordConfig,
        title="Recording",
        description="Recording and retention settings for this camera.",
    )
    review: ReviewConfig = Field(
        default_factory=ReviewConfig,
        title="Review",
        description="Settings that control alerts, detections, and GenAI review summaries used by the UI and storage for this camera.",
    )
    semantic_search: CameraSemanticSearchConfig = Field(
        default_factory=CameraSemanticSearchConfig,
        title="Semantic Search",
        description="Settings for semantic search which builds and queries object embeddings to find similar items.",
    )
    snapshots: SnapshotsConfig = Field(
        default_factory=SnapshotsConfig,
        title="Snapshots",
        description="Settings for API-generated snapshots of tracked objects for this camera.",
    )
    timestamp_style: TimestampStyleConfig = Field(
        default_factory=TimestampStyleConfig,
        title="Timestamp style",
        description="Styling options for timestamps applied to snapshots and Debug view.",
    )

    # Options without global fallback
    best_image_timeout: int = Field(
        default=60,
        title="Best image timeout",
        description="How long to wait for the image with the highest confidence score.",
    )
    mqtt: CameraMqttConfig = Field(
        default_factory=CameraMqttConfig,
        title="MQTT",
        description="MQTT image publishing settings.",
    )
    notifications: CameraNotificationConfig = Field(
        default_factory=CameraNotificationConfig,
        title="Notifications",
        description="Settings to enable and control notifications for this camera.",
    )
    onvif: OnvifConfig = Field(
        default_factory=OnvifConfig,
        title="ONVIF",
        description="ONVIF connection and PTZ autotracking settings for this camera.",
    )
    type: CameraTypeEnum = Field(
        default=CameraTypeEnum.generic,
        title="Camera type",
        description="Camera Type",
    )
    ui: CameraUiConfig = Field(
        default_factory=CameraUiConfig,
        title="Camera UI",
        description="Display ordering and visibility for this camera in the UI. Ordering affects the default dashboard. For more granular control, use camera groups.",
    )
    webui_url: str | None = Field(
        None,
        title="Camera URL",
        description="URL to visit the camera directly from system page",
    )

    profiles: dict[str, CameraProfileConfig] = Field(
        default_factory=dict,
        title="Profiles",
        description="Named config profiles with partial overrides that can be activated at runtime.",
    )
    zones: dict[str, ZoneConfig] = Field(
        default_factory=dict,
        title="Zones",
        description="Zones allow you to define a specific area of the frame so you can determine whether or not an object is within a particular area.",
    )
    enabled_in_config: bool | None = Field(
        default=None,
        title="Original camera state",
        description="Keep track of original state of camera.",
    )

    _ffmpeg_cmds: list[dict[str, list[str]]] = PrivateAttr()

    @model_validator(mode="after")
    def validate_quality_contract(self) -> "CameraConfig":
        lifecycle = self.recognition_lifecycle
        if (
            lifecycle.min_candidate_interval_seconds
            > self.quality.buffer.window_seconds
        ):
            raise ValueError(
                "recognition_lifecycle.min_candidate_interval_seconds must not "
                "exceed quality.buffer.window_seconds"
            )
        if lifecycle.max_attempts > self.quality.top_k:
            raise ValueError(
                "recognition_lifecycle.max_attempts must be less than or equal "
                "to quality.top_k when quality is enabled"
            )
        if self.quality.enabled and self.quality.buffer.sample_fps > self.detect.fps:
            raise ValueError(
                "quality.buffer.sample_fps must be less than or equal to detect.fps"
            )
        if (
            self.quality.enabled
            and self.detect.width is not None
            and self.detect.height is not None
        ):
            required = (
                self.detect.width * self.detect.height * 3 // 2 * self.quality.top_k
            )
            if self.quality.buffer.max_bytes < required:
                raise ValueError(
                    "quality.buffer.max_bytes must hold at least quality.top_k "
                    "raw I420 detect frames"
                )
        return self

    def __init__(self, **config):
        # Set zone colors
        if "zones" in config:
            colors = generate_color_palette(len(config["zones"]))

            config["zones"] = {
                name: {**z, "color": color}
                for (name, z), color in zip(config["zones"].items(), colors)
            }

        # add roles to the input if there is only one
        if len(config["ffmpeg"]["inputs"]) == 1:
            has_audio = "audio" in config["ffmpeg"]["inputs"][0].get("roles", [])

            config["ffmpeg"]["inputs"][0]["roles"] = [
                "record",
                "detect",
            ]

            if has_audio:
                config["ffmpeg"]["inputs"][0]["roles"].append("audio")

        super().__init__(**config)

    @property
    def frame_shape(self) -> tuple[int, int]:
        if self.detect.height is None or self.detect.width is None:
            raise RuntimeError("detect dimensions have not been resolved")
        return self.detect.height, self.detect.width

    @property
    def frame_shape_yuv(self) -> tuple[int, int]:
        height, width = self.frame_shape
        return height * 3 // 2, width

    @property
    def ffmpeg_cmds(self) -> list[dict[str, list[str]]]:
        return self._ffmpeg_cmds

    def get_formatted_name(self) -> str:
        """Return the friendly name if set, otherwise return a formatted version of the camera name."""
        if self.friendly_name:
            return self.friendly_name
        return self.name.replace("_", " ").title() if self.name else ""

    def create_ffmpeg_cmds(self):
        if "_ffmpeg_cmds" in self:
            return
        self._build_ffmpeg_cmds()

    def recreate_ffmpeg_cmds(self):
        """Force regeneration of ffmpeg commands from current config."""
        self._build_ffmpeg_cmds()

    def _build_ffmpeg_cmds(self):
        """Build ffmpeg commands from the current ffmpeg config."""
        ffmpeg_cmds = []
        for ffmpeg_input in self.ffmpeg.inputs:
            ffmpeg_cmd = self._get_ffmpeg_cmd(ffmpeg_input)
            if ffmpeg_cmd is None:
                continue

            ffmpeg_cmds.append(
                {"roles": [role.value for role in ffmpeg_input.roles], "cmd": ffmpeg_cmd}
            )
        self._ffmpeg_cmds = ffmpeg_cmds

    def _get_ffmpeg_cmd(self, ffmpeg_input: CameraInput):
        ffmpeg_output_args = []
        detect_height, detect_width = self.frame_shape
        if "detect" in ffmpeg_input.roles:
            detect_args = get_ffmpeg_arg_list(self.ffmpeg.output_args.detect)
            scale_detect_args = parse_preset_hardware_acceleration_scale(
                ffmpeg_input.hwaccel_args or self.ffmpeg.hwaccel_args,
                detect_args,
                self.detect.fps,
                detect_width,
                detect_height,
            )

            ffmpeg_output_args = scale_detect_args + ffmpeg_output_args + ["pipe:"]

        if "record" in ffmpeg_input.roles and self.record.enabled:
            record_args = get_ffmpeg_arg_list(
                parse_preset_output_record(
                    self.ffmpeg.output_args.record,
                    self.ffmpeg.apple_compatibility,
                )
                or self.ffmpeg.output_args.record
            )

            ffmpeg_output_args = (
                record_args
                + [
                    f"{os.path.join(CACHE_DIR, self.name or '')}@{CACHE_SEGMENT_FORMAT}.mp4"
                ]
                + ffmpeg_output_args
            )

        # if there aren't any outputs enabled for this input
        if len(ffmpeg_output_args) == 0:
            return None

        global_args = get_ffmpeg_arg_list(
            ffmpeg_input.global_args or self.ffmpeg.global_args
        )

        camera_arg = (
            self.ffmpeg.hwaccel_args if self.ffmpeg.hwaccel_args != "auto" else None
        )
        hwaccel_args = get_ffmpeg_arg_list(
            parse_preset_hardware_acceleration_decode(
                ffmpeg_input.hwaccel_args,
                self.detect.fps,
                detect_width,
                detect_height,
                self.ffmpeg.gpu,
            )
            or ffmpeg_input.hwaccel_args
            or parse_preset_hardware_acceleration_decode(
                camera_arg,
                self.detect.fps,
                detect_width,
                detect_height,
                self.ffmpeg.gpu,
            )
            or camera_arg
            or []
        )
        input_args = get_ffmpeg_arg_list(
            parse_preset_input(ffmpeg_input.input_args, self.detect.fps)
            or ffmpeg_input.input_args
            or parse_preset_input(self.ffmpeg.input_args, self.detect.fps)
            or self.ffmpeg.input_args
        )

        cmd = (
            [self.ffmpeg.ffmpeg_path]
            + global_args
            + (hwaccel_args if "detect" in ffmpeg_input.roles else [])
            + input_args
            + ["-i", escape_special_characters(ffmpeg_input.path)]
            + ffmpeg_output_args
        )

        return [part for part in cmd if part != ""]
