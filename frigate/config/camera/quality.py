"""Per-camera evidence and recognition-input quality configuration."""

from enum import Enum

from pydantic import Field, model_validator

from ..base import FrigateBaseModel


class QualitySourceRoleEnum(str, Enum):
    detect = "detect"
    evidence = "evidence"
    record = "record"


class EvidenceBufferConfig(FrigateBaseModel):
    window_seconds: float = Field(default=3.0, ge=1.0, le=10.0)
    max_bytes: int = Field(
        default=32 * 1024 * 1024,
        ge=8 * 1024 * 1024,
        le=256 * 1024 * 1024,
    )
    sample_fps: float = Field(default=5.0, gt=0)


class TaskQualityConfig(FrigateBaseModel):
    min_detail_width_px: int = Field(ge=1)
    min_detail_height_px: int = Field(ge=1)
    min_laplacian_variance: float = Field(default=20.0, ge=0)
    max_dark_fraction: float = Field(default=0.75, ge=0, le=1)
    max_bright_fraction: float = Field(default=0.75, ge=0, le=1)
    min_aspect_ratio: float = Field(default=0.0, ge=0)
    max_aspect_ratio: float = Field(default=100.0, gt=0)
    min_edge_clearance_px: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_aspect_ratio(self) -> "TaskQualityConfig":
        if self.max_aspect_ratio < self.min_aspect_ratio:
            raise ValueError("max_aspect_ratio must be >= min_aspect_ratio")
        return self


def _default_face_quality() -> TaskQualityConfig:
    return TaskQualityConfig(min_detail_width_px=24, min_detail_height_px=24)


def _default_lpr_quality() -> TaskQualityConfig:
    return TaskQualityConfig(
        min_detail_width_px=24,
        min_detail_height_px=14,
        min_aspect_ratio=1.2,
        max_aspect_ratio=6.0,
        min_edge_clearance_px=1,
    )


class CameraQualityConfig(FrigateBaseModel):
    enabled: bool = False
    source_role: QualitySourceRoleEnum = QualitySourceRoleEnum.detect
    buffer: EvidenceBufferConfig = Field(default_factory=EvidenceBufferConfig)
    top_k: int = Field(default=3, ge=1, le=5)
    face: TaskQualityConfig = Field(default_factory=_default_face_quality)
    lpr: TaskQualityConfig = Field(default_factory=_default_lpr_quality)

    @model_validator(mode="after")
    def validate_source_adapter(self) -> "CameraQualityConfig":
        if self.source_role != QualitySourceRoleEnum.detect:
            raise ValueError(
                "quality.source_role supports only 'detect' in Phase 4; "
                "the evidence/record high-resolution adapter is not implemented"
            )
        return self


class RecognitionLifecycleConfig(FrigateBaseModel):
    """Bounded per-track Face/LPR recognition policy."""

    max_attempts: int = Field(default=3, ge=1, le=5)
    candidate_collection_seconds: float = Field(default=0.4, ge=0)
    min_candidate_interval_seconds: float = Field(default=0.4, ge=0)
    max_candidate_bbox_iou: float = Field(default=0.90, ge=0, le=1)
    passage_idle_seconds: float = Field(default=1.0, gt=0, le=10.0)
    lpr_min_consensus_votes: int = Field(default=2, ge=1)
    lpr_observation_threshold: float = Field(default=0.55, gt=0, le=1)

    @model_validator(mode="after")
    def validate_attempt_contract(self) -> "RecognitionLifecycleConfig":
        if self.lpr_min_consensus_votes > self.max_attempts:
            raise ValueError(
                "recognition_lifecycle.lpr_min_consensus_votes must be less than "
                "or equal to recognition_lifecycle.max_attempts"
            )
        return self
