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


def _default_face_quality() -> TaskQualityConfig:
    return TaskQualityConfig(min_detail_width_px=24, min_detail_height_px=24)


def _default_lpr_quality() -> TaskQualityConfig:
    return TaskQualityConfig(min_detail_width_px=24, min_detail_height_px=14)


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
