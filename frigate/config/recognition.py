"""Recognition runtime configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from .base import FrigateBaseModel


class RecognitionRuntimeEnum(StrEnum):
    LOCAL = "local"
    EXTERNAL = "external"


class RecognitionTlsConfig(FrigateBaseModel):
    ca: str = Field(default="", title="Client CA certificate")
    certificate: str = Field(default="", title="Client certificate")
    key: str = Field(default="", title="Client private key")
    server_name: str | None = Field(default=None, title="TLS server name")


class RecognitionRuntimeConfig(FrigateBaseModel):
    runtime: RecognitionRuntimeEnum = Field(
        default=RecognitionRuntimeEnum.LOCAL,
        title="Recognition runtime",
    )
    endpoint: str = Field(default="", title="External recognition endpoint")
    deadline: float = Field(default=5.0, gt=0, le=60, title="Job deadline")
    observation_capacity: int = Field(default=128, gt=0, le=4096)
    control_capacity: int = Field(default=64, gt=0, le=4096)
    outcome_capacity: int = Field(default=128, gt=0, le=4096)
    shutdown_drain: float = Field(default=10.0, gt=0, le=120)
    tls: RecognitionTlsConfig = Field(default_factory=RecognitionTlsConfig)

    @model_validator(mode="after")
    def validate_external(self) -> Self:
        if self.runtime is RecognitionRuntimeEnum.EXTERNAL:
            if not self.endpoint:
                raise ValueError("external recognition runtime requires endpoint")
            missing = [
                name
                for name in ("ca", "certificate", "key")
                if not getattr(self.tls, name)
            ]
            if missing:
                raise ValueError(
                    "external recognition runtime requires TLS fields: "
                    + ", ".join(missing)
                )
        return self
