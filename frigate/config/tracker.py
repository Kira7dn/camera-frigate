"""Managed tracker-edge topology configuration."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from .base import FrigateBaseModel


class TrackerTlsConfig(FrigateBaseModel):
    ca: str = ""
    certificate: str = ""
    key: str = ""
    server_name: str = ""


class TrackerEvidenceConfig(FrigateBaseModel):
    memory_bytes_per_camera: int = Field(
        default=32 * 1024 * 1024, ge=1024 * 1024, le=1024 * 1024 * 1024
    )
    ttl: float = Field(default=45, gt=0, le=3600)


class TrackerSpoolConfig(FrigateBaseModel):
    max_bytes: int = Field(
        default=256 * 1024 * 1024, ge=1024 * 1024, le=16 * 1024 * 1024 * 1024
    )
    retention: int = Field(default=24 * 60 * 60, gt=0, le=30 * 24 * 60 * 60)


class TrackerNodeConfig(FrigateBaseModel):
    managed: bool = True
    endpoint: str
    cameras: list[str] = Field(min_length=1)
    deadline: float = Field(default=5, gt=0, le=60)
    output_capacity: int = Field(default=256, gt=0, le=8192)
    control_capacity: int = Field(default=64, gt=0, le=4096)
    shutdown_drain: float = Field(default=10, gt=0, le=120)
    evidence: TrackerEvidenceConfig = Field(default_factory=TrackerEvidenceConfig)
    spool: TrackerSpoolConfig = Field(default_factory=TrackerSpoolConfig)
    tls: TrackerTlsConfig = Field(default_factory=TrackerTlsConfig)

    @model_validator(mode="after")
    def require_private_mtls(self) -> Self:
        if not self.endpoint:
            raise ValueError("tracker node endpoint is required")
        missing = [
            field
            for field in ("ca", "certificate", "key", "server_name")
            if not getattr(self.tls, field)
        ]
        if missing:
            raise ValueError(
                "tracker nodes require mTLS fields: " + ", ".join(missing)
            )
        if len(self.cameras) != len(set(self.cameras)):
            raise ValueError("a tracker node cannot list a camera more than once")
        return self


class TrackerConfig(FrigateBaseModel):
    nodes: dict[str, TrackerNodeConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        owners: dict[str, str] = {}
        for node_id, node in self.nodes.items():
            if not node_id:
                raise ValueError("tracker node_id cannot be empty")
            for camera in node.cameras:
                previous = owners.get(camera)
                if previous is not None:
                    raise ValueError(
                        f"camera '{camera}' belongs to tracker nodes "
                        f"'{previous}' and '{node_id}'"
                    )
                owners[camera] = node_id
        return self

    @property
    def camera_owners(self) -> dict[str, str]:
        return {
            camera: node_id
            for node_id, node in self.nodes.items()
            for camera in node.cameras
        }

    def owner_for(self, camera: str) -> str | None:
        return self.camera_owners.get(camera)

    def is_edge_camera(self, camera: str) -> bool:
        return self.owner_for(camera) is not None
