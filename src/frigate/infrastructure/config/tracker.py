"""Managed tracker-edge topology configuration."""

from __future__ import annotations

import re
from typing import Self

from pydantic import Field, RootModel, model_validator

from .base import FrigateBaseModel


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

    @model_validator(mode="after")
    def validate_tracker_node(self) -> Self:
        if not self.endpoint:
            raise ValueError("tracker node endpoint is required")
        if len(self.cameras) != len(set(self.cameras)):
            raise ValueError("a tracker node cannot list a camera more than once")
        return self


class TrackerConfig(RootModel[dict[str, TrackerNodeConfig]]):
    """External tracker nodes keyed directly by node id."""

    root: dict[str, TrackerNodeConfig] = Field(default_factory=dict)

    def __iter__(self):
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, node_id: str) -> TrackerNodeConfig:
        return self.root[node_id]

    def get(self, node_id: str) -> TrackerNodeConfig | None:
        return self.root.get(node_id)

    @model_validator(mode="after")
    def validate_ownership(self) -> Self:
        owners: dict[str, str] = {}
        for node_id, node in self.root.items():
            if re.fullmatch(r"[A-Za-z0-9_-]+", node_id) is None:
                raise ValueError(
                    "tracker node_id must contain only letters, digits, underscore or dash"
                )
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
            for node_id, node in self.root.items()
            for camera in node.cameras
        }
