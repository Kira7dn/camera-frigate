"""Read the Safety section from the canonical Frigate runtime config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class SafetyConfigError(ValueError):
    """Raised when a Safety configuration is invalid."""


@dataclass(frozen=True)
class LabelPolicy:
    enabled: bool
    threshold: float

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise SafetyConfigError("label threshold must be between 0 and 1")


@dataclass(frozen=True)
class CameraSafetyConfig:
    stream: str
    inference_fps: float
    labels: Mapping[str, LabelPolicy]
    confirm_seconds: float
    clear_seconds: float

    def __post_init__(self) -> None:
        if not self.stream.strip():
            raise SafetyConfigError("camera stream must not be empty")
        if self.inference_fps <= 0:
            raise SafetyConfigError("camera inference_fps must be greater than zero")
        if self.confirm_seconds <= 0 or self.clear_seconds <= 0:
            raise SafetyConfigError("confirm_seconds and clear_seconds must be positive")
        if not any(policy.enabled for policy in self.labels.values()):
            raise SafetyConfigError("each Safety camera needs at least one enabled label")


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    providers: tuple[str, ...]


@dataclass(frozen=True)
class SafetyConfig:
    grpc_url: str
    restream_url: str
    model: ModelConfig
    cameras: Mapping[str, CameraSafetyConfig]


SAFETY_GRPC_URL = "frigate:50052"
SAFETY_RESTREAM_URL = "rtsp://frigate:8554"
SAFETY_MODEL_CONTAINER_PATH = Path("/models/smoking/best.onnx")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SafetyConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> SafetyConfig:
    """Load Safety cameras from the canonical Frigate YAML file."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SafetyConfigError(f"unable to read Safety config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SafetyConfigError(f"invalid Safety YAML: {exc}") from exc

    root = _mapping(raw, "Frigate config")
    cameras_raw = _mapping(root.get("cameras"), "cameras")

    cameras: dict[str, CameraSafetyConfig] = {}
    for name, value in cameras_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise SafetyConfigError("camera names must be non-empty strings")
        camera = _mapping(value, f"cameras.{name}")
        if camera.get("media_mode") != "external":
            continue
        review = camera.get("review") or {}
        alerts = review.get("alerts") or {}
        labels = {
            str(label): LabelPolicy(True, 0.10)
            for label in alerts.get("labels", [])
            if str(label) == "smoking"
        }
        if not labels:
            continue
        cameras[name] = CameraSafetyConfig(
            stream=name,
            inference_fps=2,
            labels=MappingProxyType(labels),
            confirm_seconds=1,
            clear_seconds=5,
        )
    if not cameras:
        raise SafetyConfigError("canonical config must define an external smoking camera")
    model_candidates = (
        SAFETY_MODEL_CONTAINER_PATH,
        config_path.parent / "best.onnx",
        Path("assets/models/smoking/best.onnx"),
    )
    model_path = next((candidate for candidate in model_candidates if candidate.is_file()), None)
    if model_path is None:
        raise SafetyConfigError("Safety model does not exist at /models/smoking/best.onnx")

    return SafetyConfig(
        grpc_url=os.environ.get("SAFETY_GRPC_URL", SAFETY_GRPC_URL),
        restream_url=SAFETY_RESTREAM_URL,
        model=ModelConfig(model_path, ("CPUExecutionProvider",)),
        cameras=MappingProxyType(cameras),
    )


def validate_camera_keys(safety: SafetyConfig, frigate_names: set[str] | frozenset[str]) -> None:
    """Require Safety cameras to be a subset of the selected Frigate cameras."""
    missing = sorted(set(safety.cameras) - set(frigate_names))
    if missing:
        raise SafetyConfigError(f"Safety cameras are missing from Frigate config: {', '.join(missing)}")


def resolve_stream_url(config: SafetyConfig, camera: str) -> str:
    if camera not in config.cameras:
        raise SafetyConfigError(f"unknown Safety camera: {camera}")
    return f"{config.restream_url}/{config.cameras[camera].stream.lstrip('/')}"
