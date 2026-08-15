"""Strict configuration for the standalone camera-safety service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    frigate_url: str
    restream_url: str
    model: ModelConfig
    cameras: Mapping[str, CameraSafetyConfig]


_ROOT_KEYS = frozenset({"frigate_url", "restream_url", "model", "cameras"})
_MODEL_KEYS = frozenset({"path", "providers"})
_CAMERA_KEYS = frozenset(
    {"stream", "inference_fps", "labels", "confirm_seconds", "clear_seconds"}
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SafetyConfigError(f"{name} must be a mapping")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SafetyConfigError(f"{name} contains unknown field(s): {', '.join(unknown)}")


def load_config(path: str | Path) -> SafetyConfig:
    """Load and validate one UTF-8 Safety YAML file."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SafetyConfigError(f"unable to read Safety config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SafetyConfigError(f"invalid Safety YAML: {exc}") from exc

    root = _mapping(raw, "Safety config")
    _strict_keys(root, _ROOT_KEYS, "Safety config")
    model = _mapping(root.get("model"), "model")
    _strict_keys(model, _MODEL_KEYS, "model")
    providers = model.get("providers")
    if not isinstance(providers, list) or not providers or not all(
        isinstance(item, str) and item.strip() for item in providers
    ):
        raise SafetyConfigError("model.providers must be a non-empty list of strings")
    cameras_raw = _mapping(root.get("cameras"), "cameras")
    if not cameras_raw:
        raise SafetyConfigError("cameras must contain at least one camera")

    cameras: dict[str, CameraSafetyConfig] = {}
    for name, value in cameras_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise SafetyConfigError("camera names must be non-empty strings")
        camera = _mapping(value, f"cameras.{name}")
        _strict_keys(camera, _CAMERA_KEYS, f"cameras.{name}")
        labels_raw = _mapping(camera.get("labels"), f"cameras.{name}.labels")
        labels: dict[str, LabelPolicy] = {}
        for label, policy_raw in labels_raw.items():
            policy = _mapping(policy_raw, f"cameras.{name}.labels.{label}")
            _strict_keys(policy, frozenset({"enabled", "threshold"}), f"labels.{label}")
            labels[str(label)] = LabelPolicy(
                enabled=bool(policy.get("enabled", True)),
                threshold=float(policy.get("threshold", 0.5)),
            )
        cameras[name] = CameraSafetyConfig(
            stream=str(camera.get("stream", name)),
            inference_fps=float(camera.get("inference_fps", 1)),
            labels=MappingProxyType(labels),
            confirm_seconds=float(camera.get("confirm_seconds", 1)),
            clear_seconds=float(camera.get("clear_seconds", 5)),
        )

    frigate_url = str(root.get("frigate_url", "")).rstrip("/")
    restream_url = str(root.get("restream_url", "")).rstrip("/")
    if not frigate_url.startswith(("http://", "https://")):
        raise SafetyConfigError("frigate_url must be an HTTP(S) URL")
    if not restream_url.startswith(("rtsp://", "rtsps://")):
        raise SafetyConfigError("restream_url must be an RTSP(S) URL")
    model_path = Path(str(model.get("path", "")))
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    if not model_path.is_file():
        raise SafetyConfigError(f"Safety model does not exist: {model_path}")

    return SafetyConfig(
        frigate_url=frigate_url,
        restream_url=restream_url,
        model=ModelConfig(model_path, tuple(providers)),
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
