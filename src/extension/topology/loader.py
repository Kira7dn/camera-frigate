"""Shared UTF-8 config loading for platform topology runtimes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from frigate.infrastructure.config import FrigateConfig


@dataclass(frozen=True, slots=True)
class PlatformConfigSource:
    """Validated source config together with its original YAML mapping."""

    path: Path
    raw: dict[str, Any]
    config: FrigateConfig


class PlatformConfigLoader:
    """Load source and materialized runtime views through one config boundary."""

    @staticmethod
    def load_source(
        path: str | Path,
        *,
        host_labelmap_path: str | Path | None = None,
    ) -> PlatformConfigSource:
        """Read and validate the launcher-owned source topology config."""
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("source config must contain a YAML mapping")
        validation_raw = copy.deepcopy(raw)
        model = validation_raw.get("model")
        if (
            host_labelmap_path is not None
            and isinstance(model, dict)
            and model.get("labelmap_path") == "/labelmap/coco-80.txt"
        ):
            model["labelmap_path"] = str(host_labelmap_path)
        config = FrigateConfig.model_validate(validation_raw)
        if config.runtime.topology_role != "source":
            raise ValueError("launcher input must be a source topology config")
        return PlatformConfigSource(config_path, raw, config)

    @staticmethod
    def load_runtime(
        path: str | Path,
        *,
        expected_role: str | None = None,
        expected_node_id: str | None = None,
        install: bool = False,
    ) -> FrigateConfig:
        """Read a materialized runtime config and validate its role ownership."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            config = FrigateConfig.parse(
                config_file,
                is_json=False,
                install=install,
            )
        if expected_role is not None and config.runtime.topology_role != expected_role:
            raise ValueError(
                f"runtime config requires topology_role={expected_role}"
            )
        if (
            expected_node_id is not None
            and config.runtime.topology_node_id != expected_node_id
        ):
            raise ValueError(
                f"runtime config requires topology_node_id={expected_node_id}"
            )
        return config
