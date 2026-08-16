"""Compile validated Frigate configuration into one immutable runtime topology."""

from __future__ import annotations

import copy
import hashlib
import json
from urllib.parse import urlsplit
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.config.recognition import RecognitionRuntimeEnum


def _write_utf8(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _service_name(node_id: str) -> str:
    return "tracker-" + node_id.lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class TrackerNodePlan:
    node_id: str
    managed: bool
    endpoint: str
    cameras: tuple[str, ...]
    service: str
    container: str
    server_name: str


@dataclass(frozen=True, slots=True)
class PlatformTopologyPlan:
    revision: str
    topology_hash: str
    recognition_external: bool
    recognition_endpoint: str
    recognition_server_name: str | None
    embedded_cameras: tuple[str, ...]
    external_cameras: tuple[str, ...]
    safety_cameras: tuple[str, ...]
    tracker_nodes: Mapping[str, TrackerNodePlan]
    camera_owners: Mapping[str, str]

    def camera_config(
        self, config: FrigateConfig, *, node_id: str | None = None
    ) -> FrigateConfig:
        """Return the already-owned camera view for main or one tracker node."""
        names = (
            self.embedded_cameras
            if node_id is None
            else self.tracker_nodes[node_id].cameras
        )
        return config.model_copy(
            update={"cameras": {name: config.cameras[name] for name in names}}
        )

    def node(self, node_id: str) -> TrackerNodePlan:
        try:
            return self.tracker_nodes[node_id]
        except KeyError as error:
            raise ValueError(f"unknown tracker node: {node_id}") from error

    def deployment_manifest(self, output_dir: Path) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "revision": self.revision,
            "topology_hash": self.topology_hash,
            "recognition": {
                "runtime": "external" if self.recognition_external else "local",
                "external": self.recognition_external,
                "endpoint": self.recognition_endpoint,
                "server_name": self.recognition_server_name,
            },
            "embedded_cameras": list(self.embedded_cameras),
            "external_cameras": list(self.external_cameras),
            "safety_cameras": list(self.safety_cameras),
            "camera_owners": dict(self.camera_owners),
            "main_config": str(output_dir / "config.main.yml"),
            "nodes": [
                {
                    "id": node.node_id,
                    "managed": node.managed,
                    "endpoint": node.endpoint,
                    "service": node.service,
                    "container": node.container,
                    "cameras": list(node.cameras),
                    "server_name": node.server_name,
                    "config_path": str(
                        output_dir / f"config.tracker.{node.node_id}.yml"
                    ),
                }
                for node in self.tracker_nodes.values()
            ],
        }


def compile_topology(config: FrigateConfig) -> PlatformTopologyPlan:
    """Resolve all runtime ownership once from validated configuration."""
    owners = config.tracker.camera_owners
    safety_cameras = {
        name
        for name, camera in config.cameras.items()
        if camera.enabled
        and camera.media_mode.value == "external"
        and "smoking" in camera.review.alerts.labels
    }
    configured_external = set(owners)
    configured_external.update(safety_cameras)
    configured_external.update(
        name
        for name, camera in config.cameras.items()
        if camera.media_mode.value == "external"
    )
    external = tuple(name for name in config.cameras if name in configured_external)
    embedded = tuple(
        name for name in config.cameras if name not in owners and name not in configured_external
    )
    nodes: dict[str, TrackerNodePlan] = {}
    for node_id in sorted(config.tracker):
        node = config.tracker[node_id]
        service = _service_name(node_id)
        nodes[node_id] = TrackerNodePlan(
            node_id=node_id,
            managed=node.managed,
            endpoint=node.endpoint,
            cameras=tuple(node.cameras),
            service=service,
            container=("edge-tracker" if node_id == "edge-local" else f"edge-tracker-{node_id}"),
            server_name=urlsplit(f"//{node.endpoint}").hostname or node.endpoint,
        )

    role = config.runtime.topology_role
    runtime_node_id = config.runtime.topology_node_id
    supplied_revision = config.runtime.topology_revision
    if role == "source" and supplied_revision:
        raise ValueError("source config cannot declare a compiled topology revision")
    if role != "source" and not supplied_revision:
        raise ValueError(f"{role} runtime view requires topology_revision")
    if role == "tracker":
        if runtime_node_id not in nodes:
            raise ValueError(
                f"tracker runtime view references unknown node: {runtime_node_id}"
            )
        expected_cameras = set(nodes[runtime_node_id].cameras)
        if set(config.cameras) != expected_cameras:
            raise ValueError(
                f"tracker runtime view for {runtime_node_id} is not camera-isolated"
            )
        if set(nodes) != {runtime_node_id}:
            raise ValueError(
                f"tracker runtime view for {runtime_node_id} contains another node"
            )
    elif runtime_node_id:
        raise ValueError(f"{role} runtime view cannot declare topology_node_id")

    safe_tracker = {
        node_id: config.tracker[node_id].model_dump(
            mode="json", exclude={"tls": {"key"}}
        )
        for node_id in sorted(config.tracker)
    }
    safe_recognition = config.recognition.model_dump(
        mode="json", exclude={"tls": {"key"}}
    )
    payload = {
        "recognition": safe_recognition,
        "tracker": safe_tracker,
        "embedded_cameras": embedded,
        "camera_owners": owners,
        "go2rtc": config.go2rtc.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "detectors": {
            name: detector.model_dump(mode="json")
            for name, detector in sorted(config.detectors.items())
        },
        "ffmpeg": config.ffmpeg.model_dump(mode="json"),
        "objects": config.objects.model_dump(mode="json"),
        "cameras": {
            name: config.cameras[name].model_dump(mode="json")
            for name in sorted(config.cameras)
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    compiled_digest = hashlib.sha256(encoded).hexdigest()
    revision = supplied_revision or compiled_digest
    return PlatformTopologyPlan(
        revision=revision,
        topology_hash=revision,
        recognition_external=(
            config.recognition.runtime is RecognitionRuntimeEnum.EXTERNAL
        ),
        recognition_endpoint=config.recognition.endpoint,
        recognition_server_name=config.recognition.tls.server_name,
        embedded_cameras=embedded,
        external_cameras=external,
        safety_cameras=tuple(name for name in config.cameras if name in safety_cameras),
        tracker_nodes=MappingProxyType(nodes),
        camera_owners=MappingProxyType(dict(owners)),
    )


def materialize_topology(
    raw_config: dict[str, Any],
    plan: PlatformTopologyPlan,
    output_dir: Path,
    *,
    managed_only: bool = True,
) -> dict[str, Any]:
    """Write launcher configs from the same compiled topology used at runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    main = copy.deepcopy(raw_config)
    main_runtime = main.setdefault("runtime", {})
    main_runtime.update(
        {
            "topology_revision": plan.revision,
            "topology_role": "main",
            "topology_node_id": "",
        }
    )
    main_path = output_dir / "config.main.yml"
    _write_utf8(
        main_path,
        yaml.safe_dump(main, sort_keys=False, allow_unicode=True),
    )
    for node in plan.tracker_nodes.values():
        edge = copy.deepcopy(raw_config)
        edge_runtime = edge.setdefault("runtime", {})
        edge_runtime.update(
            {
                "topology_revision": plan.revision,
                "topology_role": "tracker",
                "topology_node_id": node.node_id,
            }
        )
        wanted = set(node.cameras)
        edge["cameras"] = {
            name: value
            for name, value in edge.get("cameras", {}).items()
            if name in wanted
        }
        edge_streams = edge.setdefault("go2rtc", {}).setdefault("streams", {})
        edge["go2rtc"]["streams"] = {
            name: value for name, value in edge_streams.items() if name in wanted
        }
        edge["tracker"] = {node.node_id: edge["tracker"][node.node_id]}
        runtime = edge.get("runtime", {})
        for source_type in ("replay", "direct"):
            sources = runtime.get(source_type, {}).get("sources", {})
            if sources:
                runtime[source_type]["sources"] = {
                    name: value for name, value in sources.items() if name in wanted
                }
        _write_utf8(
            output_dir / f"config.tracker.{node.node_id}.yml",
            yaml.safe_dump(edge, sort_keys=False, allow_unicode=True),
        )

    manifest = plan.deployment_manifest(output_dir)
    _write_utf8(
        output_dir / "platform-topology.json",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return manifest
