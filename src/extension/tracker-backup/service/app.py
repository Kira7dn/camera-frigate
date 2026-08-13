"""Executable managed tracker node service."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import signal
import time
from pathlib import Path
from typing import cast

from extension.topology.loader import PlatformConfigLoader
from extension.tracker.config.fingerprint import tracker_config_fingerprint
from extension.tracker.runtime.node import TrackerNodeRuntime
from frigate.domain.ptz.onvif import OnvifCommandEnum
from frigate.log import setup_logging

from .grpc_server import ServerTlsConfig, TrackerGrpcService, start_secure_server
from .v1 import tracker_pb2 as pb


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="camera-tracker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--bind", default="0.0.0.0:50052")
    parser.add_argument("--spool-dir", default="/var/lib/camera-tracker/spool")
    parser.add_argument("--media-dir", default="/media/tracker")
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--client-ca", required=True)
    parser.add_argument("--allow-client", action="append", required=True)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    config = await asyncio.to_thread(
        PlatformConfigLoader.load_runtime,
        args.config,
        expected_role="tracker",
        expected_node_id=args.node_id,
        install=True,
    )
    manager = mp.Manager()
    setup_logging(manager)
    stop_event = mp.Event()
    runtime = TrackerNodeRuntime(
        config,
        args.node_id,
        manager,
        stop_event,
        spool_dir=args.spool_dir,
        media_dir=args.media_dir,
    )
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_name, stop_event.set)

    def control(
        camera: str, operation: int, payload: dict[str, object]
    ) -> tuple[bool, str, dict[str, object] | None]:
        if camera not in runtime.config.cameras:
            return False, "camera_owner_mismatch", None
        if operation == pb.CONTROL_OPERATION_MANUAL_PTZ:
            try:
                command = OnvifCommandEnum(str(payload["command"]))
            except (KeyError, ValueError):
                return False, "invalid_ptz_command", None
            runtime.onvif.handle_command(camera, command, str(payload.get("param", "")))
            return True, "accepted", None
        if operation == pb.CONTROL_OPERATION_PRESET:
            preset = str(payload.get("preset", ""))
            if not preset:
                return False, "preset_required", None
            runtime.onvif.handle_command(camera, OnvifCommandEnum.preset, preset)
            return True, "accepted", None
        if operation == pb.CONTROL_OPERATION_CALIBRATE:
            patch = runtime.pop_config_patch(camera)
            if payload.get("status"):
                return (
                    patch is not None,
                    "calibration_complete" if patch is not None else "calibration_pending",
                    patch,
                )
            samples = payload.get("samples")
            if isinstance(samples, list):
                try:
                    calibration_samples: list[dict[str, float]] = []
                    for sample in samples:
                        if not isinstance(sample, dict):
                            raise TypeError("calibration samples must be objects")
                        sample_data = cast(dict[str, object], sample)
                        calibration_samples.append(
                            {
                                "pan": float(str(sample_data["pan"])),
                                "tilt": float(str(sample_data["tilt"])),
                                "duration": float(str(sample_data["duration"])),
                            }
                        )
                    accepted = runtime.ptz.ptz_autotracker.calibrate_from_samples(
                        camera,
                        calibration_samples,
                        min_zoom=float(str(payload.get("min_zoom", 0))),
                        max_zoom=float(str(payload.get("max_zoom", 1))),
                        zoom_time=float(str(payload.get("zoom_time", 0))),
                    )
                except (KeyError, TypeError, ValueError):
                    return False, "invalid_calibration_samples", None
                patch = runtime.pop_config_patch(camera)
                return (
                    accepted and patch is not None,
                    "calibration_complete" if accepted else "calibration_failed",
                    patch,
                )
            runtime.ptz.ptz_autotracker.request_calibration(camera)
            return True, "calibration_started", None
        if operation in (
            pb.CONTROL_OPERATION_ENABLE,
            pb.CONTROL_OPERATION_DISABLE,
        ):
            enabled = operation == pb.CONTROL_OPERATION_ENABLE
            if payload.get("target") == "autotracking":
                runtime.config.cameras[camera].onvif.autotracking.enabled = enabled
                runtime.ptz_metrics[camera].autotracker_enabled.value = enabled
                runtime.ptz_metrics[camera].start_time.value = 0
            else:
                runtime.config.cameras[camera].enabled = enabled
            return True, "accepted", None
        if operation == pb.CONTROL_OPERATION_TOPOLOGY_DRAIN:
            stop_event.set()
            return True, "draining", None
        if operation == pb.CONTROL_OPERATION_MEDIA_DELETE:
            return runtime.media.delete(str(payload.get("media_id", ""))), "accepted", None
        if operation == pb.CONTROL_OPERATION_MEDIA_RETAIN:
            accepted = runtime.media.retain(str(payload.get("media_id", "")))
            return accepted, "accepted" if accepted else "media_unavailable", None
        return False, "unsupported_control", None

    service = TrackerGrpcService(
        node_id=runtime.node_id,
        node_epoch=runtime.node_epoch,
        journal=runtime.journal,
        evidence=runtime.evidence,
        media=runtime.media,
        detector_runtimes=tuple(runtime.config.detectors),
        control=control,
        allowed_client_identities=frozenset(args.allow_client),
        camera_health=runtime.camera_health,
        terminal_state=runtime.terminal_state,
        config_hash=tracker_config_fingerprint(config, args.node_id),
    )
    runtime.set_publisher(service.publish_update)
    tls = ServerTlsConfig(
        Path(args.certificate).read_bytes(),
        Path(args.key).read_bytes(),
        Path(args.client_ca).read_bytes(),
        frozenset(args.allow_client),
    )
    runtime.start()
    server = await start_secure_server(args.bind, service, tls)
    try:
        while not stop_event.is_set():
            service.degraded = runtime.degraded
            # multiprocessing.Event.wait() is blocking and would starve the
            # aio gRPC server that shares this event loop.
            await asyncio.sleep(0.5)
    finally:
        service.degraded = True
        runtime.drain_active()
        drain_deadline = time.monotonic() + runtime.node_config.shutdown_drain
        while runtime.journal.pending_count and time.monotonic() < drain_deadline:
            await asyncio.sleep(0.1)
        await server.stop(runtime.node_config.shutdown_drain)
        runtime.stop()
        manager.shutdown()


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
