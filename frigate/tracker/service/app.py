"""Executable managed tracker node service."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import signal
from pathlib import Path

from frigate.config import FrigateConfig
from frigate.log import setup_logging
from frigate.ptz.onvif import OnvifCommandEnum
from frigate.tracker.node_runtime import TrackerNodeRuntime

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
    with Path(args.config).open(encoding="utf-8") as handle:
        config = FrigateConfig.parse(handle, install=True)
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
    )
    tls = ServerTlsConfig(
        Path(args.certificate).read_bytes(),
        Path(args.key).read_bytes(),
        Path(args.client_ca).read_bytes(),
        frozenset(args.allow_client),
    )
    runtime.start()
    server = await start_secure_server(args.bind, service, tls)
    try:
        while not stop_event.wait(0.5):
            service.degraded = runtime.degraded
            await asyncio.sleep(0)
    finally:
        service.degraded = True
        await server.stop(runtime.node_config.shutdown_drain)
        runtime.stop()
        manager.shutdown()


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
