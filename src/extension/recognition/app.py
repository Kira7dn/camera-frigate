"""Executable entry point for the dedicated recognition service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import uuid
from pathlib import Path

from frigate.infrastructure.config import FrigateConfig
from frigate.application.recognition.core import RecognitionCore
from frigate.application.recognition.executor import AsyncRecognitionExecutor
from frigate.application.recognition.face import FacePolicy
from frigate.application.recognition.lpr import LprPolicy

from . import health_pb2
from .config_fingerprint import canonical_config_json, config_fingerprint
from .evidence import RawI420EvidenceResolver
from .grpc_server import RecognitionGrpcService, TlsServerConfig, create_grpc_server
from .models import FrigateRecognitionModel

logger = logging.getLogger(__name__)
SERVICE_NAME = "camera.recognition.v1.RecognitionService"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera Platform recognition runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bind", default="127.0.0.1:50051")
    parser.add_argument("--certificate")
    parser.add_argument("--key")
    parser.add_argument("--client-ca")
    parser.add_argument("--allow-client", action="append", default=[])
    return parser.parse_args()


def _load_config(path: str) -> FrigateConfig:
    with Path(path).open("r", encoding="utf-8") as config_file:
        return FrigateConfig.parse(config_file)


async def serve(arguments: argparse.Namespace) -> None:
    config = _load_config(arguments.config)
    model = FrigateRecognitionModel(config)
    evidence = RawI420EvidenceResolver()
    cameras = tuple(config.cameras.values())
    default_fps = cameras[0].detect.fps if cameras else 5

    def core_factory() -> RecognitionCore:
        return RecognitionCore(
            model,
            evidence,
            LprPolicy(
                detect_fps=default_fps,
                recognition_threshold=config.lpr.recognition_threshold,
                min_plate_length=config.lpr.min_plate_length,
                plate_format=config.lpr.format,
            ),
            FacePolicy(
                unknown_score=config.face_recognition.unknown_score,
                recognition_threshold=config.face_recognition.recognition_threshold,
                min_faces=config.face_recognition.min_faces,
            ),
        )

    executor = AsyncRecognitionExecutor(
        core_factory,
        uuid.uuid4().hex,
        observation_capacity=config.recognition.observation_capacity,
        control_capacity=config.recognition.control_capacity,
        outcome_capacity=config.recognition.outcome_capacity,
        shutdown_drain=config.recognition.shutdown_drain,
    )
    tls_values = (arguments.certificate, arguments.key, arguments.client_ca)
    if any(tls_values) and not all(tls_values):
        raise ValueError("certificate, key and client CA must be configured together")
    tls = None
    if all(tls_values):
        tls = TlsServerConfig(
            Path(arguments.certificate).read_bytes(),
            Path(arguments.key).read_bytes(),
            Path(arguments.client_ca).read_bytes(),
            frozenset(arguments.allow_client),
        )
    service = RecognitionGrpcService(
        executor,
        face_control=model.manage_face_library,
        mtls_required=tls is not None,
        allowed_client_identities=tls.allowed_client_identities if tls else frozenset(),
        config_hash=config_fingerprint(canonical_config_json(config)),
    )
    server, health = create_grpc_server(service, arguments.bind, tls=tls)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            loop.add_signal_handler(getattr(signal, name), stopping.set)
    await server.start()
    logger.info(
        "Recognition service listening on %s epoch=%s",
        arguments.bind,
        executor.service_epoch,
    )
    await stopping.wait()
    health.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
    health.set(SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
    await server.stop(config.recognition.shutdown_drain)
    await service.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(_arguments()))


if __name__ == "__main__":
    main()
