"""External recognition gRPC contract tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time

import grpc
import pytest

from frigate.application.recognition import (
    EvidenceCaptureRequest,
    FacePolicy,
    LprPolicy,
    RawRecognition,
    RecognitionArtifact,
    RecognitionCore,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
    RecognitionTask,
    TrackedObservation,
    TrackKey,
)
from frigate.application.recognition.executor import AsyncRecognitionExecutor
from frigate.application.recognition.service.evidence import (
    RawI420Evidence,
    RawI420EvidenceResolver,
)
from frigate.application.recognition.service.grpc_client import RecognitionGrpcClient
from frigate.application.recognition.service.grpc_server import (
    RecognitionGrpcService,
    TlsServerConfig,
    create_grpc_server,
)
from frigate.application.recognition.service.v1 import recognition_pb2 as pb
from frigate.application.recognition.service.v1 import recognition_pb2_grpc as pb_grpc
from frigate.application.recognition.service.wire import (
    job_from_envelope,
    outcome_from_proto,
    outcome_to_proto,
)
from frigate.util.passage_trace import (
    persist_passage_evidence_bundle,
    shutdown_passage_writers,
)


class Model:
    def recognize(self, task, observation, evidence):
        assert evidence.shape == (6, 4)
        return RawRecognition("ABC123", 0.95, area=1000)


def core_factory():
    return RecognitionCore(
        Model(),
        RawI420EvidenceResolver(),
        LprPolicy(5, 0.9),
        FacePolicy(0.8, 0.9),
    )


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def observe_envelope(
    job_id: str = "job-1",
    *,
    byte_length: int = 24,
    expected_epoch: str = "service",
) -> pb.ClientEnvelope:
    return pb.ClientEnvelope(
        observe=pb.Observe(
            job_id=job_id,
            client_id="frigate",
            expected_service_epoch=expected_epoch,
            sequence=0,
            deadline_budget_ms=1000,
            observation=pb.Observation(
                task=pb.RECOGNITION_TASK_LPR,
                key=pb.TrackKey(
                    camera_id="front", stream_epoch="stream", track_id="track"
                ),
                frame_time=1.0,
                object_bbox=pb.BBox(left=0, top=0, right=4, bottom=4),
                evidence=pb.RawI420Evidence(
                    evidence_id="evidence",
                    data=bytes(range(24)),
                    shape=[6, 4],
                    dtype="uint8",
                    layout="I420",
                    byte_length=byte_length,
                    expiry_unix_ms=int(time.time() * 1000) + 1000,
                ),
                attributes_json="{}",
            ),
        )
    )


def test_configure_stream_dedupe_and_outcome():
    asyncio.run(_configure_stream_dedupe_and_outcome())


async def _configure_stream_dedupe_and_outcome():
    executor = AsyncRecognitionExecutor(core_factory, "service")
    service = RecognitionGrpcService(executor, mtls_required=False)
    port = free_port()
    server, _ = create_grpc_server(service, f"127.0.0.1:{port}")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = pb_grpc.RecognitionServiceStub(channel)
    try:
        capabilities = await stub.GetCapabilities(pb.CapabilitiesRequest(), timeout=1)
        assert capabilities.service_epoch == "service"
        config_json = json.dumps({"lpr": {"enabled": True}}, separators=(",", ":"))
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        configured = await stub.Configure(
            pb.ConfigureRequest(
                client_id="frigate",
                config_json=config_json,
                config_hash=config_hash,
            ),
            timeout=1,
        )
        assert configured.config_hash == config_hash

        stream = stub.Recognize(timeout=3)
        hello = await stream.read()
        assert hello.hello.service_epoch == "service"
        await stream.write(observe_envelope())
        receipt = await stream.read()
        outcome = await stream.read()
        assert receipt.receipt.accepted
        assert outcome.outcome.updates[0].aggregate_value == "ABC123"
        assert outcome.outcome.updates[0].evidence_id == "evidence"

        await stream.write(observe_envelope())
        duplicate_receipt = await stream.read()
        duplicate_outcome = await stream.read()
        assert duplicate_receipt.receipt.reason == "duplicate_completed"
        assert duplicate_outcome.outcome.job_id == "job-1"
        await stream.done_writing()
    finally:
        await channel.close()
        await server.stop(0)
        await service.close()


def test_invalid_evidence_and_epoch_are_typed_rejections():
    asyncio.run(_invalid_evidence_and_epoch_are_typed_rejections())


async def _invalid_evidence_and_epoch_are_typed_rejections():
    executor = AsyncRecognitionExecutor(core_factory, "service")
    service = RecognitionGrpcService(executor, mtls_required=False)
    port = free_port()
    server, _ = create_grpc_server(service, f"127.0.0.1:{port}")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = pb_grpc.RecognitionServiceStub(channel)
    try:
        stream = stub.Recognize(timeout=3)
        await stream.read()
        await stream.write(observe_envelope(byte_length=23))
        invalid = await stream.read()
        assert not invalid.receipt.accepted
        assert "byte length" in invalid.receipt.reason

        await stream.write(observe_envelope("wrong-epoch", expected_epoch="old"))
        mismatch = await stream.read()
        assert not mismatch.receipt.accepted
        assert mismatch.receipt.reason == "epoch_mismatch"
        await stream.done_writing()
    finally:
        await channel.close()
        await server.stop(0)
        await service.close()


def test_non_loopback_plaintext_fails_closed():
    executor = AsyncRecognitionExecutor(core_factory, "service")
    service = RecognitionGrpcService(executor)
    with pytest.raises(ValueError, match="requires mTLS"):
        create_grpc_server(service, "0.0.0.0:50051")
    assert executor.shutdown()


def test_capture_request_and_source_artifact_round_trip():
    envelope = observe_envelope()
    envelope.observe.observation.evidence_capture.CopyFrom(
        pb.EvidenceCaptureRequest(
            trace_id="lpr:front:track",
            evidence_id="track-deadbeef",
            run_id="runtime",
        )
    )
    job = job_from_envelope(envelope)
    assert job.observation is not None
    assert job.observation.evidence_capture == EvidenceCaptureRequest(
        "lpr:front:track", "track-deadbeef", "runtime"
    )

    image = b"source-jpeg"
    artifact = RecognitionArtifact(
        sequence=4,
        stage="plate_crop",
        pipeline="lpr",
        trace_id="lpr:front:track",
        evidence_id="track-deadbeef",
        camera="front",
        frame_time=1.0,
        track_id="track",
        metadata={"detector_box": [1, 2, 3, 4]},
        image_jpeg=image,
        image_shape=(8, 16, 3),
        image_sha256=hashlib.sha256(image).hexdigest(),
    )
    outcome = RecognitionOutcome(
        "job",
        "frigate",
        "service",
        TrackKey("front", "stream", "track"),
        0,
        RecognitionOutcomeStatus.SUCCEEDED,
        artifacts=(artifact,),
    )
    restored = outcome_from_proto(outcome_to_proto(outcome))
    assert restored.artifacts == (artifact,)


def test_host_persists_exact_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(tmp_path))
    image = b"exact-source-jpeg"
    artifact = RecognitionArtifact(
        sequence=7,
        stage="ocr_plate_input",
        pipeline="lpr",
        trace_id="lpr:front:track",
        evidence_id="track-deadbeef",
        camera="front",
        frame_time=1.0,
        track_id="track",
        image_jpeg=image,
        image_shape=(10, 20, 3),
        image_sha256=hashlib.sha256(image).hexdigest(),
    )
    assert persist_passage_evidence_bundle((artifact,))
    assert shutdown_passage_writers()
    record = json.loads(
        (tmp_path / "lpr" / "evidence.jsonl").read_text(encoding="utf-8")
    )
    stored = tmp_path / record["artifact_path"]
    assert stored.read_bytes() == image
    assert record["artifact_sha256"] == artifact.image_sha256
    sidecar = json.loads((stored.parent / "evidence.json").read_text(encoding="utf-8"))
    assert sidecar["evidence_id"] == "track-deadbeef"
    assert sidecar["artifacts"][0]["artifact_sha256"] == artifact.image_sha256


def test_mtls_requires_allowlist_and_rejects_unknown_identity():
    executor = AsyncRecognitionExecutor(core_factory, "service")
    service = RecognitionGrpcService(
        executor, allowed_client_identities=frozenset({"frigate"})
    )
    tls = TlsServerConfig(b"cert", b"key", b"ca", frozenset())
    with pytest.raises(ValueError, match="allowlist"):
        create_grpc_server(service, "0.0.0.0:50051", tls=tls)

    class Context:
        def auth_context(self):
            return {"x509_common_name": (b"unknown",)}

        async def abort(self, code, message):
            assert code is grpc.StatusCode.UNAUTHENTICATED
            raise PermissionError(message)

    with pytest.raises(PermissionError, match="not allowed"):
        asyncio.run(service._authorize(Context()))
    assert executor.shutdown()


def test_high_level_client_receives_typed_outcome():
    asyncio.run(_high_level_client_receives_typed_outcome())


async def _high_level_client_receives_typed_outcome():
    executor = AsyncRecognitionExecutor(core_factory, "service")
    service = RecognitionGrpcService(executor, mtls_required=False)
    port = free_port()
    server, _ = create_grpc_server(service, f"127.0.0.1:{port}")
    await server.start()
    client = RecognitionGrpcClient(f"127.0.0.1:{port}", "frigate")
    try:
        assert await client.connect() == "service"
        key = TrackKey("front", "stream", "track")
        evidence = RawI420Evidence(
            "evidence",
            bytes(range(24)),
            (6, 4),
            "uint8",
            "I420",
            24,
            int(time.time() * 1000) + 1000,
        )
        observation = TrackedObservation(
            RecognitionTask.LPR,
            key,
            1.0,
            (0, 0, 4, 4),
            evidence_ref=evidence,
        )
        receipt = await client.submit(
            RecognitionJob(
                "client-job",
                "frigate",
                client.service_epoch,
                key,
                0,
                RecognitionOperation.OBSERVE,
                observation,
                time.monotonic() + 1,
            )
        )
        assert receipt.accepted
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                outcome = client.get_outcome_nowait()
                break
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
        else:
            raise AssertionError("outcome was not received")
        assert outcome.status is RecognitionOutcomeStatus.SUCCEEDED
        assert outcome.updates[0].aggregate_value == "ABC123"
    finally:
        await client.close()
        await server.stop(0)
        await service.close()
