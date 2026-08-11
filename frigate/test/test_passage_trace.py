import json

from frigate.util import passage_trace as passage_trace_module
from frigate.util.passage_trace import (
    passage_evidence,
    passage_evidence_should_capture,
    passage_trace,
)


def test_capture_cutoff_rejects_later_trace_and_evidence(tmp_path, monkeypatch) -> None:
    trace = tmp_path / "trace.jsonl"
    evidence = tmp_path / "evidence"
    cutoff = tmp_path / "cutoff"
    cutoff.write_text("10.0\n", encoding="utf-8")
    monkeypatch.setenv("PASSAGE_TRACE_PATH", str(trace))
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("PASSAGE_CAPTURE_CUTOFF_PATH", str(cutoff))

    passage_trace("before", camera="cam", frame_time=10.0)
    passage_trace("after", camera="cam", frame_time=10.1)
    passage_evidence(
        "invocation",
        evidence_id="before",
        camera="cam",
        frame_time=10.0,
        track_id="car-1",
    )
    passage_evidence(
        "invocation",
        evidence_id="after",
        camera="cam",
        frame_time=10.1,
        track_id="car-2",
    )

    trace_records = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()
    ]
    evidence_records = [
        json.loads(line)
        for line in (evidence / "lpr" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in trace_records] == ["before"]
    assert [record["evidence_id"] for record in evidence_records] == ["before"]


def test_capture_start_gate_blocks_warmup_and_tags_active_run(
    tmp_path, monkeypatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    evidence = tmp_path / "evidence"
    start = tmp_path / "start"
    monkeypatch.setenv("PASSAGE_TRACE_PATH", str(trace))
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("PASSAGE_CAPTURE_START_PATH", str(start))
    monkeypatch.setenv("PASSAGE_RUN_ID", "run-1")

    passage_trace("warmup", camera="cam", frame_time=9.0)
    passage_evidence(
        "warmup",
        evidence_id="warmup",
        camera="cam",
        frame_time=9.0,
        track_id="car-0",
    )
    assert not trace.exists()
    assert not evidence.exists()

    start.write_text("10.0\n", encoding="utf-8")
    passage_trace("too_early", camera="cam", frame_time=9.9)
    passage_trace("active", camera="cam", frame_time=10.0)
    record = passage_evidence(
        "active",
        evidence_id="active",
        camera="cam",
        frame_time=10.0,
        track_id="car-1",
    )

    trace_record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert trace_record["stage"] == "active"
    assert trace_record["run_id"] == "run-1"
    assert record is not None
    assert record["run_id"] == "run-1"


def test_evidence_sampling_uses_exact_candidate_interval(monkeypatch) -> None:
    passage_trace_module._EVIDENCE_LAST_CAPTURE.clear()
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", "enabled")
    monkeypatch.setenv("PASSAGE_EVIDENCE_MIN_INTERVAL_SECONDS", "0.4")

    assert passage_evidence_should_capture("cam", "car-1", 10.0)
    assert not passage_evidence_should_capture("cam", "car-1", 10.399)
    assert passage_evidence_should_capture("cam", "car-1", 10.4)
    assert passage_evidence_should_capture("cam", "car-2", 10.1)


def test_trace_and_evidence_have_producer_owned_trace_id(tmp_path, monkeypatch) -> None:
    trace = tmp_path / "trace.jsonl"
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("PASSAGE_TRACE_PATH", str(trace))
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(evidence))
    monkeypatch.delenv("PASSAGE_CAPTURE_CUTOFF_PATH", raising=False)

    passage_trace(
        "recognition_attempt",
        camera="car_camera",
        frame_time=12.5,
        track_id="vehicle-7",
        generation=2,
        task="lpr",
    )
    record = passage_evidence(
        "plate_crop",
        evidence_id="vehicle-7-shot",
        camera="car_camera",
        frame_time=12.5,
        track_id="vehicle-7",
        trace_id="lpr:car_camera:passage-7",
        pipeline="lpr",
        image=None,
    )

    trace_record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
    assert trace_record["trace_id"] == "lpr:car_camera:vehicle-7"
    assert trace_record["source_pts"] == 12.5
    assert record is not None
    assert record["trace_id"] == "lpr:car_camera:passage-7"
    assert record["pipeline"] == "lpr"
    assert record["source_pts"] == 12.5


def test_evidence_budget_counts_encoded_bytes_not_raw_frames(
    tmp_path, monkeypatch
) -> None:
    class LargeRawImage:
        nbytes = 1024
        shape = (16, 16, 3)

        def copy(self):
            return self

    evidence = tmp_path / "evidence"
    monkeypatch.setenv("PASSAGE_EVIDENCE_DIR", str(evidence))
    monkeypatch.setenv("PASSAGE_EVIDENCE_MAX_BYTES", "5")
    monkeypatch.delenv("PASSAGE_CAPTURE_CUTOFF_PATH", raising=False)
    monkeypatch.setattr(passage_trace_module, "_encode_jpeg", lambda _image: b"1234")

    for index in range(2):
        passage_evidence(
            "runtime_frame_object_box",
            evidence_id=f"track-{index}-shot",
            camera="car_camera",
            frame_time=10.0 + index,
            track_id=f"track-{index}",
            image=LargeRawImage(),
        )
    assert passage_trace_module.shutdown_passage_writers(1)

    records = [
        json.loads(line)
        for line in (evidence / "lpr" / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["artifact_bytes"] == 4
    assert records[1]["artifact_rejected"] == "byte_limit"
    assert (evidence / records[0]["artifact_path"]).read_bytes() == b"1234"
