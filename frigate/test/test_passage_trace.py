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
        for line in (evidence / "evidence.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in trace_records] == ["before"]
    assert [record["evidence_id"] for record in evidence_records] == ["before"]


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
    assert record is not None
    assert record["trace_id"] == "lpr:car_camera:passage-7"
    assert record["pipeline"] == "lpr"
