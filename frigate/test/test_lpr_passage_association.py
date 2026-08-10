from __future__ import annotations

from frigate.data_processing.common.license_plate.association import (
    LprPassageRegistry,
    associate_lpr_passages,
)


def vehicle(track_id: str, box, plate_box) -> dict:
    return {
        "id": track_id,
        "label": "car",
        "box": box,
        "current_attributes": [
            {"label": "license_plate", "box": plate_box, "score": 0.9}
        ],
    }


def plate(track_id: str, box) -> dict:
    return {"id": track_id, "label": "license_plate", "box": box}


def test_vehicle_and_plate_create_one_passage_with_full_lineage() -> None:
    admissions, rejections = associate_lpr_passages(
        [vehicle("car-1", (0, 0, 200, 100), (50, 50, 100, 75)), plate("p-1", (50, 50, 100, 75))]
    )
    assert not rejections
    assert [item.passage_id for item in admissions] == ["car-1"]
    assert admissions[0].vehicle_track_id == "car-1"
    assert admissions[0].plate_track_ids == ("p-1",)


def test_two_vehicles_are_not_merged() -> None:
    admissions, rejections = associate_lpr_passages(
        [
            vehicle("car-1", (0, 0, 200, 100), (20, 50, 70, 75)),
            vehicle("car-2", (250, 0, 450, 100), (300, 50, 350, 75)),
            plate("p-1", (20, 50, 70, 75)),
            plate("p-2", (300, 50, 350, 75)),
        ]
    )
    assert not rejections
    assert {item.passage_id for item in admissions} == {"car-1", "car-2"}


def test_ambiguous_parent_is_rejected_without_guessing() -> None:
    shared = (50, 50, 100, 75)
    admissions, rejections = associate_lpr_passages(
        [
            vehicle("car-1", (0, 0, 200, 100), shared),
            vehicle("car-2", (0, 0, 200, 100), shared),
            plate("p-1", shared),
        ]
    )
    assert len(admissions) == 2
    assert rejections[0].reason == "ambiguous_parent"
    assert rejections[0].candidate_vehicle_track_ids == ("car-1", "car-2")


def test_plate_track_churn_keeps_vehicle_passage_identity() -> None:
    car = vehicle("car-1", (0, 0, 200, 100), (50, 50, 100, 75))
    first, _ = associate_lpr_passages([car, plate("plate-a", (50, 50, 100, 75))])
    second, _ = associate_lpr_passages([car, plate("plate-b", (50, 50, 100, 75))])
    assert first[0].passage_id == second[0].passage_id == "car-1"
    assert first[0].plate_track_ids != second[0].plate_track_ids


def test_vehicle_raw_id_churn_keeps_canonical_passage() -> None:
    registry = LprPassageRegistry()
    first, _ = associate_lpr_passages(
        [vehicle("raw-a", (0, 0, 200, 100), (50, 50, 100, 75))],
        registry=registry,
        camera="cam",
        frame_time=1.0,
    )
    second, _ = associate_lpr_passages(
        [vehicle("raw-b", (5, 0, 205, 100), (55, 50, 105, 75))],
        registry=registry,
        camera="cam",
        frame_time=1.4,
    )
    assert first[0].passage_id == second[0].passage_id == "raw-a"
    assert second[0].vehicle_track_id == "raw-b"


def test_registry_does_not_merge_two_simultaneous_vehicles() -> None:
    registry = LprPassageRegistry()
    admissions, _ = associate_lpr_passages(
        [
            vehicle("car-a", (0, 0, 200, 100), (50, 50, 100, 75)),
            vehicle("car-b", (5, 0, 205, 100), (55, 50, 105, 75)),
        ],
        registry=registry,
        camera="cam",
        frame_time=1.0,
    )
    assert {item.passage_id for item in admissions} == {"car-a", "car-b"}


def test_same_raw_track_keeps_passage_during_continuous_motion() -> None:
    registry = LprPassageRegistry()
    passages = []
    for frame_time, box in (
        (1.0, (800, 0, 1100, 250)),
        (1.2, (700, 100, 1050, 400)),
        (1.4, (550, 250, 950, 600)),
    ):
        passages.append(
            registry.resolve(
                camera="cam",
                kind="vehicle",
                raw_id="raw-car",
                bbox=box,
                frame_time=frame_time,
                claimed=set(),
            )
        )

    assert passages == ["raw-car", "raw-car", "raw-car"]


def test_same_raw_track_starts_new_passage_on_impossible_reversal() -> None:
    registry = LprPassageRegistry()
    passages = []
    for frame_time, box in (
        (1.0, (800, 0, 1100, 250)),
        (1.2, (400, 500, 950, 1000)),
        (1.4, (1350, 0, 1700, 260)),
    ):
        passages.append(
            registry.resolve(
                camera="cam",
                kind="vehicle",
                raw_id="raw-car",
                bbox=box,
                frame_time=frame_time,
                claimed=set(),
            )
        )

    assert passages == ["raw-car", "raw-car", "raw-car-p2"]

