from frigate.data_processing.common.license_plate.association import (
    is_lpr_track_discontinuity,
)


def test_rejects_different_plate_on_disjoint_vehicle() -> None:
    assert is_lpr_track_discontinuity(
        "BEE3975", "C98191P", [120, 90, 420, 380], [760, 180, 1180, 690], 0.85
    )


def test_allows_same_plate_after_vehicle_moves() -> None:
    assert not is_lpr_track_discontinuity(
        "FKH9211", "FKH9211", [100, 90, 400, 380], [700, 180, 1100, 690], 0.85
    )


def test_allows_ocr_variant_on_same_vehicle_box() -> None:
    assert not is_lpr_track_discontinuity(
        "BEE3975", "BEE397S", [100, 90, 500, 500], [120, 100, 520, 510], 0.85
    )
