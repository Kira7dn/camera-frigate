from collections.abc import Sequence

from rapidfuzz.distance import JaroWinkler


def _intersection_over_union(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def is_lpr_track_discontinuity(
    previous_plate: str | None,
    current_plate: str,
    previous_box: Sequence[int] | None,
    current_box: Sequence[int] | None,
    similarity_threshold: float,
) -> bool:
    """Detect an OCR/object pairing that likely crossed to another vehicle."""
    if not previous_plate or previous_box is None or current_box is None:
        return False

    return (
        JaroWinkler.similarity(previous_plate, current_plate)
        < similarity_threshold
        and _intersection_over_union(previous_box, current_box) < 0.05
    )
