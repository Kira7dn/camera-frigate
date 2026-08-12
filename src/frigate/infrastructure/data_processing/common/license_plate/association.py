import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


def _coverage(inner: Sequence[int], outer: Sequence[int]) -> float:
    left = max(inner[0], outer[0])
    top = max(inner[1], outer[1])
    right = min(inner[2], outer[2])
    bottom = min(inner[3], outer[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    inner_area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    return intersection / inner_area if inner_area else 0.0


@dataclass(frozen=True, slots=True)
class LprPassageAdmission:
    passage_id: str
    obj_data: dict[str, Any]
    vehicle_track_id: str | None
    plate_track_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LprPassageRejection:
    reason: str
    plate_track_id: str
    candidate_vehicle_track_ids: tuple[str, ...]


@dataclass(slots=True)
class _PassageEntry:
    passage_id: str
    bbox: tuple[int, int, int, int]
    last_seen: float
    raw_ids: set[str]
    previous_bbox: tuple[int, int, int, int] | None = None


class LprPassageRegistry:
    """Bounded spatial-temporal registry separating passages from raw tracks."""

    def __init__(
        self,
        *,
        max_gap_seconds: float = 1.0,
        min_iou: float = 0.5,
        max_entries_per_camera: int = 64,
    ) -> None:
        self.max_gap_seconds = max_gap_seconds
        self.min_iou = min_iou
        self.max_entries_per_camera = max_entries_per_camera
        self._entries: dict[tuple[str, str, str], _PassageEntry] = {}
        self._aliases: dict[tuple[str, str, str], str] = {}
        self._generations: dict[tuple[str, str, str], int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _bottom_center(box: Sequence[int]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, float(box[3]))

    @classmethod
    def _has_impossible_reversal(
        cls, entry: _PassageEntry, current: Sequence[int]
    ) -> bool:
        """Reject a raw lineage jump to a new object moving the other way."""
        if entry.previous_bbox is None:
            return False

        older = cls._bottom_center(entry.previous_bbox)
        previous = cls._bottom_center(entry.bbox)
        candidate = cls._bottom_center(current)
        previous_delta = (previous[0] - older[0], previous[1] - older[1])
        candidate_delta = (candidate[0] - previous[0], candidate[1] - previous[1])
        previous_length = (previous_delta[0] ** 2 + previous_delta[1] ** 2) ** 0.5
        candidate_length = (candidate_delta[0] ** 2 + candidate_delta[1] ** 2) ** 0.5
        previous_height = max(1, entry.bbox[3] - entry.bbox[1])
        direction_dot = (
            previous_delta[0] * candidate_delta[0]
            + previous_delta[1] * candidate_delta[1]
        )
        return (
            previous_length >= previous_height * 0.10
            and candidate_length >= previous_height * 0.75
            and direction_dot < 0
        )

    def _new_passage_id(
        self, alias_key: tuple[str, str, str], raw_id: str
    ) -> str:
        generation = self._generations.get(alias_key, 0) + 1
        self._generations[alias_key] = generation
        return raw_id if generation == 1 else f"{raw_id}-p{generation}"

    def resolve(
        self,
        *,
        camera: str,
        kind: str,
        raw_id: str,
        bbox: Sequence[int],
        frame_time: float,
        claimed: set[str],
    ) -> str:
        alias_key = (camera, kind, raw_id)
        if len(bbox) != 4:
            raise ValueError("passage bbox must contain exactly four coordinates")
        box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        with self._lock:
            self._prune(camera, kind, frame_time)
            existing = self._aliases.get(alias_key)
            if existing is not None:
                entry = self._entries.get((camera, kind, existing))
                if entry is not None:
                    if not self._has_impossible_reversal(entry, box):
                        entry.previous_bbox = entry.bbox
                        entry.bbox = box
                        entry.last_seen = frame_time
                        entry.raw_ids.add(raw_id)
                        claimed.add(existing)
                        return existing
                    entry.raw_ids.discard(raw_id)
                    if not entry.raw_ids:
                        self._entries.pop((camera, kind, existing), None)
                    self._aliases.pop(alias_key, None)

            matches: list[tuple[float, _PassageEntry]] = []
            for (entry_camera, entry_kind, _), entry in self._entries.items():
                if entry_camera != camera or entry_kind != kind:
                    continue
                if entry.passage_id in claimed:
                    continue
                if frame_time - entry.last_seen > self.max_gap_seconds:
                    continue
                overlap = _intersection_over_union(box, entry.bbox)
                if overlap >= self.min_iou:
                    matches.append((overlap, entry))

            matches.sort(key=lambda item: item[0], reverse=True)
            # Do not guess when two active passages are geometrically equivalent.
            if len(matches) == 1 or (
                len(matches) > 1 and matches[0][0] - matches[1][0] >= 0.10
            ):
                entry = matches[0][1]
            else:
                passage_id = self._new_passage_id(alias_key, raw_id)
                entry = _PassageEntry(passage_id, box, frame_time, set())
                self._entries[(camera, kind, passage_id)] = entry

            entry.previous_bbox = entry.bbox
            entry.bbox = box
            entry.last_seen = frame_time
            entry.raw_ids.add(raw_id)
            self._aliases[alias_key] = entry.passage_id
            claimed.add(entry.passage_id)
            return entry.passage_id

    def _prune(self, camera: str, kind: str, frame_time: float) -> None:
        matching = [
            (key, entry)
            for key, entry in self._entries.items()
            if key[0] == camera and key[1] == kind
        ]
        expired = [
            key
            for key, entry in matching
            if frame_time - entry.last_seen > self.max_gap_seconds
        ]
        remaining = len(matching) - len(expired)
        if remaining >= self.max_entries_per_camera:
            candidates = sorted(
                (
                    (entry.last_seen, key)
                    for key, entry in matching
                    if key not in expired
                )
            )
            expired.extend(
                key
                for _, key in candidates[
                    : remaining - self.max_entries_per_camera + 1
                ]
            )
        for entry_key in expired:
            entry = self._entries.pop(entry_key, None)
            if entry is None:
                continue
            for raw_id in entry.raw_ids:
                self._aliases.pop((camera, kind, raw_id), None)

    def retire_raw(self, camera: str, raw_id: str) -> list[str] | None:
        """Retire aliases and return canonical passages whose last alias ended."""
        boundaries: list[str] = []
        found = False
        with self._lock:
            for alias_key in [
                key
                for key in self._aliases
                if key[0] == camera and key[2] == raw_id
            ]:
                found = True
                _, kind, _ = alias_key
                passage_id = self._aliases.pop(alias_key)
                entry_key = (camera, kind, passage_id)
                entry = self._entries.get(entry_key)
                if entry is None:
                    continue
                entry.raw_ids.discard(raw_id)
                if not entry.raw_ids:
                    self._entries.pop(entry_key, None)
                    boundaries.append(passage_id)
        return boundaries if found else None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._aliases.clear()
            self._generations.clear()


def associate_lpr_passages(
    objects: Sequence[dict[str, Any]],
    *,
    registry: LprPassageRegistry | None = None,
    camera: str = "",
    frame_time: float | None = None,
) -> tuple[list[LprPassageAdmission], list[LprPassageRejection]]:
    """Map detector lineage to one physical vehicle passage per frame.

    Vehicle Event IDs own passages. A standalone plate is folded into a vehicle
    only when its bbox uniquely matches that vehicle's plate attribute. Ambiguous
    matches are rejected instead of guessed.
    """
    vehicles = [
        obj
        for obj in objects
        if obj.get("label") in {"car", "motorcycle"} and obj.get("box")
    ]
    plates = [
        obj for obj in objects if obj.get("label") == "license_plate" and obj.get("box")
    ]
    plate_ids_by_vehicle: dict[str, list[str]] = {
        str(vehicle.get("id")): [] for vehicle in vehicles
    }
    standalone: list[dict[str, Any]] = []
    rejections: list[LprPassageRejection] = []

    for plate in plates:
        plate_box = plate["box"]
        matching: list[dict[str, Any]] = []
        for vehicle in vehicles:
            attributes = [
                attr
                for attr in vehicle.get("current_attributes", [])
                if attr.get("label") == "license_plate" and attr.get("box")
            ]
            if any(
                _intersection_over_union(plate_box, attr["box"]) >= 0.5
                for attr in attributes
            ) or _coverage(plate_box, vehicle["box"]) >= 0.9:
                matching.append(vehicle)
        plate_id = str(plate.get("id"))
        if len(matching) == 1:
            plate_ids_by_vehicle[str(matching[0].get("id"))].append(plate_id)
        elif len(matching) > 1:
            rejections.append(
                LprPassageRejection(
                    "ambiguous_parent",
                    plate_id,
                    tuple(sorted(str(vehicle.get("id")) for vehicle in matching)),
                )
            )
        else:
            standalone.append(plate)

    admissions: list[LprPassageAdmission] = []
    vehicle_claimed: set[str] = set()
    plate_claimed: set[str] = set()
    for vehicle in vehicles:
        vehicle_id = str(vehicle.get("id"))
        passage_id = (
            registry.resolve(
                camera=camera,
                kind="vehicle",
                raw_id=vehicle_id,
                bbox=vehicle["box"],
                frame_time=float(frame_time or vehicle.get("frame_time") or 0.0),
                claimed=vehicle_claimed,
            )
            if registry is not None
            else vehicle_id
        )
        enriched = dict(vehicle)
        enriched["_recognition_passage_id"] = passage_id
        enriched["_recognition_vehicle_track_id"] = vehicle_id
        enriched["_recognition_plate_track_ids"] = tuple(
            sorted(plate_ids_by_vehicle[vehicle_id])
        )
        admissions.append(
            LprPassageAdmission(
                passage_id,
                enriched,
                vehicle_id,
                enriched["_recognition_plate_track_ids"],
            )
        )
    for plate in standalone:
        plate_id = str(plate.get("id"))
        passage_id = (
            registry.resolve(
                camera=camera,
                kind="plate",
                raw_id=plate_id,
                bbox=plate["box"],
                frame_time=float(frame_time or plate.get("frame_time") or 0.0),
                claimed=plate_claimed,
            )
            if registry is not None
            else plate_id
        )
        enriched = dict(plate)
        enriched["_recognition_passage_id"] = passage_id
        enriched["_recognition_vehicle_track_id"] = None
        enriched["_recognition_plate_track_ids"] = (plate_id,)
        admissions.append(
            LprPassageAdmission(passage_id, enriched, None, (plate_id,))
        )
    return admissions, rejections


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
