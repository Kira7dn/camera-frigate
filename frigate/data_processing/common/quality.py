"""Conservative shared quality selection for Face and LPR candidates."""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .evidence import EvidenceCandidate, EvidenceRingBuffer, FrameRef


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_detail_width_px: int
    min_detail_height_px: int
    min_laplacian_variance: float
    max_dark_fraction: float
    max_bright_fraction: float
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 100.0
    min_edge_clearance_px: int = 0


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class QualitySelector:
    """Own deterministic top-K candidates per task/camera/track generation."""

    def __init__(self, ring: EvidenceRingBuffer) -> None:
        self.ring = ring
        self._selected: dict[tuple[str, str, str, int], list[EvidenceCandidate]] = {}
        self._previous_bbox: dict[
            tuple[str, str, str, int], tuple[int, int, int, int]
        ] = {}
        self._counters: Counter[str] = Counter()
        self._reason_counts: Counter[tuple[str, str]] = Counter()
        self._lock = threading.RLock()

    @staticmethod
    def candidate_id(
        task: str,
        camera: str,
        track_id: str,
        generation: int,
        frame_ref: FrameRef,
        bbox: tuple[int, int, int, int],
    ) -> str:
        canonical = "|".join(
            (
                task,
                camera,
                track_id,
                str(generation),
                frame_ref.identity,
                ",".join(str(value) for value in bbox),
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def select(
        self,
        *,
        task: str,
        camera: str,
        track_id: str,
        generation: int,
        frame_ref: FrameRef,
        object_bbox: tuple[int, int, int, int] | None,
        detail_bbox: tuple[int, int, int, int],
        detail_frame: np.ndarray,
        thresholds: QualityThresholds,
        top_k: int = 3,
        enabled: bool = True,
        detector_score: float | None = None,
        pose_score: float | None = None,
        occlusion_score: float | None = None,
    ) -> EvidenceCandidate | None:
        key = (task, camera, track_id, generation)
        candidate_id = self.candidate_id(
            task, camera, track_id, generation, frame_ref, detail_bbox
        )
        width = max(0, detail_bbox[2] - detail_bbox[0])
        height = max(0, detail_bbox[3] - detail_bbox[1])
        gray = (
            cv2.cvtColor(detail_frame, cv2.COLOR_BGR2GRAY)
            if detail_frame.ndim == 3
            else detail_frame
        )
        laplacian_variance = (
            float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else 0.0
        )
        dark_fraction = float(np.mean(gray <= 32)) if gray.size else 1.0
        bright_fraction = float(np.mean(gray >= 223)) if gray.size else 1.0
        aspect_ratio = width / max(1, height)
        edge_clearance = min(
            detail_bbox[0],
            detail_bbox[1],
            frame_ref.width - detail_bbox[2],
            frame_ref.height - detail_bbox[3],
        )
        components: dict[str, float] = {
            "dimensions": min(
                1.0,
                0.5 * width / max(1, thresholds.min_detail_width_px)
                + 0.5 * height / max(1, thresholds.min_detail_height_px),
            ),
            "blur": min(
                1.0,
                laplacian_variance / max(1.0, thresholds.min_laplacian_variance * 2.0),
            ),
            "exposure": max(0.0, 1.0 - max(dark_fraction, bright_fraction)),
            "aspect_ratio": 1.0
            if thresholds.min_aspect_ratio <= aspect_ratio <= thresholds.max_aspect_ratio
            else 0.0,
            "edge_clearance": min(
                1.0,
                max(0, edge_clearance)
                / max(1, thresholds.min_edge_clearance_px),
            )
            if thresholds.min_edge_clearance_px
            else 1.0,
        }
        unavailable: list[str] = []
        if detector_score is None:
            unavailable.append("detector_score")
        else:
            components["detector_score"] = min(1.0, max(0.0, detector_score))
        if pose_score is None:
            unavailable.append("pose")
        else:
            components["pose"] = min(1.0, max(0.0, pose_score))
        if occlusion_score is None:
            unavailable.append("occlusion")
        else:
            components["occlusion"] = min(1.0, max(0.0, occlusion_score))

        with self._lock:
            self._reset_other_generations(task, camera, track_id, generation)
            previous = self._previous_bbox.get(key)
            if previous is None:
                unavailable.append("temporal_stability")
            else:
                components["temporal_stability"] = _box_iou(previous, detail_bbox)
            self._previous_bbox[key] = detail_bbox

            selected = self._selected.setdefault(key, [])
            if any(item.candidate_id == candidate_id for item in selected):
                self._counters["deduped"] += 1
                return None

            reasons: list[str] = []
            if width < thresholds.min_detail_width_px:
                reasons.append("detail_width_below_minimum")
            if height < thresholds.min_detail_height_px:
                reasons.append("detail_height_below_minimum")
            if laplacian_variance < thresholds.min_laplacian_variance:
                reasons.append("blur_below_minimum")
            if dark_fraction > thresholds.max_dark_fraction:
                reasons.append("underexposed")
            if bright_fraction > thresholds.max_bright_fraction:
                reasons.append("overexposed")
            if not (
                thresholds.min_aspect_ratio
                <= aspect_ratio
                <= thresholds.max_aspect_ratio
            ):
                reasons.append("aspect_ratio_out_of_range")
            if edge_clearance < thresholds.min_edge_clearance_px:
                reasons.append("detail_box_edge_clipped")
            if enabled and reasons:
                self._reject(task, reasons)
                return None

            score = float(sum(components.values()) / len(components))
            lease = self.ring.acquire(frame_ref)
            if lease is None:
                self._reject(task, ["frame_expired"])
                return None
            owner = EvidenceCandidate(
                candidate_id,
                task,
                camera,
                track_id,
                generation,
                frame_ref,
                object_bbox,
                detail_bbox,
                min(1.0, max(0.0, score)),
                components,
                (),
                tuple(unavailable),
                frame_ref.source_role,
                lease,
            )
            if not enabled:
                self._counters["accepted"] += 1
                return owner
            ranking = self._rank(owner)
            if len(selected) >= top_k:
                worst = min(selected, key=self._rank)
                if ranking <= self._rank(worst):
                    owner.release()
                    self._reject(task, ["top_k_not_selected"])
                    return None
                selected.remove(worst)
                worst.release()
                self._counters["replaced"] += 1
            selected.append(owner)
            selected.sort(key=self._rank, reverse=True)
            self._counters["accepted"] += 1
            return owner.fork()

    @staticmethod
    def _rank(candidate: EvidenceCandidate) -> tuple[float, float, str]:
        detector = candidate.quality_components.get("detector_score", -1.0)
        return (candidate.quality_score, detector, candidate.candidate_id)

    def _reject(self, task: str, reasons: list[str]) -> None:
        self._counters["rejected"] += 1
        for reason in reasons:
            self._reason_counts[(task, reason)] += 1

    def record_reject(self, task: str, reason: str) -> None:
        with self._lock:
            self._reject(task, [reason])

    def _reset_other_generations(
        self, task: str, camera: str, track_id: str, generation: int
    ) -> None:
        for key in list(self._selected):
            if key[:3] == (task, camera, track_id) and key[3] != generation:
                self._release_key(key)

    def expire(
        self,
        task: str,
        camera: str,
        track_id: str,
        generation: int | None = None,
    ) -> None:
        with self._lock:
            for key in list(self._selected):
                if key[:3] == (task, camera, track_id) and (
                    generation is None or key[3] == generation
                ):
                    self._release_key(key)

    def active_candidate_ids(
        self, task: str, camera: str, track_id: str, generation: int
    ) -> set[str]:
        with self._lock:
            return {
                candidate.candidate_id
                for candidate in self._selected.get(
                    (task, camera, track_id, generation), []
                )
            }

    def is_selected(self, candidate: EvidenceCandidate) -> bool:
        return candidate.candidate_id in self.active_candidate_ids(
            candidate.task,
            candidate.camera,
            candidate.track_id,
            candidate.generation,
        )

    def rekey(self, candidate: EvidenceCandidate, generation: int) -> EvidenceCandidate:
        """Move an owned candidate to a new tracker generation without changing frame."""
        with self._lock:
            old_key = (
                candidate.task,
                candidate.camera,
                candidate.track_id,
                candidate.generation,
            )
            new_key = (
                candidate.task,
                candidate.camera,
                candidate.track_id,
                generation,
            )
            owner = next(
                (
                    value
                    for value in self._selected.get(old_key, [])
                    if value.candidate_id == candidate.candidate_id
                ),
                None,
            )
            new_id = self.candidate_id(
                candidate.task,
                candidate.camera,
                candidate.track_id,
                generation,
                candidate.frame_ref,
                candidate.detail_bbox,
            )
            owner_lease = owner.lease if owner is not None else candidate.lease
            moved_owner = EvidenceCandidate(
                new_id,
                candidate.task,
                candidate.camera,
                candidate.track_id,
                generation,
                candidate.frame_ref,
                candidate.object_bbox,
                candidate.detail_bbox,
                candidate.quality_score,
                dict(candidate.quality_components),
                candidate.reject_reasons,
                candidate.unavailable_metrics,
                candidate.source_role,
                owner_lease,
            )
            if owner is not None:
                self._selected[old_key].remove(owner)
                if not self._selected[old_key]:
                    self._selected.pop(old_key, None)
                candidate.release()
                self._selected.setdefault(new_key, []).append(moved_owner)
                return moved_owner.fork()
            return moved_owner

    def _release_key(self, key: tuple[str, str, str, int]) -> None:
        for candidate in self._selected.pop(key, []):
            candidate.release()
        self._previous_bbox.pop(key, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            depth = sum(len(items) for items in self._selected.values())
            return {
                **dict(self._counters),
                "top_k_depth": depth,
                "reject_reasons": {
                    f"{task}:{reason}": count
                    for (task, reason), count in sorted(self._reason_counts.items())
                },
            }

    def shutdown(self) -> None:
        with self._lock:
            for key in list(self._selected):
                self._release_key(key)
