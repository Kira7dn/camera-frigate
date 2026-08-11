"""Master-compatible LPR aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz.distance import JaroWinkler

from .contracts import RecognitionTask, RecognitionUpdate, TrackedObservation, TrackKey
from .ports import RawRecognition


@dataclass(frozen=True, slots=True)
class LprPolicy:
    detect_fps: int
    recognition_threshold: float
    cluster_threshold: float = 0.85
    min_plate_length: int = 0
    plate_format: str | None = None

    def __post_init__(self) -> None:
        if self.detect_fps <= 0:
            raise ValueError("detect_fps must be positive")


def select_lpr_representative(
    variants: list[RawRecognition], cluster_threshold: float = 0.85
) -> tuple[RawRecognition, list[list[RawRecognition]]]:
    """Select a representative using the exact frozen-master algorithm."""
    clusters: list[list[RawRecognition]] = []
    for variant in variants:
        for cluster in clusters:
            similarities = [
                JaroWinkler.similarity(variant.value or "", item.value or "")
                for item in cluster
            ]
            if sum(similarities) / len(similarities) >= cluster_threshold:
                cluster.append(variant)
                break
        else:
            clusters.append([variant])

    best_cluster = max(
        clusters, key=lambda cluster: (len(cluster), max(v.score for v in cluster))
    )
    return max(best_cluster, key=lambda item: item.score), clusters


class LprEngine:
    def __init__(self, policy: LprPolicy) -> None:
        self.policy = policy
        self._history: dict[TrackKey, list[RawRecognition]] = {}

    @property
    def session_count(self) -> int:
        return len(self._history)

    def observe(
        self, observation: TrackedObservation, result: RawRecognition | None
    ) -> tuple[RecognitionUpdate, ...]:
        if result is None or not result.value:
            return ()
        if result.score < self.policy.recognition_threshold:
            return ()

        history = self._history.setdefault(observation.key, [])
        history.append(result)
        window = self.policy.detect_fps * 5
        if len(history) > window:
            del history[:-window]

        representative, clusters = select_lpr_representative(
            history, self.policy.cluster_threshold
        )
        publish = True
        reason = "master_variant_representative"
        if len(representative.value or "") < self.policy.min_plate_length:
            publish = False
            reason = "below_min_plate_length"
        elif self.policy.plate_format and not re.fullmatch(
            self.policy.plate_format, representative.value or ""
        ):
            publish = False
            reason = "format_mismatch"
        metadata: dict[str, Any] = {
            "history_size": len(history),
            "cluster_sizes": tuple(len(cluster) for cluster in clusters),
            **dict(result.metadata),
        }
        return (
            RecognitionUpdate(
                task=RecognitionTask.LPR,
                key=observation.key,
                frame_time=observation.frame_time,
                evidence_ref=observation.evidence_ref,
                raw_value=result.value,
                raw_score=result.score,
                aggregate_value=representative.value,
                aggregate_score=representative.score,
                object_bbox=observation.object_bbox,
                detail_bbox=result.detail_bbox or observation.detail_bbox,
                publish=publish,
                reason=reason,
                metadata=metadata,
            ),
        )

    def end_track(self, key: TrackKey) -> None:
        self._history.pop(key, None)

    def shutdown(self) -> None:
        self._history.clear()
