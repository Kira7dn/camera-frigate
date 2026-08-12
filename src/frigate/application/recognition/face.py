"""Master-compatible Face attempt cadence and weighted aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import RecognitionTask, RecognitionUpdate, TrackedObservation, TrackKey
from .ports import RawRecognition


@dataclass(frozen=True, slots=True)
class FacePolicy:
    unknown_score: float
    recognition_threshold: float
    min_faces: int = 1
    max_attempts: int = 12
    max_attempts_after_recognition: int = 6
    area_cap: int = 4000

    def __post_init__(self) -> None:
        if self.min_faces <= 0:
            raise ValueError("min_faces must be positive")


def weighted_face_vote(
    results: list[RawRecognition], policy: FacePolicy
) -> tuple[str | None, float]:
    """Aggregate Face results using the exact frozen-master algorithm."""
    counts: dict[str, int] = {}
    weighted_scores: dict[str, float] = {}
    total_weights: dict[str, float] = {}
    for result in results:
        name = result.value
        if not name or name == "unknown":
            continue
        counts[name] = counts.get(name, 0) + 1
        weight = min(result.area, policy.area_cap)
        weight *= (result.score - policy.unknown_score) * 10
        weighted_scores[name] = weighted_scores.get(name, 0.0) + result.score * weight
        total_weights[name] = total_weights.get(name, 0.0) + weight

    if not weighted_scores:
        return None, 0.0
    best_name = max(weighted_scores, key=lambda name: weighted_scores[name])
    if counts[best_name] < policy.min_faces:
        return None, 0.0
    if any(
        name != best_name and counts[best_name] == count
        for name, count in counts.items()
    ):
        return None, 0.0
    total_weight = total_weights[best_name]
    if total_weight == 0:
        return None, 0.0
    return best_name, weighted_scores[best_name] / total_weight


class FaceEngine:
    def __init__(self, policy: FacePolicy) -> None:
        self.policy = policy
        self.person_face_history: dict[TrackKey, list[RawRecognition]] = {}

    @property
    def session_count(self) -> int:
        return len(self.person_face_history)

    def should_attempt(self, observation: TrackedObservation) -> bool:
        history = self.person_face_history.get(observation.key, [])
        if observation.attributes.get("sub_label") and not history:
            return False
        if len(history) < self.policy.max_attempts_after_recognition:
            return True
        if observation.attributes.get("sub_label"):
            return False
        return len(history) < self.policy.max_attempts

    def observe(
        self, observation: TrackedObservation, result: RawRecognition | None
    ) -> tuple[RecognitionUpdate, ...]:
        if result is None:
            return ()

        history = self.person_face_history.setdefault(observation.key, [])
        history.append(result)
        name, score = self.weighted_average(history)
        publish = name is not None and score >= self.policy.recognition_threshold
        return (
            RecognitionUpdate(
                task=RecognitionTask.FACE,
                key=observation.key,
                frame_time=observation.frame_time,
                evidence_ref=observation.evidence_ref,
                raw_value=result.value,
                raw_score=result.score,
                aggregate_value=name,
                aggregate_score=score,
                object_bbox=observation.object_bbox,
                detail_bbox=result.detail_bbox or observation.detail_bbox,
                publish=publish,
                reason="master_weighted_vote" if name else "master_vote_not_ready",
                metadata={"history_size": len(history)},
            ),
        )

    def weighted_average(
        self, results: list[RawRecognition]
    ) -> tuple[str | None, float]:
        return weighted_face_vote(results, self.policy)

    def end_track(self, key: TrackKey) -> None:
        self.person_face_history.pop(key, None)

    def shutdown(self) -> None:
        self.person_face_history.clear()
