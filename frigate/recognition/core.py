"""Synchronous recognition orchestration."""

from __future__ import annotations

from .contracts import RecognitionTask, RecognitionUpdate, TrackedObservation, TrackKey
from .face import FaceEngine, FacePolicy
from .lpr import LprEngine, LprPolicy
from .ports import EvidenceResolver, RecognitionModel, RecognitionObserver


class RecognitionCore:
    def __init__(
        self,
        model: RecognitionModel,
        evidence: EvidenceResolver,
        lpr_policy: LprPolicy,
        face_policy: FacePolicy,
        observer: RecognitionObserver | None = None,
    ) -> None:
        self._model = model
        self._evidence = evidence
        self._observer = observer
        self._lpr = LprEngine(lpr_policy)
        self._face = FaceEngine(face_policy)
        self._shutdown = False
        self._in_flight = 0
        self._active_tasks: dict[TrackKey, set[RecognitionTask]] = {}
        self._seen_observations: dict[TrackKey, set[tuple[RecognitionTask, float, str]]] = {}
        self._ended: set[TrackKey] = set()

    @property
    def stats(self) -> dict[str, int]:
        evidence_stats = getattr(self._evidence, "stats", None)
        pinned = int(evidence_stats().get("pinned", 0)) if evidence_stats else 0
        return {
            "sessions": self._lpr.session_count + self._face.session_count,
            "in_flight": self._in_flight,
            "evidence_pinned": pinned,
        }

    def observe(self, observation: TrackedObservation) -> tuple[RecognitionUpdate, ...]:
        if self._shutdown or observation.key in self._ended:
            return ()
        if observation.observed_in_frame is False:
            return ()
        observation_id = (
            observation.task,
            observation.frame_time,
            repr(observation.evidence_ref),
        )
        seen = self._seen_observations.setdefault(observation.key, set())
        if observation_id in seen:
            return ()
        seen.add(observation_id)
        engine = self._face if observation.task is RecognitionTask.FACE else self._lpr
        if isinstance(engine, FaceEngine) and not engine.should_attempt(observation):
            return ()

        self._in_flight += 1
        try:
            with self._evidence.resolve(observation) as evidence:
                result = self._model.recognize(observation.task, observation, evidence)
            updates = engine.observe(observation, result)
            if updates:
                self._active_tasks.setdefault(observation.key, set()).add(
                    observation.task
                )
            if self._observer is not None:
                for update in updates:
                    self._observer.on_update(update)
            return updates
        finally:
            self._in_flight -= 1

    def end_track(self, key: TrackKey, reason: str) -> None:
        if key in self._ended:
            return
        self._ended.add(key)
        active_tasks = self._active_tasks.pop(key, set())
        self._seen_observations.pop(key, None)
        self._lpr.end_track(key)
        self._face.end_track(key)
        if self._observer is not None:
            for task in sorted(active_tasks, key=lambda item: item.value):
                self._observer.on_track_end(task, key, reason)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for key in tuple(self._active_tasks):
            self.end_track(key, "shutdown")
        self._lpr.shutdown()
        self._face.shutdown()
