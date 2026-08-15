"""Temporal safety decisions and Frigate Manual Event synchronization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import requests

from .config import CameraSafetyConfig
from .inference import Detection


@dataclass(frozen=True)
class HazardDecision:
    camera: str
    label: str
    active: bool
    score: float
    bbox: tuple[float, float, float, float] | None


class _State(Enum):
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"


@dataclass
class _Observation:
    state: _State = _State.IDLE
    candidate_since: float | None = None
    clear_since: float | None = None
    last_score: float = 0.0
    last_bbox: tuple[float, float, float, float] | None = None


class TemporalGate:
    def __init__(self, policies: dict[str, CameraSafetyConfig]) -> None:
        self._policies = policies
        self._states: dict[tuple[str, str], _Observation] = {}

    def observe(self, camera: str, detections: Iterable[Detection], now: float) -> list[HazardDecision]:
        policy = self._policies[camera]
        by_label = {d.label: d for d in detections if d.label in policy.labels and policy.labels[d.label].enabled}
        decisions: list[HazardDecision] = []
        for label, label_policy in policy.labels.items():
            if not label_policy.enabled:
                continue
            state = self._states.setdefault((camera, label), _Observation())
            candidate = by_label.get(label)
            if candidate is not None and candidate.score >= label_policy.threshold:
                state.last_score, state.last_bbox, state.clear_since = candidate.score, candidate.bbox, None
                if state.state is _State.IDLE:
                    state.state, state.candidate_since = _State.PENDING, now
                if state.state is _State.PENDING and state.candidate_since is not None and now - state.candidate_since >= policy.confirm_seconds:
                    state.state = _State.ACTIVE
                    decisions.append(HazardDecision(camera, label, True, state.last_score, state.last_bbox))
            elif state.state is _State.PENDING:
                state.state, state.candidate_since = _State.IDLE, None
            elif state.state is _State.ACTIVE:
                state.clear_since = now if state.clear_since is None else state.clear_since
                if now - state.clear_since >= policy.clear_seconds:
                    state.state, state.clear_since = _State.IDLE, None
                    decisions.append(HazardDecision(camera, label, False, 0.0, None))
        return decisions

    def reset(self) -> None:
        self._states.clear()


class SafetyEventError(RuntimeError):
    """Frigate Event synchronization failed."""


class FrigateEventClient:
    def __init__(self, base_url: str, session: requests.Session | None = None, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.active: dict[tuple[str, str], str] = {}

    def probe_camera(self, camera: str) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/{camera}/latest.jpg", timeout=self.timeout)
            return response.status_code == 200 and float(response.headers.get("X-Frame-Time", "0")) > 0
        except (requests.RequestException, ValueError):
            return False

    def _find_open_event(self, camera: str, label: str) -> str | None:
        response = self.session.get(
            f"{self.base_url}/api/events",
            params={"camera": camera, "label": label, "sub_label": "camera-safety", "in_progress": 1, "limit": 50},
            timeout=self.timeout,
        )
        response.raise_for_status()
        events = response.json()
        if isinstance(events, dict):
            events = events.get("events", [])
        for event in events or []:
            if event.get("sub_label") == "camera-safety" and event.get("id"):
                return str(event["id"])
        return None

    def create_event(self, decision: HazardDecision) -> str:
        key = (decision.camera, decision.label)
        if key in self.active:
            return self.active[key]
        body = {
            "sub_label": "camera-safety",
            "score": decision.score,
            "duration": None,
            "include_recording": True,
            "draw": (
                {
                    "boxes": [
                        {
                            "box": list(decision.bbox),
                            "score": decision.score,
                            "color": [0, 0, 255],
                        }
                    ]
                }
                if decision.bbox is not None
                else {}
            ),
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/events/{decision.camera}/{decision.label}/create",
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            event_id = str(response.json()["event_id"])
        except requests.Timeout as exc:
            try:
                event_id = self._find_open_event(decision.camera, decision.label)
            except (requests.RequestException, KeyError, TypeError, ValueError):
                event_id = None
            if event_id is not None:
                self.active[key] = event_id
                return event_id
            raise SafetyEventError(f"unable to reconcile timed-out Safety Event create: {exc}") from exc
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            raise SafetyEventError(f"unable to create Safety Event: {exc}") from exc
        self.active[key] = event_id
        return event_id

    def end_event(self, camera: str, label: str) -> None:
        key = (camera, label)
        event_id = self.active.get(key)
        if not event_id:
            return
        try:
            response = self.session.put(
                f"{self.base_url}/api/events/{event_id}/end",
                json={},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SafetyEventError(f"unable to end Safety Event {event_id}: {exc}") from exc
        self.active.pop(key, None)

    def apply(self, decision: HazardDecision) -> None:
        if decision.active:
            self.create_event(decision)
        else:
            self.end_event(decision.camera, decision.label)

    def reconcile(self, camera_labels: Iterable[tuple[str, str]]) -> None:
        for camera, label in camera_labels:
            try:
                response = self.session.get(
                    f"{self.base_url}/api/events",
                    params={"camera": camera, "label": label, "sub_label": "camera-safety", "in_progress": 1, "limit": 50},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                events = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise SafetyEventError(f"unable to reconcile Safety Events: {exc}") from exc
            if isinstance(events, dict):
                events = events.get("events", [])
            for event in events or []:
                if event.get("sub_label") == "camera-safety" and event.get("id"):
                    self.active[(camera, label)] = str(event["id"])
                    self.end_event(camera, label)
