"""Rule-driven native notification policy and provider fan-out."""

import datetime
import hashlib
import json
import logging
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

import cv2
from titlecase import titlecase

from frigate.infrastructure.comms.base_communicator import Communicator
from frigate.infrastructure.comms.config_updater import ConfigSubscriber
from frigate.infrastructure.config import FrigateConfig
from frigate.infrastructure.config.auth import AuthConfig
from frigate.infrastructure.config.camera.notification import (
    NotificationDestinationsConfig,
    NotificationRuleConfig,
)
from frigate.application.events.canonical import CanonicalMediaStore, EventAggregator
from frigate.models import (
    EdgeMediaManifest,
    Event,
    NotificationIntent,
    NotificationRuleState,
)

from .envelope import NotificationEnvelope
from .metrics import increment
from .social import SocialClient
from .webpush import WebPushProvider

logger = logging.getLogger(__name__)
CAMERA_STATUS_TOPIC = re.compile(r"^([^/]+)/status/detect$")
OFFLINE_DEBOUNCE_SECONDS = 30
SEEN_RETENTION_SECONDS = 86400


def normalize_plate(value: Any) -> str | None:
    if isinstance(value, list | tuple):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    plate = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    if not 5 <= len(plate) <= 12:
        return None
    if not any(char.isdigit() for char in plate) or not any(
        char.isalpha() for char in plate
    ):
        return None
    return plate


class NotificationClient(Communicator):
    """Normalize Frigate events, match rules once, then fan out."""

    def __init__(self, config: FrigateConfig, stop_event: MpEvent) -> None:
        self.config = config
        self.stop_event = stop_event
        self.webpush = WebPushProvider(config, stop_event)
        self.social = SocialClient(config, stop_event)
        self.suspended_cameras = {camera: 0 for camera in config.cameras}
        self.suspension_broadcaster: Callable[[str, Any, bool], None] | None = None
        self.global_config_subscriber = ConfigSubscriber("config/")
        self._last_delivery: dict[tuple[str, str, str, str], float] = {}
        self._seen: dict[tuple[str, str, str], float] = {}
        self._alert_plates: dict[str, str] = {}
        self._lpr_updates: dict[str, dict[str, Any]] = {}
        self._camera_status: dict[str, str] = {}
        self._offline_since: dict[str, float] = {}
        self._offline_notified: set[str] = set()
        self._aggregator = EventAggregator(
            CanonicalMediaStore(
                max_storage_mb=config.notifications.media.max_storage_mb
            ),
            config.notifications.pipeline.finalization_timeout,
        )
        self._load_rule_state()
        self._maintenance_thread = threading.Thread(
            target=self._process_runtime_state, daemon=True
        )
        self._maintenance_thread.start()

    def subscribe(self, receiver: Callable) -> None:
        pass

    def set_suspension_broadcaster(
        self, broadcaster: Callable[[str, Any, bool], None]
    ) -> None:
        self.suspension_broadcaster = broadcaster

    def suspend_notifications(self, camera: str, duration: int) -> None:
        self.suspended_cameras[camera] = int(
            (
                datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(minutes=duration)
            ).timestamp()
        )

    def unsuspend_notifications(self, camera: str) -> None:
        self.suspended_cameras[camera] = 0

    def is_camera_suspended(self, camera: str) -> bool:
        suspended_until = self.suspended_cameras.get(camera, 0)
        if suspended_until and suspended_until <= self._now():
            self.unsuspend_notifications(camera)
            self._broadcast_suspension(camera)
            return False
        return bool(suspended_until)

    def _broadcast_suspension(self, camera: str) -> None:
        if self.suspension_broadcaster:
            self.suspension_broadcaster(
                f"{camera}/notifications/suspended",
                str(self.suspended_cameras.get(camera, 0)),
                True,
            )

    @staticmethod
    def _now() -> float:
        return datetime.datetime.now(datetime.UTC).timestamp()

    def _load_rule_state(self) -> None:
        try:
            for state in NotificationRuleState.select():
                key = (state.rule_id, state.camera, state.channel, state.recipient_id)
                self._last_delivery[key] = state.last_sent.timestamp()
                if state.last_source_type and state.last_source_id:
                    self._seen[
                        (state.rule_id, state.last_source_type, state.last_source_id)
                    ] = state.last_sent.timestamp()
        except Exception:
            logger.debug("Notification rule state is not available yet", exc_info=True)

    def _persist_rule_state(
        self,
        key: tuple[str, str, str, str],
        envelope: NotificationEnvelope,
        sent_at: float,
    ) -> None:
        try:
            NotificationRuleState.insert(
                rule_id=key[0],
                camera=key[1],
                channel=key[2],
                recipient_id=key[3],
                last_source_type=envelope.source_type,
                last_source_id=envelope.source_id,
                last_sent=datetime.datetime.fromtimestamp(sent_at, datetime.UTC),
            ).on_conflict(
                conflict_target=[
                    NotificationRuleState.rule_id,
                    NotificationRuleState.camera,
                    NotificationRuleState.channel,
                    NotificationRuleState.recipient_id,
                ],
                update={
                    NotificationRuleState.last_source_type: envelope.source_type,
                    NotificationRuleState.last_source_id: envelope.source_id,
                    NotificationRuleState.last_sent: datetime.datetime.fromtimestamp(
                        sent_at, datetime.UTC
                    ),
                },
            ).execute()
        except Exception:
            logger.warning("Unable to persist notification rule state", exc_info=True)

    def _process_runtime_state(self) -> None:
        while not self.stop_event.wait(1):
            now = self._now()
            for camera in tuple(self.suspended_cameras):
                self.is_camera_suspended(camera)
            for camera, offline_since in tuple(self._offline_since.items()):
                if (
                    camera not in self._offline_notified
                    and now - offline_since >= OFFLINE_DEBOUNCE_SECONDS
                    and self._camera_status.get(camera) == "offline"
                ):
                    self._offline_notified.add(camera)
                    self._route(
                        "camera_offline",
                        self._camera_status_envelope(camera, "offline", offline_since),
                    )
            cutoff = now - SEEN_RETENTION_SECONDS
            self._seen = {key: at for key, at in self._seen.items() if at >= cutoff}
            for event_id in self._aggregator.finalize_due():
                self._route_finalized_event(event_id)

    def _refresh_config(self) -> None:
        changed = False
        while True:
            topic, payload = self.global_config_subscriber.check_for_update()
            if topic is None:
                break
            if topic == "config/notifications" and payload:
                self.config.notifications = payload
                changed = True
            elif topic == "config/auth" and isinstance(payload, AuthConfig):
                self.config.auth = payload
                self.webpush.refresh_authorization()
        for camera in self.config.cameras:
            self.suspended_cameras.setdefault(camera, 0)
        if changed:
            self.social.cancel_disabled()

    def _rule_matches(
        self, rule: NotificationRuleConfig, envelope: NotificationEnvelope
    ) -> bool:
        filters = rule.filters
        if filters.cameras and envelope.camera not in filters.cameras:
            return False
        labels = set(envelope.genai.get("_labels", []))
        if envelope.object_label:
            labels.add(envelope.object_label)
        if filters.labels and not labels.intersection(filters.labels):
            return False
        zones = set(envelope.genai.get("_zones", []))
        if filters.zones and not zones.intersection(filters.zones):
            return False
        if filters.identities:
            identity = envelope.sub_label or "unknown"
            if "*" not in filters.identities and identity not in filters.identities:
                return False
            if identity == "unknown" and "unknown" not in filters.identities:
                return False
        if (
            filters.trigger_names
            and envelope.genai.get("trigger_name") not in filters.trigger_names
        ):
            return False
        return not (
            filters.conditions
            and envelope.genai.get("condition") not in filters.conditions
        )

    def _available_destinations(
        self, rule: NotificationRuleConfig, envelope: NotificationEnvelope
    ) -> tuple[NotificationDestinationsConfig, list[tuple[str, str, str, str]]]:
        now = self._now()
        selected = rule.destinations
        keys: list[tuple[str, str, str, str]] = []

        def allowed(channel: str, recipient: str) -> bool:
            key = (rule.id, envelope.camera or "*", channel, recipient)
            if now - self._last_delivery.get(key, 0) < rule.cooldown:
                increment(channel, "skipped_cooldown")
                return False
            keys.append(key)
            return True

        webpush = (
            selected.webpush
            and self.config.notifications.channels.webpush.enabled
            and self.webpush.configured
            and (not envelope.media_artifact_id or bool(self.config.notifications.public_base_url))
            and (not envelope.media_artifact_id or self.social.public_media_ready())
            and allowed("webpush", "registered_devices")
        )
        telegram = [
            r
            for r in selected.telegram
            if self.social.recipient_enabled("telegram", r, envelope.camera, rule.id)
            and allowed("telegram", r)
        ]
        zalo = [
            r
            for r in selected.zalo
            if self.social.recipient_enabled("zalo", r, envelope.camera, rule.id)
            and bool(self.config.notifications.public_base_url)
            and allowed("zalo", r)
        ]
        return (
            NotificationDestinationsConfig(
                webpush=webpush, telegram=telegram, zalo=zalo
            ),
            keys,
        )

    def _route(self, event: str, envelope: NotificationEnvelope) -> list[str]:
        if not self.config.notifications.enabled:
            increment("all", "skipped_disabled")
            return []
        if envelope.camera and self.is_camera_suspended(envelope.camera):
            increment("all", "skipped_suspended")
            return []
        envelope = self._canonical_envelope(envelope)
        if envelope.snapshot_ref and not envelope.media_artifact_id:
            # Never enqueue a provisional or dynamically rendered image.
            increment("all", "waiting_for_finalization")
            return []
        if self.config.notifications.pipeline.shadow_mode:
            increment("all", "shadow_committed")
            return []
        event_types = {event}
        if envelope.media_artifact_id:
            event_types.update(("alert", "object_detected"))
            if envelope.lpr_plate:
                event_types.add("license_plate")
            if envelope.sub_label:
                event_types.add("face_recognized")
        matching_rules = [
            rule
            for rule in self.config.notifications.rules
            if rule.enabled
            and rule.event in event_types
            and self._rule_matches(rule, envelope)
        ]
        destinations = NotificationDestinationsConfig()
        cooldown_keys: list[tuple[str, str, str, str]] = []
        dedupe_keys: list[tuple[str, str, str]] = []
        for rule in matching_rules:
            dedupe_key = (rule.id, envelope.source_type, envelope.source_id)
            if dedupe_key in self._seen:
                increment("all", "deduplicated")
                continue
            available, keys = self._available_destinations(rule, envelope)
            destinations.webpush = destinations.webpush or available.webpush
            destinations.telegram = sorted(
                set(destinations.telegram).union(available.telegram)
            )
            destinations.zalo = sorted(set(destinations.zalo).union(available.zalo))
            cooldown_keys.extend(keys)
            dedupe_keys.append(dedupe_key)
        if not dedupe_keys:
            return []
        ruled_envelope = replace(envelope, rule_id="event_revision")
        self._persist_intents(ruled_envelope, destinations)
        delivered = False
        if destinations.webpush and self.config.notifications.channels.webpush.enabled:
            delivered = self.webpush.deliver(ruled_envelope) > 0
        deliveries = self.social.enqueue(ruled_envelope, destinations)
        delivered = bool(deliveries) or delivered
        if delivered:
            now = self._now()
            for dedupe_key in dedupe_keys:
                self._seen[dedupe_key] = now
            for key in set(cooldown_keys):
                self._last_delivery[key] = now
                self._persist_rule_state(key, ruled_envelope, now)
        return deliveries

    def _persist_intents(
        self,
        envelope: NotificationEnvelope,
        destinations: NotificationDestinationsConfig,
    ) -> None:
        if not envelope.media_artifact_id or envelope.revision is None:
            return
        recipients = []
        if destinations.webpush:
            recipients.append(("webpush", "registered_devices"))
        recipients.extend(("telegram", value) for value in destinations.telegram)
        recipients.extend(("zalo", value) for value in destinations.zalo)
        for channel, recipient in recipients:
            key = f"{envelope.source_id}:{envelope.revision}:{channel}:{recipient}"
            intent_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
            facts = {**envelope.facts, "intent_id": intent_id}
            NotificationIntent.insert(
                id=intent_id,
                event_id=str(envelope.facts.get("event_id") or envelope.source_id),
                revision=envelope.revision,
                channel=channel,
                recipient_id=recipient,
                facts=facts,
                caption=f"{envelope.title}\n{envelope.message}",
                actions=[{"url": envelope.direct_url}] if envelope.direct_url else [],
                media_artifact_id=envelope.media_artifact_id,
                status="pending",
                created_at=datetime.datetime.now(datetime.UTC),
            ).on_conflict_ignore().execute()

    def _canonical_envelope(self, envelope: NotificationEnvelope) -> NotificationEnvelope:
        """Pin event presentation and media before rule evaluation or retries."""
        event_id = envelope.snapshot_ref
        if not event_id:
            return envelope
        event = Event.get_or_none(Event.id == event_id)
        if event is None or event.end_time is None or event.state != "FINALIZED":
            return envelope
        artifact = self._aggregator.media.get(event.canonical_artifact_id)
        edge_artifact = None
        if artifact is None:
            edge_artifact = (
                EdgeMediaManifest.select()
                .where(
                    (EdgeMediaManifest.event_id == event.id)
                    & (EdgeMediaManifest.media_type == "snapshot_jpg")
                )
                .order_by(EdgeMediaManifest.end_time.desc())
                .first()
            )
        if artifact is None and edge_artifact is None:
            return envelope
        identity = event.canonical_sub_label or event.sub_label
        is_face = event.label == "person" and bool(identity)
        label = event.display_label or identity or envelope.lpr_plate or event.label
        if is_face:
            confidence = (event.data or {}).get("sub_label_score") or (
                event.data or {}
            ).get("face_snapshot_score")
        else:
            confidence = envelope.lpr_score
        confidence_text = (
            f" · Tin cậy {round(confidence * 100):d}%"
            if confidence is not None
            else ""
        )
        title = f"{'👤' if is_face else '🚗'} {label} · {event.camera}"
        message = (
            f"Đã nhận diện khuôn mặt{confidence_text}"
            if is_face
            else f"Xe đã kết thúc lượt qua{confidence_text}"
        )
        return replace(
            envelope,
            title=title,
            message=message,
            snapshot_ref=None,
            media_artifact_id=(artifact.id if artifact else edge_artifact.media_id),
            revision=(artifact.revision if artifact else event.revision),
            direct_url=(
                f"{str(self.config.notifications.public_base_url).rstrip('/')}"
                f"/explore?event_id={event.id}&revision="
                f"{artifact.revision if artifact else event.revision}"
                if self.config.notifications.public_base_url
                else ""
            ),
            facts={
                **envelope.facts,
                "event_id": event.id,
                "revision": artifact.revision,
                "display_label": label,
                "identity": identity if is_face else None,
                "license_plate": envelope.lpr_plate,
                "confidence": confidence,
                "artifact_id": artifact.id if artifact else edge_artifact.media_id,
            },
        )

    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        self._refresh_config()
        try:
            decoded = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            decoded = payload

        status_match = CAMERA_STATUS_TOPIC.match(topic)
        if status_match and isinstance(decoded, str):
            self._handle_camera_status(status_match.group(1), decoded)
            return

        result: tuple[str, NotificationEnvelope] | None = None
        if topic == "reviews" and isinstance(decoded, dict):
            envelope = self._review_envelope(decoded)
            result = ("alert", envelope) if envelope else None
        elif topic == "events" and isinstance(decoded, dict):
            if decoded.get("type") == "end":
                after = decoded.get("after") or decoded
                event_id = after.get("id")
                if event_id:
                    self._aggregator.observe(
                        observation_id=self._observation_id("event_ended", decoded),
                        event_id=str(event_id),
                        kind="event_ended",
                        payload={"end_time": after.get("end_time")},
                    )
            envelope = self._lpr_envelope(decoded)
            if envelope:
                result = ("license_plate", envelope)
            else:
                envelope = self._object_envelope(decoded)
                result = ("object_detected", envelope) if envelope else None
        elif (
            topic in ("face_recognized", "tracked_object_update")
            and isinstance(decoded, dict)
            and (topic == "face_recognized" or decoded.get("type") == "face")
        ):
            envelope = self._face_envelope(decoded)
            if envelope:
                evidence_id = self._capture_evidence(
                    str(decoded.get("event_id") or decoded.get("id")), decoded
                )
                previous = Event.get_or_none(
                    Event.id == str(decoded.get("event_id") or decoded.get("id"))
                )
                previous_revision = previous.revision if previous else 0
                self._aggregator.observe(
                    observation_id=self._observation_id("face", decoded),
                    event_id=str(decoded.get("event_id") or decoded.get("id")),
                    kind="face",
                    payload={
                        "sub_label": decoded.get("identity") or decoded.get("name"),
                        "score": decoded.get("score"),
                    },
                    frame_time=decoded.get("timestamp"),
                    evidence_id=evidence_id,
                )
                current = Event.get_or_none(
                    Event.id == str(decoded.get("event_id") or decoded.get("id"))
                )
                if (
                    current
                    and current.revision > previous_revision
                    and current.state == "FINALIZED"
                ):
                    self._route_finalized_event(current.id)
            result = ("face_recognized", envelope) if envelope else None
        elif (
            topic == "tracked_object_update"
            and isinstance(decoded, dict)
            and str(decoded.get("type")) in ("lpr", "TrackedObjectUpdateTypesEnum.lpr")
        ):
            self._remember_lpr_update(decoded)
        elif topic == "triggers" and isinstance(decoded, dict):
            envelope = self._trigger_envelope(decoded)
            result = ("semantic_trigger", envelope) if envelope else None
        elif topic == "camera_monitoring" and isinstance(decoded, dict):
            envelope = self._monitoring_envelope(decoded)
            result = ("camera_monitoring", envelope) if envelope else None
        if result:
            self._route(*result)

    @staticmethod
    def _observation_id(kind: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{kind}:{canonical}".encode()).hexdigest()

    def _route_finalized_event(self, event_id: str) -> None:
        event = Event.get_or_none(Event.id == event_id)
        if event is None:
            return
        plate = event.canonical_plate or (event.data or {}).get(
            "recognized_license_plate"
        )
        envelope = NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="event_revision",
            source_id=f"{event.id}:r{event.revision}",
            camera=event.camera,
            timestamp=self._now(),
            title=event.display_label or event.label,
            message="Event finalized",
            direct_url="",
            snapshot_ref=event.id,
            notification_type="event_revision",
            object_label=event.label,
            sub_label=event.canonical_sub_label or event.sub_label,
            genai={"_labels": [event.label], "_zones": event.zones or []},
            lpr_plate=plate,
            lpr_score=event.canonical_plate_score
            or (event.data or {}).get("recognized_license_plate_score"),
        )
        self._route("license_plate" if plate else "alert", envelope)

    def _review_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        after = payload.get("after") or {}
        if after.get("severity") != "alert":
            return None
        camera = after.get("camera")
        if not camera or camera not in self.config.cameras:
            return None
        data = after.get("data") or {}
        metadata = dict(data.get("metadata") or {})
        objects = [
            value for value in data.get("objects", []) if "-verified" not in value
        ]
        objects.extend(data.get("sub_labels", []))
        metadata["_labels"] = objects
        metadata["_zones"] = data.get("zones", [])
        label = ", ".join(sorted(set(objects))) or "Activity"
        camera_name = self.config.cameras[camera].friendly_name or titlecase(
            camera.replace("_", " ")
        )
        source_id = str(after.get("id") or uuid.uuid4())
        event_ids = data.get("detections") or data.get("event_ids") or []
        snapshot_ref = str(event_ids[0]) if event_ids else None
        plate = normalize_plate(data.get("recognized_license_plate"))
        if plate and snapshot_ref:
            self._alert_plates[snapshot_ref] = plate
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="review",
            source_id=source_id,
            camera=camera,
            timestamp=float(after.get("start_time") or self._now()),
            title=metadata.get("title")
            or f"{titlecase(label.replace('_', ' '))} detected",
            message=metadata.get("shortSummary") or f"Detected on {camera_name}",
            direct_url=f"/review?id={source_id}",
            snapshot_ref=snapshot_ref,
            notification_type="alert",
            object_label=objects[0] if objects else None,
            genai=metadata,
            lpr_plate=plate,
            lpr_score=data.get("recognized_license_plate_score"),
            lpr_plate_box=data.get("license_plate_box"),
        )

    def _object_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        if payload.get("type") not in ("new", "update"):
            return None
        after = payload.get("after") or {}
        camera, event_id, label = (
            after.get("camera"),
            after.get("id"),
            after.get("label"),
        )
        if not camera or not event_id or not label:
            return None
        zones = after.get("current_zones") or after.get("entered_zones") or []
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="object",
            source_id=str(event_id),
            camera=camera,
            timestamp=float(after.get("start_time") or self._now()),
            title=f"{titlecase(str(label).replace('_', ' '))} detected",
            message=f"Detected on {camera}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=str(event_id),
            notification_type="object_detected",
            object_label=str(label),
            genai={"_labels": [label], "_zones": zones},
        )

    def _lpr_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        if payload.get("type") != "end":
            return None
        after = payload.get("after") or payload
        if after.get("label") != "car":
            return None
        data = after.get("data") or {}
        event_id = after.get("id")
        update = self._lpr_updates.pop(str(event_id), {}) if event_id else {}
        plate = normalize_plate(
            update.get("plate")
            or data.get("recognized_license_plate")
            or after.get("recognized_license_plate")
        )
        camera = after.get("camera")
        if (
            not plate
            or not camera
            or not event_id
            or self._alert_plates.get(str(event_id)) == plate
        ):
            return None
        score = update.get("score", data.get("recognized_license_plate_score"))
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="lpr",
            source_id=str(event_id),
            camera=camera,
            timestamp=float(after.get("end_time") or self._now()),
            title=f"Vehicle {plate}",
            message=f"Vehicle passage ended on {camera}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=str(event_id),
            notification_type="lpr",
            object_label="car",
            sub_label=update.get("sub_label") or after.get("sub_label"),
            genai={"_labels": ["car"]},
            lpr_plate=plate,
            lpr_score=float(score) if score is not None else None,
            lpr_plate_box=update.get("plate_box") or data.get("license_plate_box"),
        )

    def _remember_lpr_update(self, payload: dict[str, Any]) -> None:
        event_id = payload.get("id")
        plate = normalize_plate(payload.get("plate"))
        if not event_id or not plate:
            return
        self._lpr_updates[str(event_id)] = {
            "plate": plate,
            "score": payload.get("score"),
            "plate_box": payload.get("plate_box"),
            "sub_label": payload.get("name"),
        }
        evidence_id = self._capture_evidence(str(event_id), payload)
        previous = Event.get_or_none(Event.id == str(event_id))
        previous_revision = previous.revision if previous else 0
        self._aggregator.observe(
            observation_id=self._observation_id("lpr", payload),
            event_id=str(event_id),
            kind="lpr",
            payload={
                "plate": plate,
                "score": payload.get("score"),
                "sub_label": payload.get("name"),
                "plate_box": payload.get("plate_box"),
            },
            frame_time=payload.get("frame_time") or payload.get("timestamp"),
            evidence_id=evidence_id,
        )
        current = Event.get_or_none(Event.id == str(event_id))
        if current and current.revision > previous_revision and current.state == "FINALIZED":
            self._route_finalized_event(str(event_id))

    def _capture_evidence(
        self, event_id: str, payload: dict[str, Any]
    ) -> str | None:
        """Copy a detector full frame and frame-local boxes into durable evidence."""
        source = payload.get("frame_ref") or payload.get("full_frame_path")
        if not source:
            return None
        source_path = Path(str(source))
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Canonical evidence frame is unavailable for %s", event_id)
            return None
        evidence_id = str(payload.get("evidence_id") or "").strip()
        if not evidence_id:
            evidence_id = hashlib.sha256(
                f"{event_id}:{payload.get('frame_time') or payload.get('timestamp')}:{source_path}".encode()
            ).hexdigest()
        evidence_path = self._aggregator.media.root / "evidence" / f"{evidence_id}.jpg"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        if not evidence_path.exists():
            temporary = evidence_path.with_suffix(".tmp")
            temporary.write_bytes(source_path.read_bytes())
            temporary.replace(evidence_path)
        boxes = []
        object_box = payload.get("object_box") or payload.get("person_box")
        if object_box:
            boxes.append({"role": "object", "box": object_box})
        for role, key in (("plate", "plate_box"), ("face", "face_box")):
            if payload.get(key):
                boxes.append({"role": role, "box": payload[key]})
        if not boxes:
            return None
        self._aggregator.add_evidence(
            evidence_id=evidence_id,
            event_id=event_id,
            frame_ref=str(evidence_path),
            frame_time=float(
                payload.get("frame_time")
                or payload.get("source_frame_time")
                or payload.get("timestamp")
                or self._now()
            ),
            width=int(payload.get("frame_width") or image.shape[1]),
            height=int(payload.get("frame_height") or image.shape[0]),
            boxes=boxes,
            technical={
                "source": str(payload.get("type") or "enrichment"),
                "detector_timestamp": payload.get("timestamp"),
            },
        )
        return evidence_id

    def _face_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        camera = payload.get("camera")
        event_id = payload.get("event_id") or payload.get("id")
        identity = str(payload.get("identity") or payload.get("name") or "unknown")
        if not camera or not event_id:
            return None
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="face",
            source_id=f"{event_id}:{identity}",
            camera=camera,
            timestamp=float(payload.get("timestamp") or self._now()),
            title=f"Face recognized: {identity}",
            message=f"Recognized on {camera}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=str(event_id),
            notification_type="face_recognized",
            sub_label=identity,
            genai={"score": payload.get("score")},
        )

    def _trigger_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        camera, name = payload.get("camera"), payload.get("name")
        if not camera or not name:
            return None
        triggers = self.config.cameras[camera].semantic_search.triggers or {}
        if name not in triggers or "notification" not in triggers[name].actions:
            return None
        event_id = str(payload.get("event_id") or uuid.uuid4())
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="trigger",
            source_id=f"{event_id}:{name}",
            camera=camera,
            timestamp=self._now(),
            title=f"{name.replace('_', ' ')} triggered",
            message=f"Semantic trigger score {float(payload.get('score', 0)):.2f}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=event_id,
            notification_type="trigger",
            genai={"trigger_name": name},
        )

    def _monitoring_envelope(
        self, payload: dict[str, Any]
    ) -> NotificationEnvelope | None:
        camera = payload.get("camera")
        if not camera:
            return None
        source_id = str(
            payload.get("id") or payload.get("job_id") or f"{camera}:{int(self._now())}"
        )
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="camera_monitoring",
            source_id=source_id,
            camera=camera,
            timestamp=self._now(),
            title=f"{camera}: Monitoring alert",
            message=str(payload.get("message") or payload.get("reasoning") or "")[:200],
            direct_url=f"/#{camera}",
            snapshot_ref=None,
            notification_type="monitoring",
            genai={
                "condition": payload.get("condition"),
                "reasoning": payload.get("reasoning"),
            },
        )

    def _handle_camera_status(self, camera: str, status: str) -> None:
        if camera not in self.config.cameras or status not in (
            "online",
            "offline",
            "disabled",
        ):
            return
        previous = self._camera_status.get(camera)
        self._camera_status[camera] = status
        if status == "offline":
            if previous != "offline":
                self._offline_since[camera] = self._now()
            return
        self._offline_since.pop(camera, None)
        if status == "online" and camera in self._offline_notified:
            self._offline_notified.remove(camera)
            self._route(
                "camera_online",
                self._camera_status_envelope(camera, "online", self._now()),
            )

    def _camera_status_envelope(
        self, camera: str, status: str, timestamp: float
    ) -> NotificationEnvelope:
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="camera_status",
            source_id=f"{camera}:{status}:{int(timestamp)}",
            camera=camera,
            timestamp=timestamp,
            title=f"Camera {status}",
            message=f"{camera} is {status}",
            direct_url=f"/#{camera}",
            snapshot_ref=None,
            notification_type=f"camera_{status}",
        )

    @staticmethod
    def _test_envelope() -> NotificationEnvelope:
        now = datetime.datetime.now(datetime.UTC).timestamp()
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="test",
            source_id=str(uuid.uuid4()),
            camera=None,
            timestamp=now,
            title="Test Notification",
            message="This is a test notification from Frigate.",
            direct_url="/",
            snapshot_ref=None,
            notification_type="test",
        )

    def enqueue_test(self, provider: str, recipient_id: str) -> str | None:
        if provider == "webpush":
            envelope = self._test_envelope()
            return envelope.id if self.webpush.deliver(envelope) else None
        return self.social.enqueue_test(provider, recipient_id)

    def enqueue_rule_test(self, rule_id: str) -> list[str] | None:
        rule = next(
            (rule for rule in self.config.notifications.rules if rule.id == rule_id),
            None,
        )
        if rule is None or not rule.enabled:
            return None
        camera = (
            rule.filters.cameras[0]
            if rule.filters.cameras
            else next(iter(self.config.cameras), None)
        )
        envelope = replace(self._test_envelope(), camera=camera, rule_id=rule.id)
        deliveries: list[str] = []
        if (
            rule.destinations.webpush
            and self.config.notifications.channels.webpush.enabled
            and self.webpush.deliver(envelope)
        ):
            deliveries.append(envelope.id)
        deliveries.extend(self.social.enqueue(envelope, rule.destinations))
        return deliveries

    def provider_status(self) -> dict[str, Any]:
        status = self.social.status()
        status["webpush"] = {
            "enabled": self.config.notifications.channels.webpush.enabled,
            "configured": self.webpush.configured,
            "readiness": (
                "missing"
                if not self.webpush.configured
                else "degraded"
                if self.config.notifications.public_base_url
                and not self.social.public_media_ready()
                else "ready"
            ),
            "pending": self.webpush.pending,
            "last_success": None,
            "last_error": None,
        }
        return status

    def stop(self) -> None:
        self.global_config_subscriber.stop()
        self.social.stop()
        self.webpush.stop()
        self._maintenance_thread.join(timeout=5)
