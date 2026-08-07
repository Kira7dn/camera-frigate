"""Shared native notification policy and fan-out pipeline."""

import datetime
import json
import logging
import re
import threading
import uuid
from collections.abc import Callable
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

from titlecase import titlecase

from frigate.comms.base_communicator import Communicator
from frigate.comms.config_updater import ConfigSubscriber
from frigate.config import FrigateConfig
from frigate.config.auth import AuthConfig
from frigate.config.camera.updater import (
    CameraConfigUpdateEnum,
    CameraConfigUpdateSubscriber,
)

from .envelope import NotificationEnvelope
from .metrics import increment
from .social import SocialClient
from .webpush import WebPushProvider

logger = logging.getLogger(__name__)


def normalize_plate(value: Any) -> str | None:
    """Normalize a recognized plate for notification deduplication."""
    plate = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    return plate or None


class NotificationClient(Communicator):
    """Apply notification policy once and fan out to enabled providers."""

    def __init__(self, config: FrigateConfig, stop_event: MpEvent) -> None:
        self.config = config
        self.stop_event = stop_event
        self.webpush = WebPushProvider(config, stop_event)
        self.social = SocialClient(config, stop_event)
        self.suspended_cameras = {camera: 0 for camera in config.cameras}
        self.last_camera_notification_time = {camera: 0.0 for camera in config.cameras}
        self.last_notification_time = 0.0
        self.suspension_broadcaster: Callable[[str, Any, bool], None] | None = None
        self.global_config_subscriber = ConfigSubscriber("config/")
        self.config_subscriber = CameraConfigUpdateSubscriber(
            config, config.cameras, [CameraConfigUpdateEnum.notifications]
        )
        self._suspension_thread = threading.Thread(
            target=self._process_suspensions, daemon=True
        )
        self._suspension_thread.start()

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
        if (
            suspended_until
            and suspended_until <= datetime.datetime.now(datetime.UTC).timestamp()
        ):
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

    def _process_suspensions(self) -> None:
        while not self.stop_event.wait(1):
            for camera in tuple(self.suspended_cameras):
                self.is_camera_suspended(camera)

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
        updates = self.config_subscriber.check_for_updates()
        changed = changed or bool(updates)
        for camera in updates.get("add", []):
            self.suspended_cameras[camera] = 0
            self.last_camera_notification_time[camera] = 0.0
        if changed:
            self.social.cancel_disabled()

    def _eligible(self, camera: str, provider: str | None = None) -> bool:
        camera_config = self.config.cameras.get(camera)
        if camera_config is None or not camera_config.notifications.enabled:
            return False
        if provider and provider not in camera_config.notifications.providers:
            return False
        return not self.is_camera_suspended(camera)

    def _within_cooldown(self, camera: str) -> bool:
        now = datetime.datetime.now(datetime.UTC).timestamp()
        return (
            now - self.last_notification_time < self.config.notifications.cooldown
            or now - self.last_camera_notification_time.get(camera, 0)
            < self.config.cameras[camera].notifications.cooldown
        )

    def _fan_out(
        self, envelope: NotificationEnvelope, *, apply_cooldown: bool = True
    ) -> list[str]:
        camera = envelope.camera
        if camera and (
            not self._eligible(camera)
            or (apply_cooldown and self._within_cooldown(camera))
        ):
            increment("all", "skipped_policy")
            return []
        deliveries: list[str] = []
        webpush_deliveries = 0
        if (
            envelope.source_type != "lpr"
            and (camera is None or self._eligible(camera, "webpush"))
        ) and self.config.notifications.providers.webpush.enabled:
            webpush_deliveries = self.webpush.deliver(envelope)
        deliveries.extend(self.social.enqueue(envelope))
        if camera and (webpush_deliveries or deliveries):
            now = datetime.datetime.now(datetime.UTC).timestamp()
            self.last_notification_time = now
            self.last_camera_notification_time[camera] = now
        return deliveries

    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        self._refresh_config()
        try:
            decoded = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            return
        envelope: NotificationEnvelope | None = None
        if topic == "reviews" and isinstance(decoded, dict):
            envelope = self._review_envelope(decoded)
        elif topic == "triggers" and isinstance(decoded, dict):
            envelope = self._trigger_envelope(decoded)
        elif topic == "camera_monitoring" and isinstance(decoded, dict):
            envelope = self._monitoring_envelope(decoded)
        elif topic == "notification_test":
            envelope = self._test_envelope()
        elif topic == "events" and isinstance(decoded, dict):
            envelope = self._lpr_envelope(decoded)
        if envelope:
            self._fan_out(envelope, apply_cooldown=envelope.source_type != "test")

    def _review_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        after = payload.get("after") or {}
        if after.get("severity") != "alert":
            return None
        camera = after.get("camera")
        if not camera:
            return None
        data = after.get("data") or {}
        state = payload.get("type", "update")
        metadata = data.get("metadata") or {}
        objects = [
            value for value in data.get("objects", []) if "-verified" not in value
        ]
        objects.extend(data.get("sub_labels", []))
        label = ", ".join(sorted(set(objects))) or "Activity"
        camera_name = self.config.cameras[camera].friendly_name or titlecase(
            camera.replace("_", " ")
        )
        if metadata.get("title"):
            title = metadata["title"]
            message = metadata.get("shortSummary") or f"Detected on {camera_name}"
        else:
            title = f"{titlecase(label.replace('_', ' '))} detected"
            message = f"Detected on {camera_name}"
        source_id = str(after.get("id") or uuid.uuid4())
        event_ids = data.get("detections") or data.get("event_ids") or []
        snapshot_ref = str(event_ids[0]) if event_ids else None
        plate = normalize_plate(data.get("recognized_license_plate"))
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="review",
            source_id=source_id,
            camera=camera,
            timestamp=float(
                after.get("end_time")
                or after.get("start_time")
                or datetime.datetime.now(datetime.UTC).timestamp()
            ),
            title=title,
            message=message,
            direct_url=f"/review?id={source_id}"
            if state in ("end", "genai")
            else f"/#{camera}",
            snapshot_ref=snapshot_ref,
            notification_type="alert",
            object_label=label,
            genai=metadata,
            lpr_plate=plate,
            lpr_score=data.get("recognized_license_plate_score"),
            lpr_plate_box=data.get("license_plate_box"),
        )

    def _trigger_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        camera = payload.get("camera")
        name = payload.get("name")
        if not camera or not name:
            return None
        triggers = self.config.cameras[camera].semantic_search.triggers or {}
        if name not in triggers or "notification" not in triggers[name].actions:
            return None
        event_id = str(payload.get("event_id") or uuid.uuid4())
        score = float(payload.get("score", 0))
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="trigger",
            source_id=f"{event_id}:{name}",
            camera=camera,
            timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            title=f"{name.replace('_', ' ')} triggered",
            message=f"{titlecase(str(payload.get('type', 'semantic')))} trigger score {score:.2f}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=event_id,
            notification_type="trigger",
        )

    def _monitoring_envelope(
        self, payload: dict[str, Any]
    ) -> NotificationEnvelope | None:
        camera = payload.get("camera")
        if not camera:
            return None
        camera_name = self.config.cameras[camera].friendly_name or titlecase(
            camera.replace("_", " ")
        )
        message = str(payload.get("message") or payload.get("reasoning") or "")
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="camera_monitoring",
            source_id=str(
                payload.get("id")
                or f"{camera}:{int(datetime.datetime.now(datetime.UTC).timestamp())}"
            ),
            camera=camera,
            timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            title=f"{camera_name}: Monitoring Alert",
            message=message[:200],
            direct_url=f"/#{camera}",
            snapshot_ref=None,
            notification_type="monitoring",
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

    def _lpr_envelope(self, payload: dict[str, Any]) -> NotificationEnvelope | None:
        if payload.get("type") != "end":
            return None
        after = payload.get("after") or payload
        if after.get("label") != "car":
            return None
        data = after.get("data") or {}
        plate = normalize_plate(
            data.get("recognized_license_plate")
            or after.get("recognized_license_plate")
        )
        camera = after.get("camera")
        event_id = after.get("id")
        if not plate or not camera or not event_id:
            return None
        score = data.get("recognized_license_plate_score")
        return NotificationEnvelope(
            id=str(uuid.uuid4()),
            source_type="lpr",
            source_id=str(event_id),
            camera=camera,
            timestamp=float(
                after.get("end_time") or datetime.datetime.now(datetime.UTC).timestamp()
            ),
            title=f"Vehicle {plate}",
            message=f"Vehicle passage ended on {camera}",
            direct_url=f"/explore?event_id={event_id}",
            snapshot_ref=str(event_id),
            notification_type="lpr",
            object_label="car",
            sub_label=after.get("sub_label"),
            lpr_plate=plate,
            lpr_score=float(score) if score is not None else None,
            lpr_plate_box=data.get("license_plate_box"),
        )

    def enqueue_test(self, provider: str, recipient_id: str) -> str | None:
        if provider == "webpush":
            envelope = self._test_envelope()
            return envelope.id if self.webpush.deliver(envelope) else None
        return self.social.enqueue_test(provider, recipient_id)

    def provider_status(self) -> dict[str, Any]:
        status = self.social.status()
        status["webpush"] = {
            "enabled": self.config.notifications.providers.webpush.enabled,
            "configured": self.webpush.configured,
            "readiness": "ready" if self.webpush.configured else "missing",
            "pending": self.webpush.pending,
            "last_success": None,
            "last_error": None,
        }
        return status

    def stop(self) -> None:
        self.global_config_subscriber.stop()
        self.config_subscriber.stop()
        self.social.stop()
        self.webpush.stop()
        self._suspension_thread.join(timeout=5)
