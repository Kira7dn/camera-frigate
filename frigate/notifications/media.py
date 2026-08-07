"""Signed notification media URLs and snapshot loading."""

import hashlib
import hmac
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import quote

from frigate.const import CONFIG_DIR

KEY_PATH = Path(CONFIG_DIR) / ".notification_media_key"


class NotificationMediaSigner:
    """Create and verify short-lived event snapshot URLs."""

    def __init__(self, key_path: Path = KEY_PATH) -> None:
        self.key_path = key_path
        self.key = self._load_or_create_key()
        self._expiry_lock = threading.Lock()
        self._last_expiry = 0

    def _load_or_create_key(self) -> bytes:
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.key_path.open("xb") as key_file:
                key = secrets.token_bytes(32)
                key_file.write(key)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
            return key
        except FileExistsError:
            key = self.key_path.read_bytes()
            if len(key) < 32:
                key = secrets.token_bytes(32)
                self.key_path.write_bytes(key)
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
            return key

    def signature(self, event_id: str, expires: int) -> str:
        value = f"{event_id}:{expires}".encode()
        return hmac.new(self.key, value, hashlib.sha256).hexdigest()

    def verify(self, event_id: str, expires: int, signature: str) -> bool:
        if expires < int(time.time()):
            return False
        return hmac.compare_digest(self.signature(event_id, expires), signature)

    def url(self, public_base_url: str, event_id: str, ttl: int) -> str:
        with self._expiry_lock:
            expires = max(int(time.time()) + ttl, self._last_expiry + 1)
            self._last_expiry = expires
        signature = self.signature(event_id, expires)
        return (
            f"{public_base_url.rstrip('/')}/api/notifications/media/"
            f"{quote(event_id, safe='')}/snapshot.jpg?expires={expires}"
            f"&signature={signature}"
        )


def load_snapshot(event_id: str) -> bytes | None:
    """Load an annotated completed event snapshot as JPEG bytes."""
    from peewee import DoesNotExist

    from frigate.models import Event, NotificationDelivery
    from frigate.util.file import get_event_snapshot_bytes

    try:
        event = Event.get_by_id(event_id)
    except DoesNotExist:
        return None
    label = getattr(event, "sub_label", None) or event.label
    extra_overlay_boxes = []
    delivery = (
        NotificationDelivery.select()
        .where(NotificationDelivery.source_id == event_id)
        .order_by(NotificationDelivery.created_at.desc())
        .first()
    )
    if delivery is not None and isinstance(delivery.payload, dict):
        payload = delivery.payload
        label = payload.get("sub_label") or payload.get("lpr_plate") or label
        plate_box = payload.get("lpr_plate_box")
        if isinstance(plate_box, (list, tuple)) and len(plate_box) == 4:
            extra_overlay_boxes.append(
                {
                    "box": tuple(int(value) for value in plate_box),
                    "label": payload.get("lpr_plate") or "license_plate",
                    "score": payload.get("lpr_score"),
                    "color": (0, 255, 255),
                }
            )
    image, _ = get_event_snapshot_bytes(
        event,
        ext="jpg",
        bounding_box=True,
        label=label,
        extra_overlay_boxes=extra_overlay_boxes,
    )
    return image
