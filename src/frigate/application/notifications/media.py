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

    def signature(self, artifact_id: str, expires: int) -> str:
        value = f"{artifact_id}:{expires}".encode()
        return hmac.new(self.key, value, hashlib.sha256).hexdigest()

    def verify(self, artifact_id: str, expires: int, signature: str) -> bool:
        if expires < int(time.time()):
            return False
        return hmac.compare_digest(self.signature(artifact_id, expires), signature)

    def url(self, public_base_url: str, artifact_id: str, ttl: int) -> str:
        with self._expiry_lock:
            expires = max(int(time.time()) + ttl, self._last_expiry + 1)
            self._last_expiry = expires
        signature = self.signature(artifact_id, expires)
        return (
            f"{public_base_url.rstrip('/')}/api/notifications/media/"
            f"{quote(artifact_id, safe='')}/artifact.jpg?expires={expires}"
            f"&signature={signature}"
        )


def load_snapshot(artifact_id: str) -> bytes | None:
    """Load immutable artifact bytes; never derive presentation from delivery."""
    from frigate.application.events.canonical import CanonicalMediaStore

    return CanonicalMediaStore().read_bytes(artifact_id)
