"""Private edge media authority and bounded byte-range reader."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .contracts import MediaManifest


class MediaAuthority:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, MediaManifest] = {}
        self._retained: set[str] = set()

    def register(
        self,
        *,
        media_id: str,
        event_id: str,
        camera_id: str,
        data: bytes,
        start_time: float,
        end_time: float,
        codec: str,
        media_type: str,
        ttl_seconds: float,
    ) -> MediaManifest:
        if not media_id or any(char in media_id for char in ("/", "\\", "..")):
            raise ValueError("media_id must be an opaque local identifier")
        path = self.root / media_id
        path.write_bytes(data)
        manifest = MediaManifest(
            media_id=media_id,
            event_id=event_id,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            codec=codec,
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            expiry_unix_ms=int((time.time() + ttl_seconds) * 1000),
            media_type=media_type,
        )
        self._manifests[media_id] = manifest
        return manifest

    def manifest(self, media_id: str) -> MediaManifest:
        return self._manifests[media_id]

    def read_range(
        self, media_id: str, offset: int = 0, length: int | None = None
    ) -> bytes:
        manifest = self._manifests[media_id]
        if offset < 0 or offset > manifest.byte_size:
            raise ValueError("media range offset is invalid")
        if length is not None and length < 0:
            raise ValueError("media range length is invalid")
        with (self.root / media_id).open("rb") as handle:
            handle.seek(offset)
            data = handle.read() if length is None else handle.read(length)
        return data

    def retain(self, media_id: str) -> bool:
        if media_id not in self._manifests:
            return False
        self._retained.add(media_id)
        return True

    def delete(self, media_id: str) -> bool:
        manifest = self._manifests.pop(media_id, None)
        self._retained.discard(media_id)
        if manifest is None:
            return True
        (self.root / media_id).unlink(missing_ok=True)
        return True

    def expire(self, now: float | None = None) -> int:
        now_ms = int((time.time() if now is None else now) * 1000)
        expired = [
            media_id
            for media_id, manifest in self._manifests.items()
            if media_id not in self._retained and manifest.expiry_unix_ms <= now_ms
        ]
        for media_id in expired:
            self.delete(media_id)
        return len(expired)
