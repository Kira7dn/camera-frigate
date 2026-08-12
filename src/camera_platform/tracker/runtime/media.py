"""Private edge media authority and bounded byte-range reader."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from frigate.infrastructure.config import FrigateConfig
from frigate.const import CLIPS_DIR, THUMB_DIR
from frigate.domain.record.clip import materialize_recording_clip

from ..domain.contracts import MediaManifest


class MediaAuthority:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._database = sqlite3.connect(
            self.root / "media-manifests.db", check_same_thread=False
        )
        self._database.execute(
            """
            CREATE TABLE IF NOT EXISTS media_manifest (
                media_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                path TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                codec TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                expiry_unix_ms INTEGER NOT NULL,
                media_type TEXT NOT NULL,
                retained INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._database.commit()
        self._manifests: dict[str, MediaManifest] = {}
        self._paths: dict[str, Path] = {}
        self._retained: set[str] = set()
        self._restore()

    @staticmethod
    def _validate_id(media_id: str) -> None:
        if not media_id or any(char in media_id for char in ("/", "\\", "..")):
            raise ValueError("media_id must be an opaque local identifier")

    def _restore(self) -> None:
        missing: list[str] = []
        for row in self._database.execute(
            """
            SELECT media_id, event_id, camera_id, path, start_time, end_time,
                   codec, byte_size, sha256, expiry_unix_ms, media_type, retained
            FROM media_manifest
            """
        ):
            media_id, event_id, camera_id, path_text, *values = row
            path = Path(path_text)
            if not path.is_file():
                missing.append(media_id)
                continue
            start_time, end_time, codec, byte_size, sha256, expiry, media_type, retained = (
                values
            )
            self._manifests[media_id] = MediaManifest(
                media_id=media_id,
                event_id=event_id,
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                codec=codec,
                byte_size=byte_size,
                sha256=sha256,
                expiry_unix_ms=expiry,
                media_type=media_type,
            )
            self._paths[media_id] = path
            if retained:
                self._retained.add(media_id)
        if missing:
            self._database.executemany(
                "DELETE FROM media_manifest WHERE media_id = ?",
                ((media_id,) for media_id in missing),
            )
            self._database.commit()

    def _persist(self, manifest: MediaManifest, path: Path) -> None:
        self._database.execute(
            """
            INSERT OR REPLACE INTO media_manifest (
                media_id, event_id, camera_id, path, start_time, end_time, codec,
                byte_size, sha256, expiry_unix_ms, media_type, retained
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.media_id,
                manifest.event_id,
                manifest.camera_id,
                str(path),
                manifest.start_time,
                manifest.end_time,
                manifest.codec,
                manifest.byte_size,
                manifest.sha256,
                manifest.expiry_unix_ms,
                manifest.media_type,
                int(manifest.media_id in self._retained),
            ),
        )
        self._database.commit()

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
        self._validate_id(media_id)
        path = self.root / media_id
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
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
        with self._lock:
            self._manifests[media_id] = manifest
            self._paths[media_id] = path
            self._persist(manifest, path)
        return manifest

    def register_file(
        self,
        *,
        media_id: str,
        event_id: str,
        camera_id: str,
        path: str | Path,
        timestamp: float,
        media_type: str,
        codec: str,
        ttl_seconds: float,
    ) -> MediaManifest:
        self._validate_id(media_id)
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest = MediaManifest(
            media_id=media_id,
            event_id=event_id,
            camera_id=camera_id,
            start_time=timestamp,
            end_time=timestamp,
            codec=codec,
            byte_size=source.stat().st_size,
            sha256=digest.hexdigest(),
            expiry_unix_ms=int((time.time() + ttl_seconds) * 1000),
            media_type=media_type,
        )
        with self._lock:
            self._manifests[media_id] = manifest
            self._paths[media_id] = source
            self._persist(manifest, source)
        return manifest

    def register_event_files(
        self,
        *,
        camera_id: str,
        raw_track_id: str,
        event_id: str,
        frame_time: float,
        has_snapshot: bool,
        ttl_seconds: float = 3600,
    ) -> tuple[MediaManifest, ...]:
        candidates = [
            (
                f"thumbnail-{event_id}",
                Path(THUMB_DIR) / camera_id / f"{raw_track_id}.webp",
                "thumbnail",
            )
        ]
        if has_snapshot:
            candidates.append(
                (
                    f"snapshot-{event_id}",
                    Path(CLIPS_DIR) / f"{camera_id}-{raw_track_id}-clean.webp",
                    "snapshot",
                )
            )
        output = []
        for media_id, path, media_type in candidates:
            if path.is_file():
                output.append(
                    self.register_file(
                        media_id=media_id,
                        event_id=event_id,
                        camera_id=camera_id,
                        path=path,
                        timestamp=frame_time,
                        media_type=media_type,
                        codec="webp",
                        ttl_seconds=ttl_seconds,
                    )
                )
        return tuple(output)

    def register_event_clip(
        self,
        *,
        config: FrigateConfig,
        camera_id: str,
        event_id: str,
        start_time: float,
        end_time: float,
        ttl_seconds: float = 3600,
    ) -> MediaManifest | None:
        path = self.root / "clips" / f"{event_id}.mp4"
        if not materialize_recording_clip(
            config, camera_id, start_time, end_time, path
        ):
            return None
        return self.register_file(
            media_id=f"clip-{event_id}",
            event_id=event_id,
            camera_id=camera_id,
            path=path,
            timestamp=end_time,
            media_type="clip",
            codec="h264",
            ttl_seconds=ttl_seconds,
        )

    def manifest(self, media_id: str) -> MediaManifest:
        with self._lock:
            return self._manifests[media_id]

    def read_range(
        self, media_id: str, offset: int = 0, length: int | None = None
    ) -> bytes:
        with self._lock:
            manifest = self._manifests[media_id]
            path = self._paths[media_id]
        if offset < 0 or offset > manifest.byte_size:
            raise ValueError("media range offset is invalid")
        if length is not None and length < 0:
            raise ValueError("media range length is invalid")
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read() if length is None else handle.read(length)
        return data

    def retain(self, media_id: str) -> bool:
        with self._lock:
            if media_id not in self._manifests:
                return False
            self._retained.add(media_id)
            self._database.execute(
                "UPDATE media_manifest SET retained = 1 WHERE media_id = ?",
                (media_id,),
            )
            self._database.commit()
            return True

    def delete(self, media_id: str) -> bool:
        with self._lock:
            manifest = self._manifests.pop(media_id, None)
            path = self._paths.pop(media_id, None)
            self._retained.discard(media_id)
            self._database.execute(
                "DELETE FROM media_manifest WHERE media_id = ?", (media_id,)
            )
            self._database.commit()
        if manifest is None:
            return True
        if path is not None:
            path.unlink(missing_ok=True)
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

    def close(self) -> None:
        with self._lock:
            self._database.close()
