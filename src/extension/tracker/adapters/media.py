"""Authenticated Frigate-side resolver for private edge-owned media."""

from __future__ import annotations

from dataclasses import dataclass

from frigate.models import EdgeMediaManifest


@dataclass(frozen=True, slots=True)
class EdgeMediaRange:
    manifest: EdgeMediaManifest
    offset: int
    length: int | None


def resolve_event_media(
    event_id: str, media_type: str, range_header: str | None = None
) -> EdgeMediaRange | None:
    manifest = (
        EdgeMediaManifest.select()
        .where(
            (EdgeMediaManifest.event_id == event_id)
            & (EdgeMediaManifest.media_type == media_type)
        )
        .order_by(EdgeMediaManifest.end_time.desc())
        .first()
    )
    if manifest is None:
        return None
    if not range_header:
        return EdgeMediaRange(manifest, 0, None)
    if not range_header.startswith("bytes=") or "," in range_header:
        raise ValueError("invalid_media_range")
    start_text, end_text = range_header[6:].split("-", 1)
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("invalid_media_range")
        offset = max(0, manifest.byte_size - suffix)
        return EdgeMediaRange(manifest, offset, manifest.byte_size - offset)
    offset = int(start_text)
    end = manifest.byte_size - 1 if not end_text else int(end_text)
    if offset < 0 or end < offset or end >= manifest.byte_size:
        raise ValueError("invalid_media_range")
    return EdgeMediaRange(manifest, offset, end - offset + 1)


def resolve_media_id(media_id: str) -> EdgeMediaManifest | None:
    """Resolve one opaque edge artifact without exposing its private path."""
    return EdgeMediaManifest.get_or_none(EdgeMediaManifest.media_id == media_id)
