"""Shared tracker-edge contracts and runtime building blocks."""

from .contracts import (
    BoundingBox,
    EvidenceReference,
    LifecycleFailure,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
    TrackIdentity,
)

__all__ = [
    "BoundingBox",
    "EvidenceReference",
    "LifecycleFailure",
    "MediaManifest",
    "TrackIdentity",
    "TrackerOperation",
    "TrackerUpdate",
]
