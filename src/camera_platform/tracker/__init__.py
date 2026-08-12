"""External edge runtime for Frigate's shared tracking domain."""

from .domain.contracts import (
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
