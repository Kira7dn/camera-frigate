"""Normalized notification payload contract."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NotificationEnvelope:
    id: str
    source_type: str
    source_id: str
    camera: str | None
    timestamp: float
    title: str
    message: str
    direct_url: str
    snapshot_ref: str | None
    notification_type: str
    object_label: str | None = None
    sub_label: str | None = None
    genai: dict[str, Any] = field(default_factory=dict)
    lpr_plate: str | None = None
    lpr_score: float | None = None
    lpr_plate_box: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NotificationEnvelope":
        """Restore an envelope from an outbox payload."""
        return cls(**value)
