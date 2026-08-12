import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class EventResponse(BaseModel):
    id: str
    label: str
    sub_label: str | None
    camera: str
    start_time: float
    end_time: float | None
    false_positive: bool | None
    zones: list[str]
    thumbnail: str | None
    has_clip: bool
    has_snapshot: bool
    retain_indefinitely: bool
    plus_id: str | None
    model_hash: str | None
    detector_type: str | None
    model_type: str | None
    data: dict[str, Any]
    state: str = "TRACKING"
    revision: int = 0
    finalized_at: datetime.datetime | None = None
    canonical_plate: str | None = None
    canonical_plate_score: float | None = None
    canonical_sub_label: str | None = None
    display_label: str | None = None
    canonical_evidence_id: str | None = None
    canonical_artifact_id: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class EventCreateResponse(BaseModel):
    success: bool
    message: str
    event_id: str


class EventMultiDeleteResponse(BaseModel):
    success: bool
    deleted_events: list[str]
    not_found_events: list[str]


class EventUploadPlusResponse(BaseModel):
    success: bool
    plus_id: str
