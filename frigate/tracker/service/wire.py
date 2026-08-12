"""Conversion between tracker contracts and protobuf messages."""

from __future__ import annotations

import json

from ..contracts import (
    BoundingBox,
    EvidenceReference,
    LifecycleFailure,
    MediaManifest,
    TrackerOperation,
    TrackerUpdate,
)
from .v1 import tracker_pb2 as pb

_TO_PROTO = {
    TrackerOperation.START: pb.TRACKER_OPERATION_START,
    TrackerOperation.UPDATE: pb.TRACKER_OPERATION_UPDATE,
    TrackerOperation.END: pb.TRACKER_OPERATION_END,
}
_FROM_PROTO = {value: key for key, value in _TO_PROTO.items()}


def update_to_proto(update: TrackerUpdate) -> pb.TrackerUpdate:
    message = pb.TrackerUpdate(
        node_id=update.node_id,
        node_epoch=update.node_epoch,
        camera_id=update.camera_id,
        stream_epoch=update.stream_epoch,
        journal_sequence=update.journal_sequence,
        frame_seq=update.frame_seq,
        source_pts=update.source_pts,
        frame_time=update.frame_time,
        event_id=update.event_id,
        track_id=update.track_id,
        operation=_TO_PROTO[update.operation],
        label=update.label,
        score_history=update.score_history,
        score=update.score,
        bbox=pb.BBox(
            left=update.bbox.left,
            top=update.bbox.top,
            right=update.bbox.right,
            bottom=update.bbox.bottom,
        ),
        attributes_json=json.dumps(update.attributes, sort_keys=True),
        current_zones=update.current_zones,
        entered_zones=update.entered_zones,
        path_json=json.dumps(update.path),
        motion_json=json.dumps(update.motion, sort_keys=True),
        region_json=json.dumps(update.region, sort_keys=True),
    )
    if update.speed is not None:
        message.speed = update.speed
    if update.evidence is not None:
        message.evidence.CopyFrom(
            pb.EvidenceReference(
                evidence_id=update.evidence.evidence_id,
                byte_length=update.evidence.byte_length,
                sha256=update.evidence.sha256,
                expiry_unix_ms=update.evidence.expiry_unix_ms,
                durable=update.evidence.durable,
            )
        )
    if update.failure is not None:
        message.failure.CopyFrom(
            pb.LifecycleFailure(
                code=update.failure.code,
                detail=update.failure.detail,
                retryable=update.failure.retryable,
                gap=update.failure.gap,
            )
        )
    message.media.extend(
        pb.MediaManifest(
            media_id=item.media_id,
            event_id=item.event_id,
            camera_id=item.camera_id,
            start_time=item.start_time,
            end_time=item.end_time,
            codec=item.codec,
            byte_size=item.byte_size,
            sha256=item.sha256,
            expiry_unix_ms=item.expiry_unix_ms,
            media_type=item.media_type,
        )
        for item in update.media
    )
    return message


def update_from_proto(message: pb.TrackerUpdate) -> TrackerUpdate:
    return TrackerUpdate(
        node_id=message.node_id,
        node_epoch=message.node_epoch,
        camera_id=message.camera_id,
        stream_epoch=message.stream_epoch,
        journal_sequence=message.journal_sequence,
        frame_seq=message.frame_seq,
        source_pts=message.source_pts,
        frame_time=message.frame_time,
        event_id=message.event_id,
        track_id=message.track_id,
        operation=_FROM_PROTO[message.operation],
        label=message.label,
        score_history=tuple(message.score_history),
        score=message.score,
        bbox=BoundingBox(
            message.bbox.left,
            message.bbox.top,
            message.bbox.right,
            message.bbox.bottom,
        ),
        attributes=json.loads(message.attributes_json or "{}"),
        current_zones=tuple(message.current_zones),
        entered_zones=tuple(message.entered_zones),
        path=tuple(tuple(point) for point in json.loads(message.path_json or "[]")),
        speed=message.speed if message.HasField("speed") else None,
        motion=json.loads(message.motion_json or "{}"),
        region=json.loads(message.region_json or "{}"),
        evidence=(
            EvidenceReference(
                message.evidence.evidence_id,
                message.evidence.byte_length,
                message.evidence.sha256,
                message.evidence.expiry_unix_ms,
                message.evidence.durable,
            )
            if message.HasField("evidence")
            else None
        ),
        failure=(
            LifecycleFailure(
                message.failure.code,
                message.failure.detail,
                message.failure.retryable,
                message.failure.gap,
            )
            if message.HasField("failure")
            else None
        ),
        media=tuple(
            MediaManifest(
                item.media_id,
                item.event_id,
                item.camera_id,
                item.start_time,
                item.end_time,
                item.codec,
                item.byte_size,
                item.sha256,
                item.expiry_unix_ms,
                item.media_type,
            )
            for item in message.media
        ),
    )
