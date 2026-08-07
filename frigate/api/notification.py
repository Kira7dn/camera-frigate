"""Notification apis."""

import logging
import os
from typing import Any

from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from peewee import DoesNotExist
from py_vapid import Vapid01, utils
from pydantic import BaseModel

from frigate.api.auth import allow_any_authenticated, require_role
from frigate.api.defs.tags import Tags
from frigate.const import CONFIG_DIR
from frigate.models import User
from frigate.notifications.media import load_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.notifications])


class ProviderTestRequest(BaseModel):
    recipient_id: str = ""


def _notification_client(request: Request):
    dispatcher = request.app.dispatcher
    return dispatcher.notification_client if dispatcher else None


@router.get(
    "/notifications/providers/status",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Get notification provider status",
)
def get_provider_status(request: Request):
    """Return sanitized readiness and outbox state for each provider."""
    client = _notification_client(request)
    if client is None:
        return JSONResponse(
            {"success": False, "message": "Notification client unavailable"},
            status_code=503,
        )
    return client.provider_status()


@router.post(
    "/notifications/providers/{provider}/test",
    dependencies=[Depends(require_role(["admin"]))],
    status_code=202,
    summary="Test a notification provider recipient",
)
def test_provider(request: Request, provider: str, body: ProviderTestRequest):
    """Queue one test notification without exposing provider credentials."""
    if provider not in ("webpush", "telegram", "zalo"):
        return JSONResponse(
            {"success": False, "message": "Unknown notification provider"},
            status_code=404,
        )
    recipient_id = body.recipient_id.strip()
    if provider != "webpush" and not recipient_id:
        return JSONResponse(
            {"success": False, "message": "recipient_id is required"},
            status_code=400,
        )
    client = _notification_client(request)
    delivery_id = client.enqueue_test(provider, recipient_id) if client else None
    if delivery_id is None:
        return JSONResponse(
            {"success": False, "message": "Provider or recipient is not ready"},
            status_code=409,
        )
    return {"success": True, "delivery_id": delivery_id}


@router.get(
    "/notifications/media/{event_id}/snapshot.jpg",
    summary="Get a signed notification snapshot",
)
def notification_snapshot(
    request: Request,
    event_id: str,
    expires: int = Query(),
    signature: str = Query(),
):
    """Serve an event snapshot only when its short-lived HMAC is valid."""
    client = _notification_client(request)
    if client is None or not client.social.signer.verify(event_id, expires, signature):
        return JSONResponse(
            {"success": False, "message": "Invalid or expired media signature"},
            status_code=403,
        )
    snapshot = load_snapshot(event_id)
    if snapshot is None:
        return JSONResponse(
            {"success": False, "message": "Snapshot not available"},
            status_code=404,
        )
    return Response(
        snapshot,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )


@router.get(
    "/notifications/pubkey",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Get VAPID public key",
    description="""Gets the VAPID public key for the notifications.
    Returns the public key or an error if notifications are not enabled.
    """,
)
def get_vapid_pub_key(request: Request):
    config = request.app.frigate_config
    notifications_enabled = config.notifications.enabled
    camera_notifications_enabled = [
        c for c in config.cameras.values() if c.enabled and c.notifications.enabled
    ]
    if not (notifications_enabled or camera_notifications_enabled):
        return JSONResponse(
            content=({"success": False, "message": "Notifications are not enabled."}),
            status_code=400,
        )

    key = Vapid01.from_file(os.path.join(CONFIG_DIR, "notifications.pem"))
    raw_pub = key.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return JSONResponse(content=utils.b64urlencode(raw_pub), status_code=200)


@router.post(
    "/notifications/register",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Register notifications",
    description="""Registers a notifications subscription.
    Returns a success message or an error if the subscription is not provided.
    """,
)
def register_notifications(request: Request, body: dict | None = None):
    if request.app.frigate_config.auth.enabled:
        # FIXME: For FastAPI the remote-user is not being populated
        username = request.headers.get("remote-user") or "admin"
    else:
        username = "admin"

    json: dict[str, Any] = body or {}
    sub = json.get("sub")

    if not sub:
        return JSONResponse(
            content={"success": False, "message": "Subscription must be provided."},
            status_code=400,
        )

    try:
        User.update(notification_tokens=User.notification_tokens.append(sub)).where(
            User.username == username
        ).execute()
        return JSONResponse(
            content=({"success": True, "message": "Successfully saved token."}),
            status_code=200,
        )
    except DoesNotExist:
        return JSONResponse(
            content=({"success": False, "message": "Could not find user."}),
            status_code=404,
        )
