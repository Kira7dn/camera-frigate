"""Notification apis."""

import datetime
import hashlib
import io
import logging
import os
import shutil
from typing import Any

from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from filelock import FileLock, Timeout
from peewee import DoesNotExist
from py_vapid import Vapid01, utils
from pydantic import BaseModel, ValidationError
from ruamel.yaml import YAML

from frigate.api.auth import allow_any_authenticated, require_role
from frigate.api.config_util import swap_runtime_config
from frigate.api.defs.tags import Tags
from frigate.config import FrigateConfig
from frigate.config.camera.notification import (
    NotificationConfig,
    NotificationRuleConfig,
)
from frigate.const import CONFIG_DIR
from frigate.models import User
from frigate.notifications.media import load_snapshot
from frigate.util.config import find_config_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.notifications])


class ProviderTestRequest(BaseModel):
    recipient_id: str = ""


class NotificationConfigPutRequest(BaseModel):
    revision: str
    notifications: dict[str, Any]


class NotificationRulePreviewRequest(BaseModel):
    rule: dict[str, Any]


def _config_revision(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dump_yaml(data: Any) -> str:
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    output = io.StringIO()
    yaml.dump(data, output)
    return output.getvalue()


def _notification_capabilities(config: FrigateConfig) -> dict[str, Any]:
    return {
        "events": [
            "alert",
            "object_detected",
            "license_plate",
            "face_recognized",
            "camera_offline",
            "camera_online",
            "semantic_trigger",
            "camera_monitoring",
        ],
        "cameras": sorted(config.cameras),
        "filters": {
            "common": ["cameras"],
            "alert": ["labels", "zones"],
            "object_detected": ["labels", "zones"],
            "face_recognized": ["identities"],
            "semantic_trigger": ["trigger_names"],
            "camera_monitoring": ["conditions"],
        },
    }


def _restore_recipient_chat_ids(
    serialized: dict[str, Any], requested: dict[str, Any]
) -> None:
    """Keep the exact literal or ``{ENV_VAR}`` submitted by the editor."""
    requested_channels = requested.get("channels") or {}
    serialized_channels = serialized.get("channels") or {}
    for channel in ("telegram", "zalo"):
        requested_recipients = {
            str(recipient.get("id")): recipient.get("chat_id")
            for recipient in (requested_channels.get(channel) or {}).get(
                "recipients", []
            )
            if isinstance(recipient, dict)
        }
        for recipient in (serialized_channels.get(channel) or {}).get("recipients", []):
            recipient_id = str(recipient.get("id"))
            if recipient_id in requested_recipients:
                recipient["chat_id"] = requested_recipients[recipient_id]


def _notification_client(request: Request):
    dispatcher = request.app.dispatcher
    return dispatcher.notification_client if dispatcher else None


@router.get(
    "/notifications/config",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Get the authoritative notification document",
)
def get_notification_config(request: Request):
    config_file = find_config_file()
    with open(config_file, encoding="utf-8") as file:
        raw = file.read()
    yaml = YAML()
    raw_document = yaml.load(raw) or {}
    notifications = request.app.frigate_config.notifications.model_dump(
        mode="json", exclude_none=True, exclude={"enabled_in_config"}
    )
    _restore_recipient_chat_ids(
        notifications, dict(raw_document.get("notifications") or {})
    )
    return {
        "revision": _config_revision(raw),
        "notifications": notifications,
        "capabilities": _notification_capabilities(request.app.frigate_config),
        "providers": (
            _notification_client(request).provider_status()
            if _notification_client(request)
            else {}
        ),
    }


@router.put(
    "/notifications/config",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Save the complete notification document",
)
def put_notification_config(request: Request, body: NotificationConfigPutRequest):
    """Validate, serialize, and hot-reload one notification transaction."""
    config_file = find_config_file()
    lock = FileLock(f"{config_file}.lock", timeout=5)
    try:
        with lock:
            with open(config_file, encoding="utf-8") as file:
                old_raw = file.read()
            if _config_revision(old_raw) != body.revision:
                return JSONResponse(
                    {
                        "success": False,
                        "message": "Configuration changed since it was loaded",
                        "revision": _config_revision(old_raw),
                    },
                    status_code=409,
                )

            # Validate the domain document first, including recipient references.
            notification_config = NotificationConfig.model_validate(body.notifications)
            yaml = YAML()
            data = yaml.load(old_raw)
            serialized_notifications = notification_config.model_dump(
                mode="json", exclude_none=True, exclude={"enabled_in_config"}
            )
            _restore_recipient_chat_ids(serialized_notifications, body.notifications)
            data["notifications"] = serialized_notifications
            candidate_raw = _dump_yaml(data)
            config = FrigateConfig.parse(candidate_raw)

            backup_dir = os.path.join(CONFIG_DIR, "config-backups")
            os.makedirs(backup_dir, mode=0o700, exist_ok=True)
            backup_name = datetime.datetime.now(datetime.UTC).strftime(
                "config-%Y%m%d-%H%M%S-%f.yaml"
            )
            backup_path = os.path.join(backup_dir, backup_name)
            with open(backup_path, "w", encoding="utf-8", newline="\n") as backup:
                backup.write(old_raw)
            try:
                with open(config_file, "w", encoding="utf-8", newline="\n") as file:
                    file.write(candidate_raw)
                    file.flush()
                    os.fsync(file.fileno())
            except Exception:
                shutil.copyfile(backup_path, config_file)
                raise

            swap_runtime_config(request.app, config)
            request.app.config_publisher.publisher.publish(
                "config/notifications", config.notifications
            )
            return {
                "success": True,
                "revision": _config_revision(candidate_raw),
                "notifications": serialized_notifications,
            }
    except Timeout:
        return JSONResponse(
            {"success": False, "message": "Configuration is currently being updated"},
            status_code=503,
        )
    except ValidationError as error:
        return JSONResponse(
            {
                "success": False,
                "message": "Invalid notification configuration",
                "errors": error.errors(),
            },
            status_code=400,
        )
    except Exception:
        logger.exception("Unable to save notification configuration")
        return JSONResponse(
            {"success": False, "message": "Unable to save notification configuration"},
            status_code=500,
        )


@router.post(
    "/notifications/rules/preview",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Preview a notification rule",
)
def preview_notification_rule(request: Request, body: NotificationRulePreviewRequest):
    try:
        rule = NotificationRuleConfig.model_validate(body.rule)
    except ValidationError as error:
        return JSONResponse(
            {"success": False, "message": "Invalid rule", "errors": error.errors()},
            status_code=400,
        )
    cameras = rule.filters.cameras or sorted(request.app.frigate_config.cameras)
    unknown_cameras = sorted(set(cameras) - set(request.app.frigate_config.cameras))
    return {
        "success": not unknown_cameras,
        "cameras": cameras,
        "unknown_cameras": unknown_cameras,
        "destinations": rule.destinations.model_dump(mode="json"),
        "providers": (
            _notification_client(request).provider_status()
            if _notification_client(request)
            else {}
        ),
    }


@router.post(
    "/notifications/rules/{rule_id}/test",
    dependencies=[Depends(require_role(["admin"]))],
    status_code=202,
    summary="Test a saved notification rule",
)
def test_notification_rule(request: Request, rule_id: str):
    client = _notification_client(request)
    deliveries = client.enqueue_rule_test(rule_id) if client else None
    if deliveries is None:
        return JSONResponse(
            {"success": False, "message": "Rule is missing or disabled"},
            status_code=404,
        )
    if not deliveries:
        return JSONResponse(
            {"success": False, "message": "No destination is ready"},
            status_code=409,
        )
    return {"success": True, "delivery_ids": deliveries}


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
    if not notifications_enabled or not config.notifications.channels.webpush.enabled:
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
