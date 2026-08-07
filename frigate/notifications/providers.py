"""Telegram and Zalo notification provider adapters."""

import datetime
import os
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from frigate.config.camera.notification import NotificationRecipientConfig

from .envelope import NotificationEnvelope
from .media import NotificationMediaSigner, load_snapshot


@dataclass(frozen=True)
class DeliveryResult:
    sent: bool
    retryable: bool = False
    error: str | None = None
    retry_after: float | None = None


def _provider_token(primary_name: str, legacy_name: str) -> str:
    """Read a provider token without exposing it through config or status APIs."""
    return os.getenv(primary_name, "").strip() or os.getenv(legacy_name, "").strip()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(
                0.0,
                parsedate_to_datetime(value).timestamp()
                - datetime.datetime.now(datetime.UTC).timestamp(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


def classify_response(response: httpx.Response) -> DeliveryResult:
    """Classify an HTTP provider response for outbox retry handling."""
    if 200 <= response.status_code < 300:
        try:
            body: Any = response.json()
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("ok") is False:
            return DeliveryResult(False, False, "provider rejected request")
        return DeliveryResult(True)
    retryable = response.status_code in (408, 425, 429) or response.status_code >= 500
    return DeliveryResult(
        False,
        retryable,
        f"HTTP {response.status_code}",
        _retry_after(response),
    )


def envelope_text(envelope: NotificationEnvelope) -> str:
    text = f"{envelope.title}\n{envelope.message}"
    shown_label = envelope.sub_label or envelope.lpr_plate or envelope.object_label
    if envelope.lpr_plate and envelope.lpr_plate != shown_label:
        text += f"\nBiển số: {envelope.lpr_plate}"
    if envelope.direct_url:
        text += f"\n{envelope.direct_url}"
    return text


class TelegramProvider:
    name = "telegram"

    @property
    def configured(self) -> bool:
        return bool(_provider_token("FRIGATE_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"))

    async def deliver(
        self,
        client: httpx.AsyncClient,
        recipient: NotificationRecipientConfig,
        envelope: NotificationEnvelope,
    ) -> DeliveryResult:
        token = _provider_token("FRIGATE_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
        if not token:
            return DeliveryResult(False, False, "Telegram token is missing")
        base_url = f"https://api.telegram.org/bot{token}"
        text = envelope_text(envelope)
        artifact_ref = envelope.artifact_ref
        snapshot = load_snapshot(artifact_ref) if artifact_ref else None
        if artifact_ref and snapshot is None:
            return DeliveryResult(False, True, "Canonical artifact is not available")
        try:
            if snapshot:
                response = await client.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": recipient.chat_id, "caption": text[:1024]},
                    files={"photo": ("snapshot.jpg", snapshot, "image/jpeg")},
                )
            else:
                response = await client.post(
                    f"{base_url}/sendMessage",
                    json={"chat_id": recipient.chat_id, "text": text[:4096]},
                )
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            return DeliveryResult(False, True, type(error).__name__)
        return classify_response(response)


class ZaloProvider:
    name = "zalo"

    def __init__(self, signer: NotificationMediaSigner) -> None:
        self.signer = signer

    @property
    def configured(self) -> bool:
        return bool(_provider_token("FRIGATE_ZALO_BOT_TOKEN", "ZALO_BOT_TOKEN"))

    async def deliver(
        self,
        client: httpx.AsyncClient,
        recipient: NotificationRecipientConfig,
        envelope: NotificationEnvelope,
        public_base_url: str | None,
        media_url_ttl: int,
    ) -> DeliveryResult:
        token = _provider_token("FRIGATE_ZALO_BOT_TOKEN", "ZALO_BOT_TOKEN")
        if not token:
            return DeliveryResult(False, False, "Zalo token is missing")
        base_url = f"https://bot-api.zaloplatforms.com/bot{token}"
        text = envelope_text(envelope)
        payload: dict[str, Any] = {"chat_id": recipient.chat_id}
        endpoint = "sendMessage"
        artifact_ref = envelope.artifact_ref
        if public_base_url and artifact_ref:
            endpoint = "sendPhoto"
            payload.update(
                {
                    "photo": self.signer.url(
                        public_base_url, artifact_ref, media_url_ttl
                    ),
                    "caption": text,
                }
            )
        else:
            payload["text"] = text
        try:
            response = await client.post(f"{base_url}/{endpoint}", json=payload)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            return DeliveryResult(False, True, type(error).__name__)
        return classify_response(response)
