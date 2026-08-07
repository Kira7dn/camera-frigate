"""Social notification provider orchestration."""

import datetime
from dataclasses import replace
from multiprocessing.synchronize import Event as MpEvent
from typing import Any

import httpx

from frigate.config import FrigateConfig
from frigate.config.camera.notification import NotificationRecipientConfig
from frigate.models import NotificationDelivery

from .envelope import NotificationEnvelope
from .media import NotificationMediaSigner
from .metrics import increment
from .metrics import snapshot as metrics_snapshot
from .outbox import NotificationOutbox
from .providers import DeliveryResult, TelegramProvider, ZaloProvider


class SocialClient:
    """Fan out social messages through a durable outbox."""

    def __init__(self, config: FrigateConfig, stop_event: MpEvent) -> None:
        self.config = config
        self.signer = NotificationMediaSigner()
        self.telegram = TelegramProvider()
        self.zalo = ZaloProvider(self.signer)
        self.providers = {
            "telegram": self.telegram,
            "zalo": self.zalo,
        }
        self.outbox = NotificationOutbox(
            lambda: self.config.notifications.delivery,
            self._deliver,
            self.recipient_enabled,
            stop_event,
        )

    def _provider_config(self, provider: str):
        return getattr(self.config.notifications.providers, provider)

    def recipient(
        self, provider: str, recipient_id: str
    ) -> NotificationRecipientConfig | None:
        if provider not in self.providers:
            return None
        return next(
            (
                recipient
                for recipient in self._provider_config(provider).recipients
                if recipient.id == recipient_id
            ),
            None,
        )

    def recipient_enabled(
        self, provider: str, recipient_id: str, camera: str | None
    ) -> bool:
        if provider not in self.providers:
            return False
        provider_config = self._provider_config(provider)
        recipient = self.recipient(provider, recipient_id)
        if (
            not provider_config.enabled
            or not self.providers[provider].configured
            or recipient is None
            or not recipient.enabled
        ):
            return False
        if camera and recipient.cameras and camera not in recipient.cameras:
            return False
        if camera:
            camera_config = self.config.cameras.get(camera)
            if (
                camera_config is None
                or not camera_config.notifications.enabled
                or provider not in camera_config.notifications.providers
            ):
                return False
        return True

    def enqueue(self, envelope: NotificationEnvelope) -> list[str]:
        delivery_ids: list[str] = []
        dedupe_envelope = envelope
        if (
            envelope.source_type == "review"
            and envelope.lpr_plate
            and envelope.snapshot_ref
        ):
            dedupe_envelope = replace(
                envelope, source_type="lpr", source_id=envelope.snapshot_ref
            )
        if not envelope.camera:
            camera_providers = {"telegram", "zalo"}
        else:
            camera_providers = set(
                self.config.cameras[envelope.camera].notifications.providers
            )
        for provider in self.providers:
            if provider not in camera_providers:
                continue
            for recipient in self._provider_config(provider).recipients:
                if not self.recipient_enabled(provider, recipient.id, envelope.camera):
                    continue
                delivery_id = self.outbox.enqueue(
                    provider, recipient.id, dedupe_envelope
                )
                if delivery_id:
                    delivery_ids.append(delivery_id)
        return delivery_ids

    def cancel_disabled(self) -> int:
        """Cancel queued work whose provider, recipient, or camera was disabled."""
        now = datetime.datetime.now(datetime.UTC)
        cancelled = 0
        pending = NotificationDelivery.select().where(
            NotificationDelivery.status == "pending"
        )
        for delivery in pending:
            camera = (delivery.payload or {}).get("camera")
            if self.recipient_enabled(delivery.provider, delivery.recipient_id, camera):
                continue
            changed = (
                NotificationDelivery.update(
                    status="cancelled",
                    completed_at=now,
                    updated_at=now,
                    last_error="Provider or recipient disabled",
                )
                .where(
                    (NotificationDelivery.id == delivery.id)
                    & (NotificationDelivery.status == "pending")
                )
                .execute()
            )
            cancelled += changed
            if changed:
                increment(delivery.provider, "cancelled")
        return cancelled

    async def _deliver(
        self,
        client: httpx.AsyncClient,
        provider: str,
        recipient_id: str,
        envelope: NotificationEnvelope,
    ) -> DeliveryResult:
        recipient = self.recipient(provider, recipient_id)
        if recipient is None:
            return DeliveryResult(False, False, "Recipient no longer exists")
        if provider == "telegram":
            return await self.telegram.deliver(client, recipient, envelope)
        zalo_config = self.config.notifications.providers.zalo
        return await self.zalo.deliver(
            client,
            recipient,
            envelope,
            str(zalo_config.public_base_url) if zalo_config.public_base_url else None,
            zalo_config.media_url_ttl,
        )

    def enqueue_test(self, provider: str, recipient_id: str) -> str | None:
        if not self.recipient_enabled(provider, recipient_id, None):
            return None
        now = datetime.datetime.now(datetime.UTC).timestamp()
        envelope = NotificationEnvelope(
            id=f"test-{now}",
            source_type="test",
            source_id=f"{recipient_id}-{now}",
            camera=None,
            timestamp=now,
            title="Test Notification",
            message="This is a test notification from Frigate.",
            direct_url="/",
            snapshot_ref=None,
            notification_type="test",
        )
        return self.outbox.enqueue(provider, recipient_id, envelope)

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        metric_values = metrics_snapshot()
        for provider, adapter in self.providers.items():
            provider_config = self._provider_config(provider)
            deliveries = NotificationDelivery.select().where(
                NotificationDelivery.provider == provider
            )
            pending = deliveries.where(
                NotificationDelivery.status << ("pending", "processing")
            ).count()
            last_success = (
                deliveries.where(NotificationDelivery.status == "sent")
                .order_by(NotificationDelivery.completed_at.desc())
                .first()
            )
            last_error = (
                deliveries.where(NotificationDelivery.status == "failed")
                .order_by(NotificationDelivery.completed_at.desc())
                .first()
            )
            degraded = provider == "zalo" and not provider_config.public_base_url
            result[provider] = {
                "enabled": provider_config.enabled,
                "configured": adapter.configured,
                "readiness": (
                    "missing"
                    if not adapter.configured
                    else "degraded"
                    if degraded
                    else "ready"
                ),
                "pending": pending,
                "last_success": last_success.completed_at if last_success else None,
                "last_error": last_error.last_error if last_error else None,
                "metrics": metric_values.get(provider, {}),
            }
        return result

    def stop(self) -> None:
        self.outbox.stop()
