"""SQLite-backed notification delivery outbox."""

import asyncio
import datetime
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from multiprocessing.synchronize import Event as MpEvent

import httpx
from peewee import IntegrityError

from frigate.config.camera.notification import NotificationDeliveryConfig
from frigate.models import MediaArtifact, NotificationDelivery

from .envelope import NotificationEnvelope
from .metrics import increment, observe_latency, set_queue_depth
from .providers import DeliveryResult

logger = logging.getLogger(__name__)

DeliveryCallback = Callable[
    [httpx.AsyncClient, str, str, NotificationEnvelope],
    Awaitable[DeliveryResult],
]
RecipientEnabledCallback = Callable[[str, str, str | None, str | None], bool]


class NotificationOutbox:
    """Persist, claim, retry, and retain social deliveries."""

    def __init__(
        self,
        delivery_config: Callable[[], NotificationDeliveryConfig],
        deliver: DeliveryCallback,
        recipient_enabled: RecipientEnabledCallback,
        stop_event: MpEvent,
    ) -> None:
        self.delivery_config = delivery_config
        self.deliver = deliver
        self.recipient_enabled = recipient_enabled
        self.stop_event = stop_event
        self._thread = threading.Thread(target=self._run, daemon=True)
        now = datetime.datetime.now(datetime.UTC)
        NotificationDelivery.update(
            status="pending", next_attempt=now, updated_at=now
        ).where(NotificationDelivery.status == "processing").execute()
        for provider in ("telegram", "zalo"):
            self._update_depth(provider)
        self._thread.start()

    def enqueue(
        self,
        provider: str,
        recipient_id: str,
        envelope: NotificationEnvelope,
    ) -> str | None:
        pending = (
            NotificationDelivery.select()
            .where(NotificationDelivery.status << ("pending", "processing"))
            .count()
        )
        if pending >= self.delivery_config().max_pending:
            increment(provider, "rejected_queue_full")
            logger.warning(
                "Notification outbox is full at %d pending deliveries", pending
            )
            return None
        now = datetime.datetime.now(datetime.UTC)
        delivery_id = str(uuid.uuid4())
        try:
            NotificationDelivery.create(
                id=delivery_id,
                provider=provider,
                recipient_id=recipient_id,
                rule_id=envelope.rule_id or "legacy",
                source_type=envelope.source_type,
                source_id=envelope.source_id,
                payload=envelope.as_dict(),
                intent_id=envelope.facts.get("intent_id"),
                media_artifact_id=envelope.artifact_ref,
                status="pending",
                attempts=0,
                next_attempt=now,
                created_at=now,
                updated_at=now,
            )
            if envelope.artifact_ref:
                MediaArtifact.update(pinned=True).where(
                    MediaArtifact.id == envelope.artifact_ref
                ).execute()
        except IntegrityError:
            increment(provider, "deduplicated")
            return None
        increment(provider, "queued")
        self._update_depth(provider)
        return delivery_id

    @staticmethod
    def _update_depth(provider: str) -> None:
        depth = (
            NotificationDelivery.select()
            .where(
                (NotificationDelivery.provider == provider)
                & (NotificationDelivery.status << ("pending", "processing"))
            )
            .count()
        )
        set_queue_depth(provider, depth)

    def _claim(self) -> NotificationDelivery | None:
        now = datetime.datetime.now(datetime.UTC)
        database = NotificationDelivery._meta.database
        # One UPDATE statement is the transaction boundary. This works with
        # Frigate's queued SQLite writer and remains safe if another claimant is
        # added later.
        cursor = database.execute_sql(
            """
            UPDATE notification_delivery
            SET status = 'processing', updated_at = ?
            WHERE id = (
                SELECT id FROM notification_delivery
                WHERE status = 'pending' AND next_attempt <= ?
                ORDER BY next_attempt, created_at
                LIMIT 1
            )
            AND status = 'pending'
            RETURNING id
            """,
            (now, now),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return NotificationDelivery.get_by_id(row[0])

    async def _process(
        self, client: httpx.AsyncClient, delivery: NotificationDelivery
    ) -> None:
        envelope = NotificationEnvelope.from_dict(delivery.payload)
        if not self.recipient_enabled(
            delivery.provider, delivery.recipient_id, envelope.camera, envelope.rule_id
        ):
            self._complete(delivery, "cancelled", "Provider or recipient disabled")
            increment(delivery.provider, "cancelled")
            self._update_depth(delivery.provider)
            return
        started = time.monotonic()
        result = await self.deliver(
            client, delivery.provider, delivery.recipient_id, envelope
        )
        observe_latency(delivery.provider, time.monotonic() - started)
        if result.sent:
            self._complete(delivery, "sent", None)
            increment(delivery.provider, "sent")
            self._update_depth(delivery.provider)
            return
        attempts = delivery.attempts + 1
        config = self.delivery_config()
        if not result.retryable or attempts >= config.max_attempts:
            delivery.attempts = attempts
            self._complete(delivery, "failed", result.error)
            increment(delivery.provider, "failed")
            self._update_depth(delivery.provider)
            return
        delay = result.retry_after
        if delay is None:
            delay = min(
                config.max_backoff,
                config.initial_backoff * (2 ** max(0, attempts - 1)),
            )
        now = datetime.datetime.now(datetime.UTC)
        (
            NotificationDelivery.update(
                status="pending",
                attempts=attempts,
                next_attempt=now + datetime.timedelta(seconds=delay),
                updated_at=now,
                last_error=result.error,
            )
            .where(NotificationDelivery.id == delivery.id)
            .execute()
        )
        increment(delivery.provider, "retry")

    @staticmethod
    def _complete(
        delivery: NotificationDelivery, status: str, error: str | None
    ) -> None:
        now = datetime.datetime.now(datetime.UTC)
        (
            NotificationDelivery.update(
                status=status,
                completed_at=now,
                updated_at=now,
                last_error=error,
                attempts=delivery.attempts,
            )
            .where(NotificationDelivery.id == delivery.id)
            .execute()
        )

    def _cleanup(self) -> None:
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=self.delivery_config().retention_days
        )
        NotificationDelivery.delete().where(
            (NotificationDelivery.status << ("sent", "failed", "cancelled"))
            & (NotificationDelivery.completed_at < cutoff)
        ).execute()

    async def _worker(self) -> None:
        last_cleanup = 0.0
        async with httpx.AsyncClient(timeout=20.0) as client:
            while not self.stop_event.is_set():
                delivery = self._claim()
                if delivery is None:
                    await asyncio.sleep(0.5)
                else:
                    try:
                        await self._process(client, delivery)
                    except Exception:
                        logger.exception(
                            "Unexpected notification delivery failure for %s",
                            delivery.id,
                        )
                        now = datetime.datetime.now(datetime.UTC)
                        NotificationDelivery.update(
                            status="pending", next_attempt=now, updated_at=now
                        ).where(NotificationDelivery.id == delivery.id).execute()
                if time.monotonic() - last_cleanup > 3600:
                    self._cleanup()
                    last_cleanup = time.monotonic()

    def _run(self) -> None:
        asyncio.run(self._worker())

    def stop(self) -> None:
        self._thread.join(timeout=5)
