"""Tests for durable notification outbox bounds and deduplication."""

import unittest
from types import SimpleNamespace

from playhouse.sqlite_ext import SqliteExtDatabase

from frigate.models import NotificationDelivery
from frigate.notifications.envelope import NotificationEnvelope
from frigate.notifications.outbox import NotificationOutbox


class TestNotificationOutbox(unittest.TestCase):
    def setUp(self):
        self.database = SqliteExtDatabase(":memory:")
        self.database.bind([NotificationDelivery])
        self.database.create_tables([NotificationDelivery])
        self.outbox = NotificationOutbox.__new__(NotificationOutbox)
        self.outbox.delivery_config = lambda: SimpleNamespace(max_pending=1)

    def tearDown(self):
        self.database.drop_tables([NotificationDelivery])
        self.database.close()

    @staticmethod
    def envelope(source_id: str) -> NotificationEnvelope:
        return NotificationEnvelope(
            id=source_id,
            source_type="lpr",
            source_id=source_id,
            camera="car_camera",
            timestamp=1.0,
            title="Vehicle",
            message="Passage ended",
            direct_url="/",
            snapshot_ref=source_id,
            notification_type="lpr",
        )

    def test_unique_delivery_and_bounded_queue(self):
        first = self.outbox.enqueue("telegram", "ops", self.envelope("event-1"))
        duplicate = self.outbox.enqueue("telegram", "ops", self.envelope("event-1"))
        full = self.outbox.enqueue("telegram", "ops", self.envelope("event-2"))
        self.assertIsNotNone(first)
        self.assertIsNone(duplicate)
        self.assertIsNone(full)
        self.assertEqual(NotificationDelivery.select().count(), 1)

    def test_claim_moves_only_one_due_delivery_to_processing(self):
        self.outbox.delivery_config = lambda: SimpleNamespace(max_pending=2)
        first_id = self.outbox.enqueue("telegram", "ops", self.envelope("event-1"))
        second_id = self.outbox.enqueue("telegram", "ops", self.envelope("event-2"))
        claimed = self.outbox._claim()
        self.assertIn(claimed.id, (first_id, second_id))
        self.assertEqual(claimed.status, "processing")
        self.assertEqual(
            NotificationDelivery.select()
            .where(NotificationDelivery.status == "processing")
            .count(),
            1,
        )
