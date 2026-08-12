"""Tests for social recipient filtering and LPR passage deduplication."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from frigate.infrastructure.config.camera.notification import (
    NotificationDestinationsConfig,
    NotificationRecipientConfig,
)
from frigate.application.notifications.envelope import NotificationEnvelope
from frigate.application.notifications.social import SocialClient


class TestNotificationSocialClient(unittest.TestCase):
    def setUp(self):
        self.client = SocialClient.__new__(SocialClient)
        self.telegram = SimpleNamespace(configured=True)
        self.zalo = SimpleNamespace(configured=True)
        self.client.providers = {
            "telegram": self.telegram,
            "zalo": self.zalo,
        }
        self.recipient = NotificationRecipientConfig(
            id="ops",
            name="Operators",
            chat_id="123",
        )
        self.client.config = SimpleNamespace(
            notifications=SimpleNamespace(
                channels=SimpleNamespace(
                    telegram=SimpleNamespace(enabled=True, recipients=[self.recipient]),
                    zalo=SimpleNamespace(enabled=False, recipients=[]),
                ),
                rules=[],
            ),
            cameras={
                "car_camera": SimpleNamespace(),
                "other_camera": SimpleNamespace(),
            },
        )
        self.client.outbox = MagicMock()
        self.client.outbox.enqueue.return_value = "delivery-1"

    @staticmethod
    def envelope() -> NotificationEnvelope:
        return NotificationEnvelope(
            id="review-1",
            source_type="review",
            source_id="review-1",
            camera="car_camera",
            timestamp=1,
            title="Vehicle",
            message="Detected",
            direct_url="/review?id=review-1",
            snapshot_ref="event-1",
            notification_type="alert",
            lpr_plate="51A12345",
        )

    def test_recipient_and_channel_enabled(self):
        self.assertTrue(self.client.recipient_enabled("telegram", "ops", "car_camera"))
        self.assertFalse(self.client.recipient_enabled("zalo", "ops", "car_camera"))

    def test_review_with_plate_uses_lpr_passage_dedupe_key(self):
        self.assertEqual(
            self.client.enqueue(
                self.envelope(),
                NotificationDestinationsConfig(telegram=["ops"]),
            ),
            ["delivery-1"],
        )
        queued_envelope = self.client.outbox.enqueue.call_args.args[2]
        self.assertEqual(queued_envelope.source_type, "lpr")
        self.assertEqual(queued_envelope.source_id, "event-1")
