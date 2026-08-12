"""Tests for signed social notification media URLs."""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from frigate.api.notification import notification_snapshot
from frigate.application.notifications.envelope import NotificationEnvelope
from frigate.application.notifications.media import NotificationMediaSigner


class TestNotificationMediaSigner(unittest.TestCase):
    def test_signature_binds_event_and_expiry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            signer = NotificationMediaSigner(Path(temp_dir) / "key")
            url = signer.url("https://camera.example.com", "event-1", 300)
            query = parse_qs(urlparse(url).query)
            expires = int(query["expires"][0])
            signature = query["signature"][0]
            self.assertTrue(signer.verify("event-1", expires, signature))
            self.assertFalse(signer.verify("event-2", expires, signature))
            self.assertFalse(signer.verify("event-1", expires + 1, signature))
            self.assertNotEqual(
                url, signer.url("https://camera.example.com", "event-1", 300)
            )

    def test_expired_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            signer = NotificationMediaSigner(Path(temp_dir) / "key")
            expires = int(time.time()) - 1
            self.assertFalse(
                signer.verify("event-1", expires, signer.signature("event-1", expires))
            )

    def test_envelope_never_falls_back_to_event_snapshot(self):
        envelope = NotificationEnvelope(
            id="notification-1",
            source_type="event_revision",
            source_id="event-1",
            camera="car_camera",
            timestamp=1.0,
            title="title",
            message="message",
            direct_url="",
            snapshot_ref="event-1",
            notification_type="alert",
        )
        self.assertIsNone(envelope.artifact_ref)

    def test_signed_route_fetches_private_edge_artifact(self):
        class Runtime:
            async def fetch_media(self, node_id, media_id):
                self.request = (node_id, media_id)
                return b"edge-jpeg"

        runtime = Runtime()
        signer = MagicMock()
        signer.verify.return_value = True
        request = SimpleNamespace(
            app=SimpleNamespace(
                dispatcher=SimpleNamespace(
                    notification_client=SimpleNamespace(
                        social=SimpleNamespace(signer=signer)
                    )
                ),
                tracker_maintainer=runtime,
            )
        )
        manifest = SimpleNamespace(node_id="edge-local", media_id="snapshot-jpg-1")
        with (
            patch("frigate.api.notification.load_snapshot", return_value=None),
            patch("frigate.api.notification.resolve_media_id", return_value=manifest),
        ):
            response = asyncio.run(
                notification_snapshot(request, "snapshot-jpg-1", 123, "signature")
            )
        self.assertEqual(response.body, b"edge-jpeg")
        self.assertEqual(runtime.request, ("edge-local", "snapshot-jpg-1"))
        signer.verify.assert_called_once_with("snapshot-jpg-1", 123, "signature")
