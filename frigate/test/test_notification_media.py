"""Tests for signed social notification media URLs."""

import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from frigate.notifications.media import NotificationMediaSigner


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
