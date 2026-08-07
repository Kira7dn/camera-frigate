"""Tests for social notification provider HTTP contracts."""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from frigate.config.camera.notification import NotificationRecipientConfig
from frigate.notifications.envelope import NotificationEnvelope
from frigate.notifications.providers import (
    TelegramProvider,
    ZaloProvider,
    classify_response,
)


def envelope() -> NotificationEnvelope:
    return NotificationEnvelope(
        id="delivery",
        source_type="lpr",
        source_id="event-1",
        camera="car_camera",
        timestamp=1.0,
        title="Vehicle 51A12345",
        message="Vehicle passage ended",
        direct_url="/explore?event_id=event-1",
        snapshot_ref="event-1",
        notification_type="lpr",
        lpr_plate="51A12345",
        lpr_score=0.91,
    )


class TestNotificationProviders(unittest.TestCase):
    def test_legacy_environment_token_names_remain_supported(self):
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "legacy-telegram",
                "ZALO_BOT_TOKEN": "legacy-zalo",
            },
            clear=True,
        ):
            self.assertTrue(TelegramProvider().configured)
            self.assertTrue(ZaloProvider(MagicMock()).configured)

    def test_retry_classification_and_retry_after(self):
        response = httpx.Response(429, headers={"Retry-After": "12"})
        result = classify_response(response)
        self.assertTrue(result.retryable)
        self.assertEqual(result.retry_after, 12)
        self.assertFalse(classify_response(httpx.Response(400)).retryable)

    def test_telegram_uses_multipart_send_photo(self):
        request_seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_seen.append(request)
            return httpx.Response(200, json={"ok": True})

        async def run_test():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch(
                    "frigate.notifications.providers.load_snapshot",
                    return_value=b"jpeg",
                ):
                    return await TelegramProvider().deliver(
                        client,
                        NotificationRecipientConfig(
                            id="ops", name="Operators", chat_id="123"
                        ),
                        envelope(),
                    )

        with patch.dict(os.environ, {"FRIGATE_TELEGRAM_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", request_seen[0].url.path)
        self.assertTrue(
            request_seen[0].headers["content-type"].startswith("multipart/form-data")
        )
        self.assertNotIn("secret", str(request_seen[0].content))

    def test_zalo_uses_a_fresh_signed_snapshot_url(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"ok": True})

        signer = MagicMock()
        signer.url.return_value = "https://camera.example.com/signed-snapshot"

        async def run_test():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with patch(
                    "frigate.notifications.providers.load_snapshot",
                    return_value=b"jpeg",
                ):
                    return await ZaloProvider(signer).deliver(
                        client,
                        NotificationRecipientConfig(
                            id="ops", name="Operators", chat_id="123"
                        ),
                        envelope(),
                        "https://camera.example.com",
                        300,
                    )

        with patch.dict(os.environ, {"FRIGATE_ZALO_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", requests[0].url.path)
        self.assertIn("signed-snapshot", requests[0].content.decode())
        signer.url.assert_called_once_with("https://camera.example.com", "event-1", 300)
