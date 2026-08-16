"""Tests for social notification provider HTTP contracts."""

import asyncio
import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import httpx

from frigate.application.notifications.envelope import NotificationEnvelope
from frigate.application.notifications.providers import (
    TelegramProvider,
    ZaloProvider,
    classify_response,
)
from frigate.infrastructure.config.camera.notification import (
    NotificationRecipientConfig,
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
        media_artifact_id="event-1",
        lpr_plate="51A12345",
        lpr_score=0.91,
    )


class TestNotificationProviders(unittest.TestCase):
    def test_environment_token_names_are_supported(self):
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "ZALO_BOT_TOKEN": "zalo-token",
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

    def test_provider_media_fetch_failures_are_retryable(self):
        telegram = httpx.Response(
            400,
            json={"description": "Bad Request: failed to get HTTP URL content"},
        )
        zalo = httpx.Response(
            200,
            json={"ok": False, "message": "temporary photo fetch failure"},
        )

        self.assertTrue(classify_response(telegram).retryable)
        zalo_result = classify_response(zalo)
        self.assertTrue(zalo_result.retryable)
        self.assertEqual(zalo_result.error, "temporary photo fetch failure")

    def test_telegram_uses_event_snapshot_url_without_artifact(self):
        requests: list[httpx.Request] = []

        async def run_test():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: (
                        requests.append(request)
                        or httpx.Response(200, json={"ok": True})
                    )
                )
            ) as client:
                return await TelegramProvider().deliver(
                    client,
                    NotificationRecipientConfig(
                        id="ops", name="Operators", chat_id="123"
                    ),
                    replace(
                        envelope(),
                        media_artifact_id=None,
                        snapshot_url="https://camera.example.com/api/events/event-1/snapshot.jpg",
                    ),
                    "https://camera.example.com",
                    300,
                )

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", requests[0].url.path)
        self.assertIn("snapshot.jpg", requests[0].content.decode())

    def test_telegram_uses_multipart_send_photo(self):
        request_seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            request_seen.append(request)
            return httpx.Response(200, json={"ok": True})

        async def run_test():
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with patch(
                    "frigate.application.notifications.providers.load_snapshot",
                    return_value=b"jpeg",
                ):
                    return await TelegramProvider().deliver(
                        client,
                        NotificationRecipientConfig(
                            id="ops", name="Operators", chat_id="123"
                        ),
                        envelope(),
                    )

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", request_seen[0].url.path)
        self.assertTrue(
            request_seen[0].headers["content-type"].startswith("multipart/form-data")
        )
        self.assertNotIn("secret", str(request_seen[0].content))

    def test_telegram_uses_signed_frigate_url_for_edge_artifact(self):
        requests: list[httpx.Request] = []
        signer = MagicMock()
        signer.url.return_value = "https://camera.example.com/signed-edge-snapshot"
        edge_envelope = envelope()
        object.__setattr__(edge_envelope, "media_artifact_id", "snapshot-jpg-event-1")

        async def run_test():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: (
                        requests.append(request)
                        or httpx.Response(200, json={"ok": True})
                    )
                )
            ) as client:
                with patch(
                    "frigate.application.notifications.providers.load_snapshot",
                    return_value=None,
                ):
                    return await TelegramProvider(signer).deliver(
                        client,
                        NotificationRecipientConfig(
                            id="ops", name="Operators", chat_id="123"
                        ),
                        edge_envelope,
                        "https://camera.example.com",
                        300,
                    )

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", requests[0].url.path)
        self.assertIn("signed-edge-snapshot", requests[0].content.decode())
        signer.url.assert_called_once_with(
            "https://camera.example.com", "snapshot-jpg-event-1", 300
        )

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
                    "frigate.application.notifications.providers.load_snapshot",
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

        with patch.dict(os.environ, {"ZALO_BOT_TOKEN": "secret"}):
            result = asyncio.run(run_test())
        self.assertTrue(result.sent)
        self.assertIn("/sendPhoto", requests[0].url.path)
        self.assertIn("signed-snapshot", requests[0].content.decode())
        signer.url.assert_called_once_with("https://camera.example.com", "event-1", 300)
