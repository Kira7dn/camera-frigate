"""Tests for native notification provider configuration."""

import os
import unittest

from frigate.config.camera.notification import (
    CameraNotificationConfig,
    NotificationConfig,
)


class TestNativeNotificationConfig(unittest.TestCase):
    def test_camera_defaults_to_webpush_only(self):
        config = CameraNotificationConfig()
        self.assertEqual(config.providers, ["webpush"])

    def test_social_provider_contract(self):
        config = NotificationConfig.model_validate(
            {
                "providers": {
                    "telegram": {
                        "enabled": True,
                        "recipients": [
                            {
                                "id": "security_team",
                                "name": "Security Team",
                                "chat_id": "-100123",
                                "cameras": ["car_camera"],
                            }
                        ],
                    },
                    "zalo": {
                        "enabled": True,
                        "public_base_url": "https://camera.example.com",
                        "recipients": [],
                    },
                }
            }
        )
        self.assertEqual(
            config.providers.telegram.recipients[0].cameras, ["car_camera"]
        )
        self.assertEqual(config.delivery.max_pending, 5000)

    def test_tokens_are_not_part_of_config_dump(self):
        os.environ["FRIGATE_TELEGRAM_BOT_TOKEN"] = "do-not-serialize"
        os.environ["FRIGATE_ZALO_BOT_TOKEN"] = "do-not-serialize-either"
        dumped = str(NotificationConfig().model_dump())
        self.assertNotIn("do-not-serialize", dumped)

    def test_recipient_ids_must_be_unique(self):
        with self.assertRaises(ValueError):
            NotificationConfig.model_validate(
                {
                    "providers": {
                        "telegram": {
                            "recipients": [
                                {"id": "ops", "name": "One", "chat_id": "1"},
                                {"id": "ops", "name": "Two", "chat_id": "2"},
                            ]
                        }
                    }
                }
            )
