"""Tests for native notification provider configuration."""

import os
import unittest

from frigate.api.notification import _restore_recipient_chat_ids
from frigate.infrastructure.config.camera.notification import (
    CameraNotificationConfig,
    NotificationConfig,
)
from frigate.util.config import migrate_notification_config_v2


class TestNativeNotificationConfig(unittest.TestCase):
    def test_chat_id_env_placeholder_is_preserved_on_save(self):
        serialized = {
            "channels": {
                "telegram": {"recipients": [{"id": "ops", "chat_id": "resolved-value"}]}
            }
        }
        requested = {
            "channels": {
                "telegram": {
                    "recipients": [
                        {"id": "ops", "chat_id": "{TELEGRAM_CHAT_ID}"}
                    ]
                }
            }
        }
        _restore_recipient_chat_ids(serialized, requested)
        self.assertEqual(
            serialized["channels"]["telegram"]["recipients"][0]["chat_id"],
            "{TELEGRAM_CHAT_ID}",
        )

    def test_camera_defaults_to_webpush_only(self):
        config = CameraNotificationConfig()
        self.assertEqual(config.providers, ["webpush"])

    def test_rule_and_channel_contract(self):
        config = NotificationConfig.model_validate(
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "recipients": [
                            {
                                "id": "security_team",
                                "name": "Security Team",
                                "chat_id": "-100123",
                            }
                        ],
                    },
                    "zalo": {
                        "enabled": True,
                        "public_base_url": "https://camera.example.com",
                        "recipients": [],
                    },
                },
                "rules": [
                    {
                        "id": "car_alert",
                        "name": "Car alert",
                        "event": "alert",
                        "filters": {"cameras": ["car_camera"]},
                        "destinations": {"telegram": ["security_team"]},
                    }
                ],
            }
        )
        self.assertEqual(config.channels.telegram.recipients[0].id, "security_team")
        self.assertEqual(config.rules[0].filters.cameras, ["car_camera"])
        self.assertEqual(config.delivery.max_pending, 5000)

    def test_tokens_are_not_part_of_config_dump(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "do-not-serialize"
        os.environ["ZALO_BOT_TOKEN"] = "do-not-serialize-either"
        dumped = str(NotificationConfig().model_dump())
        self.assertNotIn("do-not-serialize", dumped)

    def test_recipient_ids_must_be_unique(self):
        with self.assertRaises(ValueError):
            NotificationConfig.model_validate(
                {
                    "channels": {
                        "telegram": {
                            "recipients": [
                                {"id": "ops", "name": "One", "chat_id": "1"},
                                {"id": "ops", "name": "Two", "chat_id": "2"},
                            ]
                        }
                    }
                }
            )

    def test_rule_rejects_unknown_recipient(self):
        with self.assertRaises(ValueError):
            NotificationConfig.model_validate(
                {
                    "rules": [
                        {
                            "id": "bad",
                            "name": "Bad",
                            "event": "alert",
                            "destinations": {"telegram": ["missing"]},
                        }
                    ]
                }
            )

    def test_legacy_routing_migrates_without_enabling_new_sources(self):
        raw = {
            "notifications": {
                "enabled": True,
                "cooldown": 10,
                "providers": {
                    "webpush": {"enabled": True},
                    "telegram": {
                        "enabled": True,
                        "recipients": [
                            {
                                "id": "ops",
                                "name": "Ops",
                                "chat_id": "1",
                                "cameras": ["car_camera"],
                            }
                        ],
                    },
                },
            },
            "cameras": {
                "car_camera": {
                    "notifications": {
                        "enabled": True,
                        "cooldown": 30,
                        "providers": ["webpush", "telegram"],
                    }
                }
            },
        }
        self.assertTrue(migrate_notification_config_v2(raw))
        notifications = raw["notifications"]
        self.assertEqual(notifications["schema_version"], 2)
        self.assertNotIn("notifications", raw["cameras"]["car_camera"])
        self.assertEqual(
            {rule["event"] for rule in notifications["rules"]},
            {"alert", "semantic_trigger", "camera_monitoring", "license_plate"},
        )
        self.assertTrue(all(rule["cooldown"] == 30 for rule in notifications["rules"]))
