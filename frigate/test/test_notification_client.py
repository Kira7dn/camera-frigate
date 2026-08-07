"""Tests for normalized native notification policy inputs."""

import unittest
from types import SimpleNamespace

from frigate.notifications.client import NotificationClient, normalize_plate


class TestNotificationClient(unittest.TestCase):
    def setUp(self):
        self.client = NotificationClient.__new__(NotificationClient)

    def test_normalize_plate(self):
        self.assertEqual(normalize_plate("51a-123.45"), "51A12345")
        self.assertIsNone(normalize_plate(""))

    def test_lpr_final_car_event_builds_complete_envelope(self):
        envelope = self.client._lpr_envelope(
            {
                "type": "end",
                "after": {
                    "id": "event-1",
                    "camera": "car_camera",
                    "label": "car",
                    "end_time": 123.0,
                    "data": {
                        "recognized_license_plate": "51A-123.45",
                        "recognized_license_plate_score": 0.91,
                        "license_plate_box": [1, 2, 3, 4],
                    },
                },
            }
        )
        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.source_id, "event-1")
        self.assertEqual(envelope.camera, "car_camera")
        self.assertEqual(envelope.lpr_plate, "51A12345")
        self.assertEqual(envelope.lpr_score, 0.91)

    def test_lpr_ignores_updates_non_cars_and_missing_plates(self):
        base = {
            "type": "end",
            "after": {
                "id": "event-1",
                "camera": "car_camera",
                "label": "car",
                "data": {},
            },
        }
        self.assertIsNone(self.client._lpr_envelope(base))
        base["after"]["data"] = {"recognized_license_plate": "51A12345"}
        base["after"]["label"] = "person"
        self.assertIsNone(self.client._lpr_envelope(base))
        base["after"]["label"] = "car"
        base["type"] = "update"
        self.assertIsNone(self.client._lpr_envelope(base))

    def test_camera_provider_selection_defaults_are_enforced(self):
        self.client.config = SimpleNamespace(
            cameras={
                "car_camera": SimpleNamespace(
                    notifications=SimpleNamespace(
                        enabled=True,
                        providers=["webpush"],
                    )
                )
            }
        )
        self.client.suspended_cameras = {"car_camera": 0}
        self.assertTrue(self.client._eligible("car_camera", "webpush"))
        self.assertFalse(self.client._eligible("car_camera", "telegram"))
