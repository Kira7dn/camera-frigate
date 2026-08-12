"""Tests for normalized native notification policy inputs."""

import unittest

from frigate.infrastructure.config.camera.notification import NotificationRuleConfig
from frigate.application.notifications.client import NotificationClient, normalize_plate
from frigate.application.notifications.envelope import NotificationEnvelope


class TestNotificationClient(unittest.TestCase):
    def setUp(self):
        self.client = NotificationClient.__new__(NotificationClient)
        self.client._alert_plates = {}
        self.client._lpr_updates = {}

    def test_normalize_plate(self):
        self.assertEqual(normalize_plate("51a-123.45"), "51A12345")
        self.assertIsNone(normalize_plate(""))
        self.assertIsNone(normalize_plate(3789.091994191706))
        self.assertIsNone(normalize_plate(("3789", 0.91)))

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

    def test_lpr_update_supplies_plate_box_and_sub_label(self):
        self.client._remember_lpr_update(
            {
                "id": "event-2",
                "plate": "51A12345",
                "score": 0.95,
                "plate_box": [10, 20, 100, 60],
                "name": "company_car",
            }
        )
        envelope = self.client._lpr_envelope(
            {
                "type": "end",
                "after": {
                    "id": "event-2",
                    "camera": "car_camera",
                    "label": "car",
                    "data": {},
                },
            }
        )
        self.assertEqual(envelope.lpr_plate_box, [10, 20, 100, 60])
        self.assertEqual(envelope.sub_label, "company_car")

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

    def test_rule_matches_camera_label_and_zone(self):
        rule = NotificationRuleConfig.model_validate(
            {
                "id": "car_gate",
                "name": "Car gate",
                "event": "object_detected",
                "filters": {
                    "cameras": ["car_camera"],
                    "labels": ["car"],
                    "zones": ["gate"],
                },
            }
        )
        envelope = NotificationEnvelope(
            id="1",
            source_type="object",
            source_id="event-1",
            camera="car_camera",
            timestamp=1,
            title="Car",
            message="Detected",
            direct_url="/",
            snapshot_ref="event-1",
            notification_type="object_detected",
            object_label="car",
            genai={"_zones": ["gate"]},
        )
        self.assertTrue(self.client._rule_matches(rule, envelope))
        self.assertFalse(
            self.client._rule_matches(
                rule,
                envelope.__class__.from_dict({**envelope.as_dict(), "camera": "other"}),
            )
        )
