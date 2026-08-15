"""Tests for normalized native notification policy inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frigate.application.notifications.client import NotificationClient, normalize_plate
from frigate.application.notifications.envelope import NotificationEnvelope
from frigate.infrastructure.config.camera.notification import NotificationRuleConfig


class TestNotificationClient(unittest.TestCase):
    def setUp(self):
        self.client = NotificationClient.__new__(NotificationClient)
        self.client._alert_plates = {}
        self.client._lpr_updates = {}
        self.client._aggregator = SimpleNamespace(
            observe=lambda **_kwargs: None,
            media=SimpleNamespace(get=lambda _artifact_id: None),
        )
        self.client.config = SimpleNamespace(
            notifications=SimpleNamespace(public_base_url=None)
        )
        self.event_lookup = patch(
            "frigate.application.notifications.client.Event.get_or_none",
            return_value=None,
        )
        self.event_lookup.start()
        self.addCleanup(self.event_lookup.stop)

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

    def test_smoking_review_matches_safety_alert_rule(self):
        self.client.config = SimpleNamespace(
            cameras={
                "safety_camera": SimpleNamespace(
                    friendly_name="Safety Smoking Camera"
                )
            }
        )
        envelope = self.client._review_envelope(
            {
                "after": {
                    "id": "review-smoking-1",
                    "camera": "safety_camera",
                    "severity": "alert",
                    "start_time": 123.0,
                    "data": {
                        "objects": ["smoking: camera-safety"],
                        "detections": ["event-smoking-1"],
                    },
                }
            }
        )
        rule = NotificationRuleConfig.model_validate(
            {
                "id": "smoking_alert",
                "name": "Smoking alert",
                "event": "alert",
                "filters": {
                    "cameras": ["safety_camera"],
                    "labels": ["smoking"],
                },
            }
        )

        self.assertIsNotNone(envelope)
        self.assertEqual(envelope.object_label, "smoking")
        self.assertTrue(self.client._rule_matches(rule, envelope))
        self.assertFalse(
            self.client._rule_matches(
                rule,
                envelope.__class__.from_dict(
                    {
                        **envelope.as_dict(),
                        "object_label": "car",
                        "genai": {"_labels": ["car"], "_zones": []},
                    }
                ),
            )
        )

    def test_finalized_smoking_event_keeps_safety_notification_copy(self):
        self.client.config = SimpleNamespace(
            cameras={
                "safety_camera": SimpleNamespace(
                    friendly_name="Safety Smoking Camera"
                )
            },
            notifications=SimpleNamespace(public_base_url=None),
        )
        self.client._aggregator = SimpleNamespace(
            media=SimpleNamespace(
                get=lambda _artifact_id: SimpleNamespace(id="media-1", revision=2)
            )
        )
        event = SimpleNamespace(
            id="event-smoking-1",
            camera="safety_camera",
            label="smoking",
            display_label=None,
            canonical_sub_label=None,
            sub_label="camera-safety",
            canonical_plate=None,
            canonical_plate_score=None,
            data={},
            zones=[],
            state="FINALIZED",
            end_time=125.0,
            revision=2,
            canonical_artifact_id="artifact-1",
            has_snapshot=True,
        )
        envelope = NotificationEnvelope(
            id="notification-1",
            source_type="review",
            source_id="review-smoking-1",
            camera="safety_camera",
            timestamp=123.0,
            title="Smoking detected",
            message="Detected",
            direct_url="/review?id=review-smoking-1",
            snapshot_ref="event-smoking-1",
            notification_type="alert",
            object_label="smoking",
        )

        with patch(
            "frigate.application.notifications.client.Event.get_or_none",
            return_value=event,
        ):
            canonical = self.client._canonical_envelope(envelope)

        self.assertIn("Smoking", canonical.title)
        self.assertIn("Phát hiện Smoking", canonical.message)
        self.assertNotIn("Xe đã kết thúc", canonical.message)

    def test_unfinalized_event_never_routes_raw_event_notification(self):
        event = SimpleNamespace(
            id="event-pending-1",
            camera="car_camera",
            label="car",
            state="END_SEEN",
            end_time=125.0,
        )
        self.client._route = MagicMock()
        with patch(
            "frigate.application.notifications.client.Event.get_or_none",
            return_value=event,
        ):
            self.client._route_finalized_event(event.id)
        self.client._route.assert_not_called()
