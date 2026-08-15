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
        self.client._lpr_updates = {}
        self.client._review_ids = {}
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
        self.assertEqual(
            self.client._lpr_updates["event-2"],
            {
                "plate": "51A12345",
                "score": 0.95,
                "plate_box": [10, 20, 100, 60],
                "sub_label": "company_car",
            },
        )

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

    def test_review_only_remembers_canonical_event_url(self):
        self.client._remember_review_id(
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
        self.assertEqual(self.client._review_ids, {"event-smoking-1": "review-smoking-1"})

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

    def test_active_smoking_event_routes_from_the_same_event_id(self):
        event = SimpleNamespace(
            id="event-smoking-active-1",
            camera="safety_camera",
            label="smoking",
            display_label=None,
            canonical_sub_label=None,
            sub_label="camera-safety",
            canonical_plate=None,
            canonical_plate_score=None,
            data={"score": 0.91},
            zones=[],
            state="ACTIVE",
            end_time=None,
        )
        self.client._route = MagicMock()
        with patch(
            "frigate.application.notifications.client.Event.get_or_none",
            return_value=event,
        ):
            self.client._route_finalized_event(event.id)

        self.client._route.assert_called_once()
        event_name, envelope = self.client._route.call_args.args
        self.assertEqual(event_name, "alert")
        self.assertEqual(envelope.source_type, "event")
        self.assertEqual(envelope.source_id, event.id)
