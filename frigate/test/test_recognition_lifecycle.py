from __future__ import annotations

import unittest

from pydantic import ValidationError

from frigate.config.camera.quality import RecognitionLifecycleConfig
from frigate.data_processing.common.recognition import (
    RecognitionKey,
    RecognitionLifecycle,
    RecognitionPolicy,
    RecognitionStatus,
)


class RecognitionLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lifecycle = RecognitionLifecycle()
        self.key = RecognitionKey("lpr", "cam", "track", 1)
        self.policy = RecognitionPolicy(3, 0.4, 0.90)

    def begin(
        self,
        candidate_id: str,
        frame_time: float,
        bbox: tuple[int, int, int, int] = (0, 0, 100, 100),
    ):
        return self.lifecycle.begin_attempt(
            self.key,
            candidate_id=candidate_id,
            frame_time=frame_time,
            detail_bbox=bbox,
            quality_score=1.0,
            policy=self.policy,
        )

    def test_exact_diversity_boundaries_are_independent(self) -> None:
        first, reason = self.begin("one", 1.0)
        self.assertIsNone(reason)
        self.assertTrue(self.lifecycle.complete_attempt(first))

        interval_boundary, reason = self.begin("two", 1.4)
        self.assertIsNone(reason)
        self.assertTrue(self.lifecycle.complete_attempt(interval_boundary))

        iou_boundary, reason = self.begin("three", 1.5, (0, 0, 90, 100))
        self.assertIsNone(reason)
        self.assertTrue(self.lifecycle.complete_attempt(iou_boundary))

    def test_candidate_id_and_strict_time_iou_duplicate_detection(self) -> None:
        first, _ = self.begin("same", 1.0)
        self.lifecycle.complete_attempt(first)
        duplicate, reason = self.begin("same", 2.0, (500, 500, 600, 600))
        self.assertIsNone(duplicate)
        self.assertEqual(reason, "duplicate_candidate")

        close, reason = self.begin("different", 1.39)
        self.assertIsNone(close)
        self.assertEqual(reason, "insufficient_diversity")

    def test_budget_inflight_completion_and_terminal_are_idempotent(self) -> None:
        leases = []
        for index in range(3):
            lease, reason = self.begin(f"candidate-{index}", 1.0 + index)
            self.assertIsNone(reason)
            leases.append(lease)
        denied, reason = self.begin("four", 10.0)
        self.assertIsNone(denied)
        self.assertEqual(reason, "attempt_budget_exhausted")
        self.assertEqual(self.lifecycle.stats()["in_flight"], 3)

        self.assertTrue(self.lifecycle.complete_attempt(leases[0], result="ok"))
        self.assertFalse(self.lifecycle.complete_attempt(leases[0], result="stale"))
        self.assertTrue(
            self.lifecycle.terminal(
                self.key, RecognitionStatus.ACCEPTED, "consensus_accepted"
            )
        )
        self.assertFalse(
            self.lifecycle.terminal(
                self.key, RecognitionStatus.EXHAUSTED, "attempt_budget_exhausted"
            )
        )
        self.assertFalse(self.lifecycle.complete_attempt(leases[1]))

    def test_generation_reset_and_expire_return_to_baseline(self) -> None:
        lease, _ = self.begin("one", 1.0)
        next_key = RecognitionKey("lpr", "cam", "track", 2)
        next_lease, reason = self.lifecycle.begin_attempt(
            next_key,
            candidate_id="one",
            frame_time=1.0,
            detail_bbox=(0, 0, 100, 100),
            quality_score=1.0,
            policy=self.policy,
        )
        self.assertIsNone(reason)
        self.assertNotEqual(lease.key, next_lease.key)
        self.lifecycle.expire(self.key)
        self.lifecycle.expire(next_key, "shutdown")
        stats = self.lifecycle.stats()
        self.assertEqual(stats["active_tracks"], 0)
        self.assertEqual(stats["in_flight"], 0)
        self.assertEqual(stats["pending_cancellations"], 2)

    def test_config_defaults_and_three_attempt_bound(self) -> None:
        config = RecognitionLifecycleConfig()
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.min_candidate_interval_seconds, 0.4)
        self.assertEqual(config.max_candidate_bbox_iou, 0.90)
        self.assertEqual(config.lpr_min_consensus_votes, 2)
        with self.assertRaises(ValidationError):
            RecognitionLifecycleConfig(max_attempts=4)
        self.assertEqual(
            RecognitionLifecycleConfig(
                max_attempts=1, lpr_min_consensus_votes=3
            ).max_attempts,
            1,
        )


if __name__ == "__main__":
    unittest.main()
