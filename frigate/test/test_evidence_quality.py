"""Tests for bounded evidence ownership and shared quality selection."""

from __future__ import annotations

import threading
import unittest
from multiprocessing import Manager

import numpy as np

from frigate.data_processing.common.evidence import (
    EvidenceBufferPolicy,
    EvidenceRingBuffer,
    EvidenceSourceRole,
)
from frigate.data_processing.common.quality import QualitySelector, QualityThresholds
from frigate.data_processing.common.recognition import RecognitionLifecycle
from frigate.data_processing.types import DataProcessorMetrics
from frigate.embeddings.maintainer import EmbeddingMaintainer


def i420(width: int = 8, height: int = 8, value: int = 96) -> np.ndarray:
    return np.full((height * 3 // 2, width), value, dtype=np.uint8)


class TestMetricSync(unittest.TestCase):
    def test_quality_metric_sync_publishes_atomic_proxy_snapshots(self) -> None:
        with Manager() as manager:
            maintainer = object.__new__(EmbeddingMaintainer)
            maintainer.metrics = DataProcessorMetrics(manager, [])
            maintainer.evidence_ring = EvidenceRingBuffer({})
            maintainer.quality_selector = QualitySelector(maintainer.evidence_ring)
            maintainer.recognition_lifecycle = RecognitionLifecycle()

            # Shared dict readers use copy(), so the writer must update in place
            # instead of exposing the transient empty state from clear/update.
            maintainer.metrics.recognition_lifecycle_stats["exhausted"] = 2
            maintainer._sync_quality_metrics()

            snapshot = maintainer.metrics.recognition_lifecycle_stats.copy()
            self.assertEqual(snapshot["exhausted"], 2)
            self.assertEqual(snapshot["active_lifecycles"], 0)
            self.assertEqual(snapshot["in_flight"], 0)


class TestEvidenceRingBuffer(unittest.TestCase):
    def test_dedupes_and_honors_byte_and_time_bounds(self) -> None:
        ring = EvidenceRingBuffer(
            {"cam": EvidenceBufferPolicy(1.0, i420().nbytes * 2, 10.0)}
        )
        first = ring.ingest("cam", "detect", "one", 1.0, i420(value=1))
        duplicate = ring.ingest("cam", "detect", "one", 1.0, i420(value=2))
        second = ring.ingest("cam", "detect", "two", 1.2, i420(value=2))
        third = ring.ingest("cam", "detect", "three", 1.4, i420(value=3))

        self.assertEqual(first, duplicate)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)
        stats = ring.stats()
        self.assertEqual(stats["cameras"]["cam"]["frames"], 2)
        self.assertEqual(stats["capacity_evictions"], 1)

        ring.expire({"cam": 3.0})
        self.assertEqual(ring.stats()["cameras"]["cam"]["frames"], 0)
        self.assertGreaterEqual(ring.stats()["time_evictions"], 2)

    def test_pinned_capacity_drops_new_frame_and_lease_survives_unindex(self) -> None:
        size = i420().nbytes
        ring = EvidenceRingBuffer({"cam": EvidenceBufferPolicy(1.0, size, 10.0)})
        ref = ring.ingest("cam", "detect", "one", 1.0, i420(value=77))
        self.assertIsNotNone(ref)
        lease = ring.acquire(ref)
        self.assertIsNotNone(lease)

        ring.expire({"cam": 3.0})
        self.assertEqual(int(lease.frame[0, 0]), 77)
        self.assertIsNone(ring.ingest("cam", "detect", "two", 3.0, i420()))
        self.assertEqual(ring.last_reject_reason("cam"), "buffer_capacity")
        self.assertEqual(ring.stats()["pinned_capacity_drops"], 1)

        fork = lease.fork()
        lease.release()
        self.assertEqual(int(fork.frame[0, 0]), 77)
        fork.release()
        self.assertEqual(ring.stats()["cameras"]["cam"]["bytes"], 0)

    def test_concurrent_face_and_lpr_leases_stay_within_budget(self) -> None:
        frame = i420(32, 24)
        ring = EvidenceRingBuffer(
            {"cam": EvidenceBufferPolicy(3.0, frame.nbytes * 2, 10.0)}
        )
        ref = ring.ingest("cam", EvidenceSourceRole.detect, "shared", 1.0, frame)
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    lease = ring.acquire(ref)
                    if lease is None:
                        raise AssertionError("shared evidence expired")
                    self.assertEqual(lease.frame.shape, frame.shape)
                    lease.release()
            except (AssertionError, RuntimeError) as error:
                errors.append(error)

        workers = [threading.Thread(target=reader) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertFalse(errors)
        self.assertLessEqual(ring.stats()["cameras"]["cam"]["bytes"], frame.nbytes * 2)


class TestQualitySelector(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = EvidenceRingBuffer(
            {"cam": EvidenceBufferPolicy(3.0, 8 * 1024 * 1024, 10.0)}
        )
        self.selector = QualitySelector(self.ring)
        self.thresholds = QualityThresholds(24, 14, 20.0, 0.75, 0.75)
        checker = np.indices((32, 48)).sum(axis=0) % 2
        self.sharp = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, 2)

    def _ref(self, index: int):
        return self.ring.ingest(
            "cam", "detect", str(index), float(index), i420(64, 48, 90)
        )

    def _select(self, index: int, crop: np.ndarray, **kwargs):
        return self.selector.select(
            task="lpr",
            camera="cam",
            track_id="track",
            generation=kwargs.pop("generation", 1),
            frame_ref=self._ref(index),
            object_bbox=(0, 0, 60, 40),
            detail_bbox=(2, 3, 50, 35),
            detail_frame=crop,
            thresholds=self.thresholds,
            **kwargs,
        )

    def test_hard_gates_report_stable_reasons(self) -> None:
        blurred = np.full((32, 48, 3), 128, dtype=np.uint8)
        dark = np.zeros((32, 48, 3), dtype=np.uint8)
        bright = np.full((32, 48, 3), 255, dtype=np.uint8)

        self.assertIsNone(self._select(1, blurred))
        self.assertIsNone(self._select(2, dark))
        self.assertIsNone(self._select(3, bright))
        reasons = self.selector.stats()["reject_reasons"]
        self.assertEqual(reasons["lpr:blur_below_minimum"], 3)
        self.assertEqual(reasons["lpr:underexposed"], 1)
        self.assertEqual(reasons["lpr:overexposed"], 1)

    def test_unavailable_metrics_do_not_reject_and_id_is_deterministic(self) -> None:
        candidate = self._select(1, self.sharp, top_k=3)
        self.assertIsNotNone(candidate)
        self.assertIn("detector_score", candidate.unavailable_metrics)
        expected = self.selector.candidate_id(
            "lpr", "cam", "track", 1, candidate.frame_ref, candidate.detail_bbox
        )
        self.assertEqual(candidate.candidate_id, expected)
        self.assertIsNone(
            self.selector.select(
                task="lpr",
                camera="cam",
                track_id="track",
                generation=1,
                frame_ref=candidate.frame_ref,
                object_bbox=candidate.object_bbox,
                detail_bbox=candidate.detail_bbox,
                detail_frame=self.sharp,
                thresholds=self.thresholds,
            )
        )
        self.assertEqual(self.selector.stats()["deduped"], 1)
        candidate.release()

    def test_lpr_aspect_and_edge_clipping_are_hard_quality_gates(self) -> None:
        thresholds = QualityThresholds(
            24, 14, 20.0, 0.75, 0.75, 1.2, 6.0, 1
        )
        clipped = self.selector.select(
            task="lpr",
            camera="cam",
            track_id="edge",
            generation=1,
            frame_ref=self._ref(4),
            object_bbox=(0, 0, 60, 40),
            detail_bbox=(0, 3, 48, 35),
            detail_frame=self.sharp,
            thresholds=thresholds,
        )
        square = self.selector.select(
            task="lpr",
            camera="cam",
            track_id="aspect",
            generation=1,
            frame_ref=self._ref(5),
            object_bbox=(0, 0, 60, 40),
            detail_bbox=(2, 3, 34, 35),
            detail_frame=self.sharp,
            thresholds=thresholds,
        )
        self.assertIsNone(clipped)
        self.assertIsNone(square)
        reasons = self.selector.stats()["reject_reasons"]
        self.assertEqual(reasons["lpr:detail_box_edge_clipped"], 1)
        self.assertEqual(reasons["lpr:aspect_ratio_out_of_range"], 1)

    def test_top_k_replacement_and_generation_reset_release_owners(self) -> None:
        first = self._select(1, self.sharp, top_k=1, detector_score=0.1)
        second = self._select(2, self.sharp, top_k=1, detector_score=0.9)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(self.selector.stats()["replaced"], 1)
        self.assertFalse(self.selector.is_selected(first))
        self.assertTrue(self.selector.is_selected(second))

        third = self._select(3, self.sharp, top_k=1, detector_score=1.0, generation=2)
        self.assertIsNotNone(third)
        self.assertFalse(self.selector.is_selected(second))
        self.assertTrue(self.selector.is_selected(third))
        first.release()
        second.release()
        third.release()
        self.selector.shutdown()
        self.ring.close()
        self.assertEqual(self.ring.stats()["cameras"]["cam"]["bytes"], 0)

    def test_temporal_stability_uses_geometry_relative_to_moving_object(self) -> None:
        first = self.selector.select(
            task="lpr",
            camera="cam",
            track_id="moving",
            generation=1,
            frame_ref=self._ref(6),
            object_bbox=(0, 0, 32, 24),
            detail_bbox=(4, 8, 28, 22),
            detail_frame=self.sharp,
            thresholds=self.thresholds,
        )
        second = self.selector.select(
            task="lpr",
            camera="cam",
            track_id="moving",
            generation=1,
            frame_ref=self._ref(7),
            object_bbox=(30, 20, 62, 44),
            detail_bbox=(34, 28, 58, 42),
            detail_frame=self.sharp,
            thresholds=self.thresholds,
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIn("temporal_stability", first.unavailable_metrics)
        self.assertAlmostEqual(
            second.quality_components["temporal_stability"], 1.0
        )
        first.release()
        second.release()


if __name__ == "__main__":
    unittest.main()
