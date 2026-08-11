import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from frigate.util.image import intersection, transliterate_to_latin
from frigate.util.object import (
    get_clipped_object_recovery_region,
    get_cluster_boundary,
    get_cluster_candidates,
    get_cluster_region,
    get_region_from_grid,
    recovery_detection_supersedes,
    reduce_detections,
)
from frigate.video.ffmpeg import (
    CameraWatchdog,
    ordered_source_frame_time,
    put_latest_frame,
    put_ordered_frame,
)


class TestFiniteSourceWatchdog(unittest.TestCase):
    def test_eof_marker_disables_restart_only_for_matching_camera(self):
        watchdog = CameraWatchdog.__new__(CameraWatchdog)
        watchdog.config = SimpleNamespace(name="car_camera")

        with self.subTest("feature is opt in"):
            with patch.dict("os.environ", {}, clear=True):
                self.assertFalse(watchdog._finite_source_has_ended())

        with self.subTest("different camera marker is ignored"):
            with self._temporary_directory() as marker_dir:
                Path(marker_dir, "face_camera.end").write_text(
                    "1.0\n", encoding="utf-8"
                )
                with patch.dict(
                    "os.environ", {"PASSAGE_SOURCE_START_DIR": marker_dir}
                ):
                    self.assertFalse(watchdog._finite_source_has_ended())

        with self.subTest("matching camera marker disables restart"):
            with self._temporary_directory() as marker_dir:
                Path(marker_dir, "car_camera.end").write_text(
                    "1.0\n", encoding="utf-8"
                )
                with patch.dict(
                    "os.environ", {"PASSAGE_SOURCE_START_DIR": marker_dir}
                ):
                    self.assertTrue(watchdog._finite_source_has_ended())

    def test_live_camera_without_marker_keeps_normal_watchdog_behavior(self):
        watchdog = CameraWatchdog.__new__(CameraWatchdog)
        watchdog.config = SimpleNamespace(name="live_camera")

        with self._temporary_directory() as marker_dir:
            with patch.dict(
                "os.environ", {"PASSAGE_SOURCE_START_DIR": marker_dir}
            ):
                self.assertFalse(watchdog._finite_source_has_ended())

    @staticmethod
    def _temporary_directory():
        from tempfile import TemporaryDirectory

        return TemporaryDirectory()


class _Queue:
    def __init__(self, items=None, full=False):
        self.items = list(items or [])
        self.full = full

    def put(self, item, block, timeout=None):
        if self.full:
            self.full = False
            raise queue.Full
        self.items.append(item)

    def get(self, block):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


class _FrameManager:
    def __init__(self):
        self.closed = []

    def close(self, name):
        self.closed.append(name)


class _Skipped:
    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1


class _Stop:
    def is_set(self):
        return False


def test_capture_queue_replaces_stale_frame_with_latest() -> None:
    frame_queue = _Queue([("old", 1.0)], full=True)
    manager = _FrameManager()
    skipped = _Skipped()

    put_latest_frame(frame_queue, manager, "new", 2.0, skipped)

    assert frame_queue.items == [("new", 2.0)]
    assert manager.closed == ["old", "new"]
    assert skipped.count == 1


def test_file_capture_waits_without_removing_stale_frame() -> None:
    frame_queue = _Queue([("old", 1.0)], full=True)
    manager = _FrameManager()

    assert put_ordered_frame(frame_queue, manager, "new", 2.0, _Stop())

    assert frame_queue.items == [("old", 1.0), ("new", 2.0)]
    assert manager.closed == ["new"]


def test_file_capture_uses_source_timeline_under_backpressure() -> None:
    assert ordered_source_frame_time(1000.0, 0, 5) == 1000.0
    assert ordered_source_frame_time(1000.0, 74, 5) == 1014.8


def draw_box(frame, box, color=(255, 0, 0), thickness=2):
    cv2.rectangle(
        frame,
        (box[0], box[1]),
        (box[2], box[3]),
        color,
        thickness,
    )


def save_clusters_image(name, boxes, candidates, regions=[]):
    from norfair.drawing.color import Palette
    from norfair.drawing.drawer import Drawer

    canvas = np.zeros((1000, 2000, 3), np.uint8)
    for cluster in candidates:
        color = Palette.choose_color(np.random.rand())
        for b in cluster:
            box = boxes[b]
            draw_box(canvas, box, color, 2)
            # bottom right
            text_anchor = (
                box[2],
                box[3],
            )
            canvas = Drawer.text(
                canvas,
                str(b),
                position=text_anchor,
                size=None,
                color=(255, 255, 255),
                thickness=None,
            )
    for r in regions:
        draw_box(canvas, r, (0, 255, 0), 2)
    cv2.imwrite(
        f"debug/frames/{name}.jpg",
        canvas,
    )


def save_cluster_boundary_image(name, boxes, bounding_boxes):
    from norfair.drawing.color import Palette

    canvas = np.zeros((1000, 2000, 3), np.uint8)
    color = Palette.choose_color(np.random.rand())
    for box in boxes:
        draw_box(canvas, box, color, 2)
    for bound in bounding_boxes:
        draw_box(canvas, bound, (0, 255, 0), 2)
    cv2.imwrite(
        f"debug/frames/{name}.jpg",
        canvas,
    )


class TestRegion(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (1000, 2000)
        self.min_region_size = 160

    def test_cluster_candidates(self):
        boxes = [(100, 100, 200, 200), (202, 150, 252, 200), (900, 900, 950, 950)]

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        # save_clusters_image("cluster_candidates", boxes, cluster_candidates)

        assert len(cluster_candidates) == 2

    def test_cluster_candidates_partition_boxes(self):
        # every box index must appear in exactly one cluster (no box used twice,
        # none dropped) - the invariant the used-box tracking enforces
        boxes = [
            (100, 100, 200, 200),
            (202, 150, 252, 200),
            (210, 160, 260, 210),
            (900, 900, 950, 950),
            (905, 905, 955, 955),
        ]

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        assigned = [idx for cluster in cluster_candidates for idx in cluster]
        self.assertEqual(sorted(assigned), list(range(len(boxes))))

    def test_transliterate_to_latin(self):
        self.assertEqual(transliterate_to_latin("frégate"), "fregate")
        self.assertEqual(transliterate_to_latin("utilité"), "utilite")
        self.assertEqual(transliterate_to_latin("imágé"), "image")

    def test_cluster_boundary(self):
        boxes = [(100, 100, 200, 200), (215, 215, 325, 325)]
        boundary_boxes = [
            get_cluster_boundary(box, self.min_region_size) for box in boxes
        ]

        # save_cluster_boundary_image("bound", boxes, boundary_boxes)
        assert len(boundary_boxes) == 2

    def test_cluster_regions(self):
        boxes = [(100, 100, 200, 200), (202, 150, 252, 200), (900, 900, 950, 950)]

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        # save_clusters_image("regions", boxes, cluster_candidates, regions)
        assert len(regions) == 2

    def test_clipped_fast_vehicle_gets_expanded_recovery_region(self):
        detection = (
            "car",
            0.7098,
            (442, 163, 659, 360),
            42749,
            1.10,
            (339, 40, 659, 360),
        )

        recovery_region = get_clipped_object_recovery_region(
            (720, 1280), 320, detection
        )

        assert recovery_region is not None
        assert recovery_region[0] <= 356
        assert recovery_region[1] <= 164
        assert recovery_region[2] >= 656
        assert recovery_region[3] >= 457

    def test_complete_recovery_supersedes_partial_vehicle_detection(self):
        partial = (
            "car",
            0.5079,
            (389, 77, 687, 376),
            89102,
            1.0,
            (339, 0, 799, 460),
        )
        complete = (
            "car",
            0.6699,
            (289, 207, 616, 523),
            103332,
            1.0,
            (201, 57, 753, 609),
        )

        assert recovery_detection_supersedes(partial, complete, (720, 1280))

    def test_box_too_small_for_cluster(self):
        boxes = [(100, 100, 600, 600), (655, 100, 700, 145)]

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        save_clusters_image("too_small", boxes, cluster_candidates, regions)

        assert len(cluster_candidates) == 2
        assert len(regions) == 2

    def test_redundant_clusters(self):
        boxes = [(100, 100, 200, 200), (305, 305, 415, 415)]

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        # save_clusters_image("redundant", boxes, cluster_candidates, regions)

        assert len(cluster_candidates) == 2
        assert all([len(c) == 1 for c in cluster_candidates])
        assert len(regions) == 2

    def test_combine_boxes(self):
        boxes = [
            (480, 0, 540, 128),
            (536, 0, 558, 99),
        ]

        # boundary_boxes = [get_cluster_boundary(box) for box in boxes]
        # save_cluster_boundary_image("combine_bound", boxes, boundary_boxes)

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        # save_clusters_image("combine", boxes, cluster_candidates, regions)
        assert len(regions) == 1

    def test_dont_combine_smaller_boxes(self):
        boxes = [
            (460, 0, 561, 144),
            (565, 0, 586, 71),
        ]

        # boundary_boxes = [get_cluster_boundary(box) for box in boxes]
        # save_cluster_boundary_image("combine_bound", boxes, boundary_boxes)

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        # save_clusters_image("combine", boxes, cluster_candidates, regions)
        assert len(regions) == 2

    def test_dont_combine_boxes(self):
        boxes = [
            (460, 0, 532, 129),
            (586, 0, 606, 46),
        ]

        # boundary_boxes = [get_cluster_boundary(box) for box in boxes]
        # save_cluster_boundary_image("dont_combine_bound", boxes, boundary_boxes)

        cluster_candidates = get_cluster_candidates(
            self.frame_shape, self.min_region_size, boxes
        )

        regions = [
            get_cluster_region(self.frame_shape, self.min_region_size, candidate, boxes)
            for candidate in cluster_candidates
        ]

        # save_clusters_image("dont_combine", boxes, cluster_candidates, regions)
        assert len(regions) == 2


class TestObjectBoundingBoxes(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_box_intersection(self):
        box_a = [2012, 191, 2031, 205]
        box_b = [887, 92, 985, 151]
        box_c = [899, 128, 1080, 175]

        assert intersection(box_a, box_b) == None
        assert intersection(box_b, box_c) == (899, 128, 985, 151)

    def test_overlapping_objects_reduced(self):
        """Test that object not on edge of region is used when a higher scoring object at the edge of region is provided."""
        detections = [
            (
                "car",
                0.81,
                (1209, 73, 1437, 163),
                20520,
                2.53333333,
                (1150, 0, 1500, 200),
            ),
            (
                "car",
                0.88,
                (1238, 73, 1401, 171),
                15974,
                1.663265306122449,
                (1242, 0, 1602, 360),
            ),
        ]
        frame_shape = (720, 2560)
        consolidated_detections = reduce_detections(frame_shape, detections)
        assert consolidated_detections == [
            (
                "car",
                0.81,
                (1209, 73, 1437, 163),
                20520,
                2.53333333,
                (1150, 0, 1500, 200),
            )
        ]

    def test_non_overlapping_objects_not_reduced(self):
        """Test that non overlapping objects are not reduced."""
        detections = [
            (
                "car",
                0.81,
                (1209, 73, 1437, 163),
                20520,
                2.53333333,
                (1150, 0, 1500, 200),
            ),
            (
                "car",
                0.83203125,
                (1121, 55, 1214, 100),
                4185,
                2.066666666666667,
                (922, 0, 1242, 320),
            ),
            (
                "car",
                0.85546875,
                (1414, 97, 1571, 186),
                13973,
                1.7640449438202248,
                (1248, 0, 1568, 320),
            ),
        ]
        frame_shape = (720, 2560)
        consolidated_detections = reduce_detections(frame_shape, detections)
        assert len(consolidated_detections) == len(detections)

    def test_overlapping_different_size_objects_not_reduced(self):
        """Test that overlapping objects that are significantly different in size are not reduced."""
        detections = [
            (
                "car",
                0.81,
                (164, 279, 816, 719),
                286880,
                1.48,
                (90, 0, 910, 820),
            ),
            (
                "car",
                0.83203125,
                (248, 340, 328, 385),
                3600,
                1.777,
                (0, 0, 460, 460),
            ),
        ]
        frame_shape = (720, 2560)
        consolidated_detections = reduce_detections(frame_shape, detections)
        assert len(consolidated_detections) == len(detections)

    def test_vert_stacked_cars_not_reduced(self):
        detections = [
            ("car", 0.8, (954, 312, 1247, 475), 498512, 1.48, (800, 200, 1400, 600)),
            ("car", 0.85, (970, 380, 1273, 610), 698752, 1.56, (800, 200, 1400, 700)),
        ]
        frame_shape = (720, 1280)
        consolidated_detections = reduce_detections(frame_shape, detections)
        assert len(consolidated_detections) == len(detections)


class TestRegionGrid(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_region_in_range(self):
        """Test that region is kept at minimal size when within std dev."""
        frame_shape = (720, 1280)
        box = [450, 450, 550, 550]
        region_grid = [
            [],
            [],
            [],
            [{}, {}, {}, {}, {}, {"sizes": [0.25], "mean": 0.26, "std_dev": 0.01}],
        ]

        region = get_region_from_grid(frame_shape, box, 320, region_grid)
        assert region[2] - region[0] == 320

    def test_region_out_of_range(self):
        """Test that region is upsized when outside of std dev."""
        frame_shape = (720, 1280)
        box = [450, 450, 550, 550]
        region_grid = [
            [],
            [],
            [],
            [{}, {}, {}, {}, {}, {"sizes": [0.5], "mean": 0.5, "std_dev": 0.1}],
        ]

        region = get_region_from_grid(frame_shape, box, 320, region_grid)
        assert region[2] - region[0] > 320
