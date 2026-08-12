import os
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from frigate.util import file as file_util


class TestFileUtils(TestCase):
    def _write_clean_snapshot(
        self, clips_dir: str, event_id: str, image: np.ndarray
    ) -> None:
        assert cv2.imwrite(
            os.path.join(clips_dir, f"front_door-{event_id}-clean.webp"),
            image,
        )

    def test_get_event_snapshot_bytes_reads_clean_webp(self):
        event_id = "clean-webp"
        image = np.zeros((100, 200, 3), np.uint8)
        event = SimpleNamespace(
            id=event_id,
            camera="front_door",
            label="Mock",
            top_score=100,
            score=0,
            start_time=0,
            data={
                "box": [0.25, 0.25, 0.25, 0.5],
                "score": 0.85,
                "attributes": [],
            },
        )

        with (
            tempfile.TemporaryDirectory() as clips_dir,
            patch.object(file_util, "CLIPS_DIR", clips_dir),
        ):
            self._write_clean_snapshot(clips_dir, event_id, image)

            snapshot_image, is_clean = file_util.load_event_snapshot_image(
                event, clean_only=True
            )

            assert is_clean
            assert snapshot_image is not None
            assert snapshot_image.shape[:2] == image.shape[:2]

            rendered_bytes, _ = file_util.get_event_snapshot_bytes(
                event,
                ext="jpg",
                timestamp=False,
                bounding_box=True,
                crop=False,
                height=40,
                quality=None,
                timestamp_style=None,
                colormap={},
            )
            assert rendered_bytes is not None

            rendered_image = cv2.imdecode(
                np.frombuffer(rendered_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            assert rendered_image is not None
            assert rendered_image.shape[0] == 40
            assert rendered_image.max() > 0

    def test_snapshot_bbox_uses_sub_label_with_person_fallback(self):
        image = np.zeros((20, 20, 3), np.uint8)
        for sub_label, expected in (("alice", "alice"), (None, "person")):
            event = SimpleNamespace(
                id="label-event",
                camera="front_door",
                label="person",
                sub_label=sub_label,
                top_score=0.9,
                score=0.9,
                start_time=1.0,
                data={"box": [0.1, 0.1, 0.5, 0.5], "attributes": []},
            )
            with (
                patch.object(
                    file_util,
                    "load_event_snapshot_image",
                    return_value=(image, True),
                ),
                patch.object(
                    file_util,
                    "get_snapshot_bytes",
                    return_value=(b"snapshot", 1.0),
                ) as renderer,
            ):
                file_util.get_event_snapshot_bytes(event, ext="jpg", bounding_box=True)
            self.assertEqual(renderer.call_args.kwargs["label"], expected)
