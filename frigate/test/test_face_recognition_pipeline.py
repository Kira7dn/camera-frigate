"""Tests for bounded multi-camera face capture and ArcFace batching."""

import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from frigate.embeddings.maintainer import EmbeddingMaintainer  # noqa: F401
from frigate.data_processing.common.face.model import ArcFaceRecognizer
from frigate.data_processing.common.face.pipeline import (
    FaceCaptureRequest,
    FaceRecognitionPipeline,
    LatestFaceCandidateStore,
    crop_yuv_region_to_bgr,
)
from frigate.detectors.detection_runners import CudaGraphRunner
from frigate.embeddings.onnx.face_embedding import ArcfaceEmbedding
from frigate.embeddings.types import EnrichmentModelTypeEnum


def request(camera: str, event_id: str, frame_time: float = 1.0) -> FaceCaptureRequest:
    return FaceCaptureRequest(
        camera=camera,
        event_id=event_id,
        frame_time=frame_time,
        generation=0,
        person_box=(0, 0, 4, 4),
        yuv_frame=np.zeros((6, 4), dtype=np.uint8),
        detection_threshold=0.5,
        min_area=1,
        requires_face_detection=True,
        attribute_face_box=None,
        vote_count=0,
        created_monotonic=time.monotonic(),
        quality=frame_time,
    )


class FaceRecognitionPipelineTest(unittest.TestCase):
    def test_keyed_store_is_latest_only_bounded_and_releases_drops(self) -> None:
        dropped = []
        store = LatestFaceCandidateStore[FaceCaptureRequest](
            max_per_camera=4,
            on_drop=lambda item, reason: dropped.append((item.event_id, reason)),
        )
        store.submit(request("cam", "same", 1.0))
        store.submit(request("cam", "same", 2.0))
        for index in range(4):
            store.submit(request("cam", f"event-{index}", 3.0 + index))
        self.assertEqual(len(store), 4)
        self.assertIn(("same", "replaced"), dropped)
        self.assertTrue(any(reason == "camera_limit" for _, reason in dropped))
        selected = store.take_fair(4, 0)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item.key for item in selected}), 4)

    def test_store_cleans_ttl_end_track_and_discontinuity_replacement(self) -> None:
        dropped = []
        store = LatestFaceCandidateStore[FaceCaptureRequest](
            ttl_seconds=0.01,
            on_drop=lambda item, reason: dropped.append((item.key, reason)),
        )
        old = replace(request("cam", "old"), created_monotonic=time.monotonic() - 1)
        store.submit(old)
        self.assertEqual(store.take_fair(1, 0), [])
        self.assertIn((old.key, "ttl"), dropped)
        live = request("cam", "live")
        store.submit(live)
        self.assertTrue(store.remove(live.key, "discontinuity"))
        self.assertEqual(len(store), 0)
        store.submit(request("cam", "ended"))
        self.assertEqual(store.remove_camera_missing("cam", set()), 1)
        self.assertEqual(len(store), 0)

    def test_scheduler_is_fair_across_eight_cameras_and_caps_batch(self) -> None:
        store = LatestFaceCandidateStore[FaceCaptureRequest]()
        for camera in range(8):
            for track in range(2):
                item = request(f"cam-{camera}", f"track-{track}")
                store.submit(replace(item, vote_count=track))
        first = store.take_fair(4, 0)
        second = store.take_fair(4, 0)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertEqual(len({item.camera for item in first}), 4)
        self.assertEqual(len({item.camera for item in second}), 4)
        self.assertEqual({item.camera for item in first + second}, {f"cam-{i}" for i in range(8)})
        self.assertTrue(all(item.vote_count == 0 for item in first + second))

    def test_yuv_person_crop_maps_back_to_full_frame(self) -> None:
        bgr = np.zeros((16, 20, 3), dtype=np.uint8)
        bgr[4:12, 6:14] = (20, 100, 220)
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        cropped, region = crop_yuv_region_to_bgr(yuv, (6, 4, 14, 12))
        self.assertEqual(region, (6, 4, 14, 12))
        self.assertEqual(cropped.shape[:2], (8, 8))
        self.assertGreater(float(cropped[:, :, 2].mean()), 180)

    def test_arcface_preprocess_handles_all_batch_inputs(self) -> None:
        embedder = ArcfaceEmbedding.__new__(ArcfaceEmbedding)
        images = [np.full((112, 112, 3), value, np.uint8) for value in (0, 32, 96, 255)]
        processed = embedder._preprocess_inputs(images)
        self.assertEqual(len(processed), 4)
        means = [float(item["data"].mean()) for item in processed]
        self.assertEqual(means, sorted(means))
        self.assertEqual(processed[0]["data"].shape, (1, 3, 112, 112))

    def test_dynamic_arcface_batch_does_not_use_fixed_cuda_graph_buffers(self) -> None:
        self.assertFalse(
            CudaGraphRunner.is_model_supported(
                EnrichmentModelTypeEnum.arcface.value
            )
        )

    def test_arcface_batch_preserves_order_and_library_swap_is_atomic(self) -> None:
        recognizer = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
        recognizer.model_builder_queue = None
        recognizer.library_lock = threading.Lock()
        recognizer.embedding_lock = threading.Lock()
        recognizer.library_snapshot = (
            ("alice", "bob"),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            1,
        )
        class Embedder:
            def embed_preprocessed(self, images):
                return [
                    np.asarray(
                        [float(image[0, 0]), float(image[0, 1])],
                        dtype=np.float32,
                    )
                    for image in images
                ]

        recognizer.face_embedder = Embedder()
        prepared = [
            (np.asarray([[0.0, 1.0]], dtype=np.float32), 0.0),
            (np.asarray([[1.0, 0.0]], dtype=np.float32), 0.0),
        ]
        self.assertEqual(
            [result[0] for result in recognizer.classify_prepared_batch(prepared)],
            ["bob", "alice"],
        )
        with recognizer.library_lock:
            recognizer.library_snapshot = (
                ("carol",),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                2,
            )
        self.assertEqual(
            recognizer.classify_prepared_batch([prepared[1]])[0][0], "carol"
        )

    def test_executor_batches_without_accumulating_a_fifo_backlog(self) -> None:
        class Detector:
            def setInputSize(self, size) -> None:
                self.size = size

            def detect(self, image):
                height, width = image.shape[:2]
                return None, np.asarray(
                    [[0, 0, width, height, 0, 0, 0, 0, 0, 0, 0, 0, 0.99]],
                    dtype=np.float32,
                )

        class Recognizer:
            def __init__(self) -> None:
                self.batch_sizes = []

            def create_landmark_detector(self):
                return object()

            def prepare_face(self, face, detector):
                return face, 0.0

            def classify_prepared_batch(self, prepared):
                self.batch_sizes.append(len(prepared))
                time.sleep(0.01)
                return [("alice", 0.9) for _ in prepared]

        recognizer = Recognizer()
        pipeline = FaceRecognitionPipeline(recognizer, detector_factory=Detector)
        try:
            bgr = np.full((16, 16, 3), 127, dtype=np.uint8)
            yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
            for camera in range(8):
                item = request(f"cam-{camera}", "track")
                pipeline.submit(
                    replace(
                        item,
                        yuv_frame=yuv,
                        person_box=(0, 0, 16, 16),
                        min_area=1,
                    )
                )
            deadline = time.monotonic() + 2
            outcomes = []
            while len(outcomes) < 8 and time.monotonic() < deadline:
                outcomes.extend(pipeline.drain_results())
                time.sleep(0.01)
            self.assertEqual(len(outcomes), 8)
            self.assertTrue(all(size <= 4 for size in recognizer.batch_sizes))
            self.assertEqual(pipeline.pending_count(), 0)
        finally:
            pipeline.stop()

    def test_hot_path_has_no_full_frame_yuv_to_bgr_conversion(self) -> None:
        source = (
            Path(__file__).parents[1] / "embeddings" / "maintainer.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("COLOR_YUV2BGR_I420", source)


if __name__ == "__main__":
    unittest.main()
