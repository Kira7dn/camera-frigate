import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from scipy import stats

from frigate.config import FrigateConfig
from frigate.const import FACE_DIR, MODEL_CACHE_DIR
from frigate.embeddings.onnx.face_embedding import ArcfaceEmbedding, FaceNetEmbedding
from frigate.log import redirect_output_to_logger
from frigate.util.face_snapshot import is_face_identity_directory

cv2_face = cast(Any, cv2).face
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceMatch:
    top1_label: str
    top1_score: float
    top2_label: str | None
    top2_score: float

    @property
    def margin(self) -> float:
        return self.top1_score - self.top2_score


class FaceRecognizer(ABC):
    """Face recognition runner."""

    def __init__(self, config: FrigateConfig) -> None:
        self.config = config
        self.landmark_detector: Any | None = None
        self.init_landmark_detector()

    @abstractmethod
    def build(self) -> None:
        """Build face recognition model."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear current built model."""
        pass

    @abstractmethod
    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        pass

    @redirect_output_to_logger(logger, logging.DEBUG)  # type: ignore[misc]
    def init_landmark_detector(self) -> None:
        self.landmark_detector = self.create_landmark_detector()

    @redirect_output_to_logger(logger, logging.DEBUG)  # type: ignore[misc]
    def create_landmark_detector(self) -> Any | None:
        """Create one landmark detector for one owning CPU worker."""
        landmark_model = os.path.join(MODEL_CACHE_DIR, "facedet/landmarkdet.yaml")
        if not os.path.exists(landmark_model):
            return None
        landmark_detector = cv2_face.createFacemarkLBF()
        landmark_detector.loadModel(landmark_model)
        return landmark_detector

    def align_face(
        self,
        image: np.ndarray,
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        return self.align_face_with(
            self.landmark_detector, image, output_width, output_height
        )

    @staticmethod
    def align_face_with(
        landmark_detector: Any | None,
        image: np.ndarray,
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        if not landmark_detector:
            raise ValueError("Landmark detector not initialized")

        # landmark is run on grayscale images
        if image.ndim == 3:
            land_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            land_image = image

        _, lands = landmark_detector.fit(
            land_image, np.array([(0, 0, land_image.shape[1], land_image.shape[0])])
        )
        landmarks: np.ndarray = lands[0][0]

        # get landmarks for eyes
        leftEyePts = landmarks[42:48]
        rightEyePts = landmarks[36:42]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - 0.35

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX**2) + (dY**2))
        desiredDist = desiredRightEyeX - 0.35
        desiredDist *= output_width
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        # grab the rotation matrix for rotating and scaling the face
        eyesCenter = (
            int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2),
        )
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = output_width * 0.5
        tY = output_height * 0.35
        M[0, 2] += tX - eyesCenter[0]
        M[1, 2] += tY - eyesCenter[1]

        # apply the affine transformation
        return cv2.warpAffine(
            image, M, (output_width, output_height), flags=cv2.INTER_CUBIC
        )

    def prepare_face(
        self,
        face_image: np.ndarray,
        landmark_detector: Any | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run CPU-only scoring and alignment with a worker-owned detector."""
        blur_reduction = self.get_blur_confidence_reduction(face_image)
        detector = landmark_detector or self.landmark_detector
        return (
            self.align_face_with(
                detector, face_image, face_image.shape[1], face_image.shape[0]
            ),
            blur_reduction,
        )

    def classify_prepared_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[tuple[str, float] | None]:
        """Default bounded sequential path used by the static-batch FaceNet model."""
        return [self.classify(image) for image, _ in prepared]

    def classify_prepared_top2_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[FaceMatch | None]:
        """Compatibility adapter for recognizers without a ranked matcher."""
        return [
            FaceMatch(result[0], float(result[1]), None, 0.0)
            if result is not None
            else None
            for result in self.classify_prepared_batch(prepared)
        ]

    def get_blur_confidence_reduction(self, input: np.ndarray) -> float:
        """Calculates the reduction in confidence based on the blur of the image."""
        if not self.config.face_recognition.blur_confidence_filter:
            return 0.0

        variance = cv2.Laplacian(input, cv2.CV_64F).var()
        logger.debug(f"face detected with blurriness {variance}")

        if variance < 120:  # image is very blurry
            return 0.06
        elif variance < 160:  # image moderately blurry
            return 0.04
        elif variance < 200:  # image is slightly blurry
            return 0.02
        elif variance < 250:  # image is mostly clear
            return 0.01
        else:
            return 0.0


def build_class_mean(
    embs: list[np.ndarray],
    trim: float = 0.15,
    outlier_threshold: float = 0.30,
    min_keep_frac: float = 0.7,
    max_iters: int = 3,
) -> np.ndarray:
    """Build a class-mean embedding with two-layer outlier protection.

    Layer 1 (iterative, vector-wise): drop whole embeddings whose cosine
    similarity to the current class mean is below ``outlier_threshold``.
    Catches mislabeled or corrupted training samples (wrong face in the
    folder, full-frame screenshots, extreme crops) that per-dimension
    trimming cannot detect.

    Layer 2 (per-dimension): ``scipy.stats.trim_mean`` on the retained set
    to smooth per-component noise (lighting, expression, alignment jitter).

    Collections with fewer than 5 images bypass outlier rejection — too few
    samples to establish a reliable class center.
    """
    arr = np.stack(embs, axis=0)

    if len(arr) < 5:
        return np.asarray(stats.trim_mean(arr, trim, axis=0))

    keep = np.ones(len(arr), dtype=bool)
    floor = max(5, int(np.ceil(min_keep_frac * len(arr))))

    for _ in range(max_iters):
        mean = stats.trim_mean(arr[keep], trim, axis=0)
        m_norm = mean / (np.linalg.norm(mean) + 1e-9)
        e_norms = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        cos = e_norms @ m_norm
        new_keep = cos >= outlier_threshold

        if new_keep.sum() < floor:
            top = np.argsort(-cos)[:floor]
            new_keep = np.zeros(len(arr), dtype=bool)
            new_keep[top] = True

        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    dropped = int((~keep).sum())

    if dropped:
        logger.debug(
            f"Vector-wise outlier filter dropped {dropped}/{len(arr)} embeddings"
        )

    return np.asarray(stats.trim_mean(arr[keep], trim, axis=0))


def similarity_to_confidence(
    cosine_similarity: Any,
    median: float = 0.3,
    range_width: float = 0.6,
    slope_factor: float = 12,
) -> Any:
    """
    Default sigmoid function to map cosine similarity to confidence.

    Args:
        cosine_similarity (float): The input cosine similarity.
        median (float): Assumed median of cosine similarity distribution.
        range_width (float): Assumed range of cosine similarity distribution (90th percentile - 10th percentile).
        slope_factor (float): Adjusts the steepness of the curve.

    Returns:
        float: The confidence score.
    """

    # Calculate slope and bias
    slope = slope_factor / range_width
    bias = median

    # Calculate confidence
    confidence: float = 1 / (1 + np.exp(-slope * (cosine_similarity - bias)))
    return confidence


class FaceNetRecognizer(FaceRecognizer):
    def __init__(self, config: FrigateConfig):
        super().__init__(config)
        self.mean_embs: dict[str, np.ndarray] = {}
        self.face_embedder: FaceNetEmbedding = FaceNetEmbedding()
        self.model_builder_queue: queue.Queue | None = None
        self.build_generation = 0
        self.build_lock = threading.Lock()
        self.embedding_lock = threading.Lock()

    def clear(self) -> None:
        with self.build_lock:
            self.build_generation += 1
            self.model_builder_queue = None
        self.run_build_task()

    def run_build_task(self) -> None:
        with self.build_lock:
            if self.model_builder_queue is not None:
                return
            generation = self.build_generation
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            self.model_builder_queue = result_queue

        def build_model() -> None:
            face_embeddings_map: dict[str, list[np.ndarray]] = {}
            idx = 0
            landmark_detector = self.create_landmark_detector()

            dir = FACE_DIR
            for name in os.listdir(dir):
                face_folder = os.path.join(dir, name)

                if not is_face_identity_directory(name, face_folder):
                    continue

                face_embeddings_map[name] = []
                for image in os.listdir(face_folder):
                    img = cv2.imread(os.path.join(face_folder, image))

                    if img is None:
                        continue  # type: ignore[unreachable]

                    img = self.align_face_with(
                        landmark_detector, img, img.shape[1], img.shape[0]
                    )
                    with self.embedding_lock:
                        emb = self.face_embedder([img])[0].squeeze()
                    face_embeddings_map[name].append(emb)

                idx += 1

            result_queue.put((generation, face_embeddings_map))

        thread = threading.Thread(target=build_model, daemon=True)
        thread.start()

    def build(self) -> None:
        if not self.landmark_detector:
            self.init_landmark_detector()
            return None

        if self.model_builder_queue is not None:
            try:
                generation, face_embeddings_map = self.model_builder_queue.get(
                    timeout=0.1
                )
                self.model_builder_queue = None
            except queue.Empty:
                return
        else:
            self.run_build_task()
            return

        if generation != self.build_generation:
            self.run_build_task()
            return
        mean_embs: dict[str, np.ndarray] = {}
        for name, embs in face_embeddings_map.items():
            if embs:
                mean_embs[name] = build_class_mean(embs)

        self.mean_embs = mean_embs

        logger.debug("Finished building ArcFace model")

    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        if not self.landmark_detector:
            return None

        if self.model_builder_queue is not None:
            self.build()
        if not self.mean_embs:
            self.build()

            if not self.mean_embs:
                return None

        # face recognition is best run on grayscale images

        # get blur factor before aligning face
        blur_reduction = self.get_blur_confidence_reduction(face_image)

        # align face and run recognition
        img = self.align_face(face_image, face_image.shape[1], face_image.shape[0])
        embedding = self.face_embedder([img])[0].squeeze()

        score: float = 0
        label = ""

        for name, mean_emb in self.mean_embs.items():
            dot_product = np.dot(embedding, mean_emb)
            magnitude_A = np.linalg.norm(embedding)
            magnitude_B = np.linalg.norm(mean_emb)

            cosine_similarity = dot_product / (magnitude_A * magnitude_B)
            confidence = similarity_to_confidence(
                cosine_similarity, median=0.5, range_width=0.6
            )

            if confidence > score:
                score = confidence
                label = name

        return label, max(0, round(score - blur_reduction, 2))

    def classify_prepared_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[tuple[str, float] | None]:
        return [
            (match.top1_label, match.top1_score) if match is not None else None
            for match in self.classify_prepared_top2_batch(prepared)
        ]

    def classify_prepared_top2_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[FaceMatch | None]:
        """Keep FaceNet sequential because its TFLite input is static batch one."""
        if self.model_builder_queue is not None:
            self.build()
        if not self.mean_embs:
            self.build()
            if not self.mean_embs:
                return [None] * len(prepared)
        results: list[FaceMatch | None] = []
        for image, blur_reduction in prepared:
            with self.embedding_lock:
                embedding = self.face_embedder([image])[0].squeeze()
            ranked: list[tuple[float, str]] = []
            for name, mean_emb in self.mean_embs.items():
                cosine_similarity = np.dot(embedding, mean_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(mean_emb)
                )
                confidence = similarity_to_confidence(
                    cosine_similarity, median=0.5, range_width=0.6
                )
                ranked.append((float(confidence), name))
            ranked.sort(reverse=True)
            if not ranked:
                results.append(None)
                continue
            top1_score, top1_label = ranked[0]
            top2_score, top2_label = ranked[1] if len(ranked) > 1 else (0.0, None)
            results.append(
                FaceMatch(
                    top1_label,
                    max(0, round(top1_score - blur_reduction, 2)),
                    top2_label,
                    max(0, round(top2_score - blur_reduction, 2)),
                )
            )
        return results


class ArcFaceRecognizer(FaceRecognizer):
    def __init__(self, config: FrigateConfig):
        super().__init__(config)
        self.mean_embs: dict[str, np.ndarray] = {}
        self.face_embedder: ArcfaceEmbedding = ArcfaceEmbedding(config.face_recognition)
        self.model_builder_queue: queue.Queue | None = None
        self.build_generation = 0
        self.build_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        self.library_lock = threading.Lock()
        self.library_snapshot: tuple[tuple[str, ...], np.ndarray, int] = (
            (),
            np.empty((0, 0), dtype=np.float32),
            0,
        )

    def clear(self) -> None:
        with self.build_lock:
            self.build_generation += 1
            self.model_builder_queue = None
        self.run_build_task()

    def run_build_task(self) -> None:
        with self.build_lock:
            if self.model_builder_queue is not None:
                return
            generation = self.build_generation
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            self.model_builder_queue = result_queue

        def build_model() -> None:
            face_embeddings_map: dict[str, list[np.ndarray]] = {}
            idx = 0
            landmark_detector = self.create_landmark_detector()

            dir = FACE_DIR
            for name in os.listdir(dir):
                face_folder = os.path.join(dir, name)

                if not is_face_identity_directory(name, face_folder):
                    continue

                face_embeddings_map[name] = []
                for image in os.listdir(face_folder):
                    img = cv2.imread(os.path.join(face_folder, image))

                    if img is None:
                        continue  # type: ignore[unreachable]

                    img = self.align_face_with(
                        landmark_detector, img, img.shape[1], img.shape[0]
                    )
                    with self.embedding_lock:
                        emb = cast(Any, self.face_embedder)([img])[0].squeeze()
                    face_embeddings_map[name].append(emb)

                idx += 1

            result_queue.put((generation, face_embeddings_map))

        thread = threading.Thread(target=build_model, daemon=True)
        thread.start()

    def build(self) -> None:
        if not self.landmark_detector:
            self.init_landmark_detector()
            return None

        if self.model_builder_queue is not None:
            try:
                generation, face_embeddings_map = self.model_builder_queue.get(
                    timeout=0.1
                )
                self.model_builder_queue = None
            except queue.Empty:
                return
        else:
            self.run_build_task()
            return

        if generation != self.build_generation:
            self.run_build_task()
            return
        mean_embs: dict[str, np.ndarray] = {}
        for name, embs in face_embeddings_map.items():
            if embs:
                mean_embs[name] = build_class_mean(embs)

        labels = tuple(sorted(mean_embs))
        if labels:
            matrix = np.stack([mean_embs[label] for label in labels]).astype(
                np.float32, copy=False
            )
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            normalized_means = {
                label: matrix[index] for index, label in enumerate(labels)
            }
        else:
            matrix = np.empty((0, 0), dtype=np.float32)
            normalized_means = {}
        with self.library_lock:
            self.mean_embs = normalized_means
            self.library_snapshot = (labels, matrix, generation)

        logger.debug("Finished building ArcFace model")

    def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
        if not self.landmark_detector:
            return None

        if self.model_builder_queue is not None:
            self.build()
        if not self.mean_embs:
            self.build()

            if not self.mean_embs:
                return None

        prepared = self.prepare_face(face_image)
        return self.classify_prepared_batch([prepared])[0]

    def prepare_face(
        self,
        face_image: np.ndarray,
        landmark_detector: Any | None = None,
    ) -> tuple[np.ndarray, float]:
        """Align and normalize on CPU before work reaches the GPU executor."""
        aligned, blur_reduction = super().prepare_face(
            face_image, landmark_detector
        )
        return self.face_embedder.preprocess_one(aligned), blur_reduction

    def classify_prepared_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[tuple[str, float] | None]:
        return [
            (match.top1_label, match.top1_score) if match is not None else None
            for match in self.classify_prepared_top2_batch(prepared)
        ]

    def classify_prepared_top2_batch(
        self, prepared: list[tuple[np.ndarray, float]]
    ) -> list[FaceMatch | None]:
        """Embed a dynamic batch and classify it with one matrix multiply."""
        if not prepared:
            return []
        if self.model_builder_queue is not None:
            self.build()
        with self.library_lock:
            labels, library, _ = self.library_snapshot
        if not labels or library.size == 0:
            self.build()
            return [None] * len(prepared)

        with self.embedding_lock:
            embeddings = np.stack(
                [
                    np.asarray(embedding).squeeze()
                    for embedding in self.face_embedder.embed_preprocessed(
                        [image for image, _ in prepared]
                    )
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        similarities = embeddings @ library.T
        confidences = np.asarray(similarity_to_confidence(similarities))
        results: list[FaceMatch | None] = []
        for row in range(confidences.shape[0]):
            order = np.argsort(-confidences[row])
            best_index = int(order[0])
            second_index = int(order[1]) if len(order) > 1 else None
            reduction = prepared[row][1]
            top1_score = max(
                0, round(float(confidences[row, best_index]) - reduction, 2)
            )
            top2_score = (
                max(0, round(float(confidences[row, second_index]) - reduction, 2))
                if second_index is not None
                else 0.0
            )
            results.append(
                FaceMatch(
                    labels[best_index],
                    top1_score,
                    labels[second_index] if second_index is not None else None,
                    top2_score,
                )
            )
        return results
