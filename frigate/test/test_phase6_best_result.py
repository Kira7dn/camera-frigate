from __future__ import annotations

import numpy as np

from frigate.data_processing.common.evidence import (
    EvidenceBufferPolicy,
    EvidenceRingBuffer,
)
from frigate.data_processing.common.quality import QualitySelector, QualityThresholds
from frigate.data_processing.common.recognition import (
    BestResultReducer,
    face_result_outcome,
    lpr_result_outcome,
)


def image_rank(candidate_id: str, weakest: float = 0.8):
    return (1.0, weakest, 0.9, 0.9, 1000, candidate_id)


def i420(width: int = 64, height: int = 48) -> np.ndarray:
    return np.full((height * 3 // 2, width), 90, dtype=np.uint8)


def sharp_crop() -> np.ndarray:
    checker = np.indices((32, 48)).sum(axis=0) % 2
    return np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, 2)


def test_rolling_pool_uses_coverage_and_freezes_admission() -> None:
    ring = EvidenceRingBuffer(
        {"cam": EvidenceBufferPolicy(3.0, 8 * 1024 * 1024, 10.0)}
    )
    selector = QualitySelector(ring)
    thresholds = QualityThresholds(24, 14, 20.0, 0.75, 0.75)

    def select(index: int, detector_score: float | None):
        ref = ring.ingest("cam", "detect", str(index), float(index), i420())
        assert ref is not None
        return selector.select(
            task="lpr",
            camera="cam",
            track_id="passage",
            generation=1,
            frame_ref=ref,
            object_bbox=(0, 0, 60, 40),
            detail_bbox=(2, 3, 50, 35),
            detail_frame=sharp_crop(),
            thresholds=thresholds,
            top_k=3,
            detector_score=detector_score,
        )

    missing = select(1, None)
    complete = select(2, 0.9)
    assert missing is not None and complete is not None
    assert missing.quality_components["coverage"] < 1.0
    assert complete.image_rank > missing.image_rank
    frozen = selector.freeze("lpr", "cam", "passage", 1)
    assert [candidate.candidate_id for candidate in frozen] == [
        complete.candidate_id,
        missing.candidate_id,
    ]
    for candidate in frozen:
        candidate.release()
    assert select(3, 1.0) is None
    missing.release()
    complete.release()
    selector.shutdown()
    ring.close()


def test_lpr_three_different_texts_choose_best_valid_rank() -> None:
    reducer = BestResultReducer("lpr")
    reducer.add(
        lpr_result_outcome(
            candidate_id="a",
            image_rank=image_rank("a"),
            payload="111111",
            character_scores=(0.95, 0.95, 0.60),
            recognition_threshold=0.8,
            length_valid=True,
            format_valid=True,
            recognized_text_area=900,
        )
    )
    reducer.add(
        lpr_result_outcome(
            candidate_id="b",
            image_rank=image_rank("b"),
            payload="657648",
            character_scores=(0.91, 0.92, 0.93),
            recognition_threshold=0.8,
            length_valid=True,
            format_valid=True,
            recognized_text_area=800,
        )
    )
    reducer.add(
        lpr_result_outcome(
            candidate_id="c",
            image_rank=image_rank("c"),
            payload="INVALID",
            character_scores=(0.99, 0.99, 0.99),
            recognition_threshold=0.8,
            length_valid=True,
            format_valid=False,
            recognized_text_area=2000,
        )
    )
    assert reducer.winner().payload == "657648"


def test_face_margin_outranks_higher_ambiguous_top1() -> None:
    reducer = BestResultReducer("face")
    reducer.add(
        face_result_outcome(
            candidate_id="ambiguous",
            image_rank=image_rank("ambiguous"),
            payload="Jane",
            top1_score=0.99,
            top2_score=0.95,
            recognition_threshold=0.8,
            min_identity_margin=0.1,
            image_quality_valid=True,
        )
    )
    reducer.add(
        face_result_outcome(
            candidate_id="clear",
            image_rank=image_rank("clear"),
            payload="Jack",
            top1_score=0.90,
            top2_score=0.40,
            recognition_threshold=0.8,
            min_identity_margin=0.1,
            image_quality_valid=True,
        )
    )
    assert reducer.winner().payload == "Jack"


def test_face_terminal_reason_is_deterministic() -> None:
    reducer = BestResultReducer("face")
    reducer.add(
        face_result_outcome(
            candidate_id="unknown",
            image_rank=image_rank("unknown"),
            payload="unknown",
            top1_score=0.7,
            top2_score=0.2,
            recognition_threshold=0.8,
            min_identity_margin=0.1,
            image_quality_valid=True,
        )
    )
    assert reducer.exhausted_reason() == "unknown"
