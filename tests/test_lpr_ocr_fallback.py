from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from frigate.infrastructure.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)


def test_missing_text_box_does_not_try_ocr_crop_variants() -> None:
    mixin = object.__new__(LicensePlateProcessingMixin)
    mixin.config = SimpleNamespace(
        cameras={
            "cam": SimpleNamespace(lpr=SimpleNamespace(enhancement=0))
        }
    )
    mixin.model_runner = SimpleNamespace(
        detection_model=SimpleNamespace(runner=object()),
        classification_model=SimpleNamespace(runner=object()),
        recognition_model=SimpleNamespace(runner=object()),
    )
    recognize_calls = []
    mixin._detect = MethodType(lambda _self, _image, _debug: [], mixin)
    mixin._recognize = MethodType(
        lambda _self, *_args, **_kwargs: recognize_calls.append(True), mixin
    )

    assert mixin._process_license_plate(
        "cam", "passage", np.zeros((40, 80, 3), dtype=np.uint8), 1
    ) == ([], [], [])
    assert recognize_calls == []
    assert mixin._last_ocr_text_box_count == 0
    assert mixin._last_ocr_failure_stage == "text_detector_empty"
    assert not hasattr(mixin, "_direct_plate_fallback")


def test_text_detector_resize_does_not_downscale_small_plate_crop() -> None:
    mixin = object.__new__(LicensePlateProcessingMixin)
    mixin.max_size = 960
    image = np.zeros((110, 198, 3), dtype=np.uint8)

    resized = mixin._resize_image(image)

    assert resized.shape == (128, 224, 3)


def test_text_detector_resize_stays_within_max_size() -> None:
    mixin = object.__new__(LicensePlateProcessingMixin)
    mixin.max_size = 960
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)

    resized = mixin._resize_image(image)

    assert resized.shape == (480, 960, 3)
