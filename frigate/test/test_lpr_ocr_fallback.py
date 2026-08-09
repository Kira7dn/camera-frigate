from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from frigate.data_processing.common.license_plate.mixin import (
    LicensePlateProcessingMixin,
)


def test_missing_text_box_does_not_try_ocr_crop_variants() -> None:
    mixin = object.__new__(LicensePlateProcessingMixin)
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
    assert not hasattr(mixin, "_direct_plate_fallback")
