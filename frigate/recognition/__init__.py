"""Standalone synchronous recognition core.

This package deliberately has no imports from Frigate detection, events, comms,
database, recording, or notification modules.
"""

from .contracts import (
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
)
from .core import RecognitionCore
from .face import FacePolicy
from .lpr import LprPolicy
from .ports import RawRecognition

__all__ = [
    "FacePolicy",
    "LprPolicy",
    "RawRecognition",
    "RecognitionCore",
    "RecognitionTask",
    "RecognitionUpdate",
    "TrackKey",
    "TrackedObservation",
]
