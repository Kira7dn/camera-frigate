"""Standalone synchronous recognition core.

This package deliberately has no imports from Frigate detection, events, comms,
database, recording, or notification modules.
"""

from .contracts import (
    EvidenceCaptureRequest,
    JobReceipt,
    RecognitionArtifact,
    RecognitionJob,
    RecognitionOperation,
    RecognitionOutcome,
    RecognitionOutcomeStatus,
    RecognitionTask,
    RecognitionUpdate,
    TrackedObservation,
    TrackKey,
)
from .core import RecognitionCore
from .face import FacePolicy
from .lpr import LprPolicy
from .ports import ModelRecognition, RawRecognition

__all__ = [
    "FacePolicy",
    "EvidenceCaptureRequest",
    "LprPolicy",
    "JobReceipt",
    "RawRecognition",
    "ModelRecognition",
    "RecognitionArtifact",
    "RecognitionCore",
    "RecognitionJob",
    "RecognitionOperation",
    "RecognitionOutcome",
    "RecognitionOutcomeStatus",
    "RecognitionTask",
    "RecognitionUpdate",
    "TrackKey",
    "TrackedObservation",
]
