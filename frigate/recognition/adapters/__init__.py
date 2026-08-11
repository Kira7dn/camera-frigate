"""Host integration adapters for :mod:`frigate.recognition`."""

from .frigate import (
    BorrowedEvidenceResolver,
    FrigateEventAdapter,
    FrigateRecognitionAdapter,
)

__all__ = [
    "BorrowedEvidenceResolver",
    "FrigateEventAdapter",
    "FrigateRecognitionAdapter",
]
