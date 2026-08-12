# Generated typing stub for protobuf runtime module.
from typing import Any
from google.protobuf.message import Message

class HealthCheckRequest(Message):
    def __init__(self, *, service: str = ...) -> None: ...
    service: str

class HealthCheckResponse(Message):
    UNKNOWN: int
    SERVING: int
    NOT_SERVING: int
    SERVICE_UNKNOWN: int
    def __init__(self, *, status: int = ...) -> None: ...
    status: int
