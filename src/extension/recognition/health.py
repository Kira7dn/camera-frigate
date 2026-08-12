"""Async implementation of the standard gRPC health service."""

from __future__ import annotations

import asyncio

import grpc

from . import health_pb2, health_pb2_grpc


class HealthServicer(health_pb2_grpc.HealthServicer):
    def __init__(self) -> None:
        self._statuses = {"": health_pb2.HealthCheckResponse.SERVING}
        self._watchers: dict[str, set[asyncio.Queue[int]]] = {}

    async def Check(self, request, context):
        status = self._statuses.get(request.service)
        if status is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown service")
            raise RuntimeError("unknown health service")
        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(self, request, context):
        queue: asyncio.Queue[int] = asyncio.Queue()
        service = request.service
        self._watchers.setdefault(service, set()).add(queue)
        await queue.put(
            self._statuses.get(
                service, health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
            )
        )
        try:
            while True:
                yield health_pb2.HealthCheckResponse(status=await queue.get())
        finally:
            self._watchers.get(service, set()).discard(queue)

    def set(self, service: str, status: int) -> None:
        self._statuses[service] = status
        for queue in tuple(self._watchers.get(service, ())):
            queue.put_nowait(status)
