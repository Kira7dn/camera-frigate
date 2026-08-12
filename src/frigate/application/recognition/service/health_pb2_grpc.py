# ruff: noqa
"""gRPC bindings for the standard grpc.health.v1 service."""

import grpc

from . import health_pb2


class HealthStub:
    def __init__(self, channel):
        self.Check = channel.unary_unary(
            "/grpc.health.v1.Health/Check",
            request_serializer=health_pb2.HealthCheckRequest.SerializeToString,
            response_deserializer=health_pb2.HealthCheckResponse.FromString,
        )
        self.Watch = channel.unary_stream(
            "/grpc.health.v1.Health/Watch",
            request_serializer=health_pb2.HealthCheckRequest.SerializeToString,
            response_deserializer=health_pb2.HealthCheckResponse.FromString,
        )


class HealthServicer:
    async def Check(self, request, context):
        raise NotImplementedError()

    async def Watch(self, request, context):
        raise NotImplementedError()


def add_HealthServicer_to_server(servicer, server):
    handlers = {
        "Check": grpc.unary_unary_rpc_method_handler(
            servicer.Check,
            request_deserializer=health_pb2.HealthCheckRequest.FromString,
            response_serializer=health_pb2.HealthCheckResponse.SerializeToString,
        ),
        "Watch": grpc.unary_stream_rpc_method_handler(
            servicer.Watch,
            request_deserializer=health_pb2.HealthCheckRequest.FromString,
            response_serializer=health_pb2.HealthCheckResponse.SerializeToString,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("grpc.health.v1.Health", handlers),)
    )
