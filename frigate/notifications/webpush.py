"""WebPush provider adapter for normalized notifications."""

from multiprocessing.synchronize import Event as MpEvent

from frigate.comms.webpush import WebPushClient
from frigate.config import FrigateConfig

from .envelope import NotificationEnvelope


class WebPushProvider:
    """Lazily host the existing encrypted WebPush delivery engine."""

    def __init__(self, config: FrigateConfig, stop_event: MpEvent) -> None:
        self.config = config
        self.stop_event = stop_event
        self.client: WebPushClient | None = None

    @property
    def configured(self) -> bool:
        return bool(self.config.notifications.email)

    @property
    def pending(self) -> int:
        return self.client.notification_queue.qsize() if self.client else 0

    def _get_client(self) -> WebPushClient:
        if self.client is None:
            self.client = WebPushClient(
                self.config, self.stop_event, manage_suspensions=False
            )
        return self.client

    def deliver(self, envelope: NotificationEnvelope) -> int:
        return self._get_client().send_envelope(envelope)

    def refresh_authorization(self) -> None:
        if self.client:
            self.client._refresh_user_cameras()

    def stop(self) -> None:
        if self.client:
            self.client.stop()
