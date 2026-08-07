"""Notification configuration models."""

from typing import Literal

from pydantic import Field, HttpUrl, model_validator

from ..base import FrigateBaseModel
from ..env import EnvString

NotificationProviderName = Literal["webpush", "telegram", "zalo"]

__all__ = [
    "CameraNotificationConfig",
    "NotificationConfig",
    "NotificationDeliveryConfig",
    "NotificationProviderName",
    "NotificationProvidersConfig",
    "NotificationRecipientConfig",
]


class NotificationRecipientConfig(FrigateBaseModel):
    """A destination for a social notification provider."""

    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        title="Recipient ID",
    )
    name: str = Field(min_length=1, max_length=100, title="Recipient name")
    chat_id: EnvString = Field(min_length=1, max_length=128, title="Chat ID")
    enabled: bool = Field(default=True, title="Enabled")
    cameras: list[str] = Field(
        default_factory=list,
        title="Cameras",
        description="Cameras delivered to this recipient. An empty list allows all cameras.",
    )


class WebPushProviderConfig(FrigateBaseModel):
    enabled: bool = Field(default=True, title="Enabled")


class TelegramProviderConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enabled")
    recipients: list[NotificationRecipientConfig] = Field(
        default_factory=list, title="Recipients"
    )

    @model_validator(mode="after")
    def validate_recipient_ids(self):
        ids = [recipient.id for recipient in self.recipients]
        if len(ids) != len(set(ids)):
            raise ValueError("Telegram recipient IDs must be unique")
        return self


class ZaloProviderConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enabled")
    public_base_url: HttpUrl | None = Field(
        default=None,
        title="Public base URL",
        description="Public HTTPS base URL used for signed Zalo snapshot links.",
    )
    media_url_ttl: int = Field(default=300, ge=30, le=3600, title="Media URL lifetime")
    recipients: list[NotificationRecipientConfig] = Field(
        default_factory=list, title="Recipients"
    )

    @model_validator(mode="after")
    def validate_recipient_ids(self):
        ids = [recipient.id for recipient in self.recipients]
        if len(ids) != len(set(ids)):
            raise ValueError("Zalo recipient IDs must be unique")
        return self


class NotificationProvidersConfig(FrigateBaseModel):
    webpush: WebPushProviderConfig = Field(
        default_factory=WebPushProviderConfig, title="WebPush"
    )
    telegram: TelegramProviderConfig = Field(
        default_factory=TelegramProviderConfig, title="Telegram"
    )
    zalo: ZaloProviderConfig = Field(default_factory=ZaloProviderConfig, title="Zalo")


class NotificationDeliveryConfig(FrigateBaseModel):
    max_attempts: int = Field(default=5, ge=1, le=100, title="Maximum attempts")
    initial_backoff: int = Field(
        default=5, ge=1, le=3600, title="Initial retry backoff"
    )
    max_backoff: int = Field(default=300, ge=1, le=86400, title="Maximum retry backoff")
    retention_days: int = Field(
        default=7, ge=1, le=365, title="Delivery retention days"
    )
    max_pending: int = Field(
        default=5000, ge=1, le=1000000, title="Maximum pending deliveries"
    )

    @model_validator(mode="after")
    def validate_backoff(self):
        if self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff must be greater than initial_backoff")
        return self


class CameraNotificationConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enable notifications")
    cooldown: int = Field(default=0, ge=0, title="Cooldown period")
    providers: list[NotificationProviderName] = Field(
        default_factory=lambda: ["webpush"],
        title="Notification providers",
        description="Providers selected for this camera. Existing configurations default to webpush only.",
    )
    enabled_in_config: bool | None = Field(default=None, title="Original state")


class NotificationConfig(FrigateBaseModel):
    """Global notification settings."""

    enabled: bool = Field(default=False, title="Enable notifications")
    email: str | None = Field(default=None, title="Notification email")
    cooldown: int = Field(default=0, ge=0, title="Cooldown period")
    enabled_in_config: bool | None = Field(default=None, title="Original state")
    providers: NotificationProvidersConfig = Field(
        default_factory=NotificationProvidersConfig, title="Providers"
    )
    delivery: NotificationDeliveryConfig = Field(
        default_factory=NotificationDeliveryConfig, title="Delivery"
    )
