"""Notification configuration models.

The global ``notifications`` document is the only persistent notification
configuration.  Per-camera notification fields remain readable for legacy
configuration migration, but the v2 runtime never consults them.
"""

import os
from typing import Any, Literal, cast

from pydantic import Field, HttpUrl, model_validator

from ..base import FrigateBaseModel
from ..env import EnvString

NotificationProviderName = Literal["webpush", "telegram", "zalo"]
NotificationEventName = Literal[
    "alert",
    "object_detected",
    "license_plate",
    "face_recognized",
    "camera_offline",
    "camera_online",
    "semantic_trigger",
    "camera_monitoring",
]

__all__ = [
    "CameraNotificationConfig",
    "NotificationChannelsConfig",
    "NotificationConfig",
    "NotificationDeliveryConfig",
    "NotificationDestinationsConfig",
    "NotificationEventName",
    "NotificationProviderName",
    "NotificationRecipientConfig",
    "NotificationRuleConfig",
    "NotificationRuleFiltersConfig",
]


class NotificationRecipientConfig(FrigateBaseModel):
    """A named destination within a social notification channel."""

    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        title="Recipient ID",
    )
    name: str = Field(min_length=1, max_length=100, title="Recipient name")
    chat_id: EnvString = Field(min_length=1, max_length=128, title="Chat ID")
    enabled: bool = Field(default=True, title="Enabled")


class WebPushChannelConfig(FrigateBaseModel):
    enabled: bool = Field(default=True, title="Enabled")


class TelegramChannelConfig(FrigateBaseModel):
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


class ZaloChannelConfig(FrigateBaseModel):
    enabled: bool = Field(default=False, title="Enabled")
    # Deprecated read-only compatibility. The v2 runtime uses the global URL.
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


class NotificationChannelsConfig(FrigateBaseModel):
    webpush: WebPushChannelConfig = Field(
        default_factory=WebPushChannelConfig, title="WebPush"
    )
    telegram: TelegramChannelConfig = Field(
        default_factory=TelegramChannelConfig, title="Telegram"
    )
    zalo: ZaloChannelConfig = Field(default_factory=ZaloChannelConfig, title="Zalo")


class NotificationRuleFiltersConfig(FrigateBaseModel):
    cameras: list[str] = Field(default_factory=list, title="Cameras")
    labels: list[str] = Field(default_factory=list, title="Object labels")
    zones: list[str] = Field(default_factory=list, title="Zones")
    identities: list[str] = Field(default_factory=list, title="Face identities")
    trigger_names: list[str] = Field(default_factory=list, title="Semantic triggers")
    conditions: list[str] = Field(default_factory=list, title="Monitoring conditions")


class NotificationDestinationsConfig(FrigateBaseModel):
    webpush: bool = Field(default=False, title="WebPush")
    telegram: list[str] = Field(default_factory=list, title="Telegram recipients")
    zalo: list[str] = Field(default_factory=list, title="Zalo recipients")


class NotificationRuleConfig(FrigateBaseModel):
    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        title="Rule ID",
    )
    name: str = Field(min_length=1, max_length=100, title="Name")
    enabled: bool = Field(default=True, title="Enabled")
    event: NotificationEventName = Field(title="Event")
    filters: NotificationRuleFiltersConfig = Field(
        default_factory=NotificationRuleFiltersConfig, title="Filters"
    )
    destinations: NotificationDestinationsConfig = Field(
        default_factory=NotificationDestinationsConfig, title="Destinations"
    )
    cooldown: int = Field(default=0, ge=0, le=86400, title="Cooldown seconds")


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


class NotificationPipelineConfig(FrigateBaseModel):
    finalization_timeout: float = Field(default=5, ge=0, le=60)
    shadow_mode: bool = Field(
        default=True,
        description="Build projections and artifacts without enqueueing providers.",
    )


class NotificationPresentationConfig(FrigateBaseModel):
    locale: Literal["vi"] = "vi"


class NotificationMediaConfig(FrigateBaseModel):
    max_storage_mb: int = Field(default=2048, ge=128, le=1048576)


class CameraNotificationConfig(FrigateBaseModel):
    """Legacy camera notification settings, read only for v1 migration."""

    enabled: bool = Field(default=False, title="Enable notifications")
    cooldown: int = Field(default=0, ge=0, title="Cooldown period")
    providers: list[NotificationProviderName] = Field(
        default_factory=lambda: [cast(NotificationProviderName, "webpush")],
        title="Notification providers",
    )
    enabled_in_config: bool | None = Field(default=None, title="Original state")


class NotificationConfig(FrigateBaseModel):
    """Authoritative global notification document."""

    schema_version: Literal[2] = Field(default=2, title="Schema version")
    enabled: bool = Field(default=False, title="Enable notifications")
    email: str | None = Field(
        default=None,
        title="WebPush contact email",
        description="Contact used for the VAPID WebPush subscription.",
    )
    enabled_in_config: bool | None = Field(default=None, title="Original state")
    public_base_url: HttpUrl | None = Field(
        default=None,
        description="Public HTTPS base URL for immutable signed media and actions.",
    )
    channels: NotificationChannelsConfig = Field(
        default_factory=NotificationChannelsConfig, title="Channels"
    )
    rules: list[NotificationRuleConfig] = Field(default_factory=list, title="Rules")
    delivery: NotificationDeliveryConfig = Field(
        default_factory=NotificationDeliveryConfig, title="Delivery"
    )
    pipeline: NotificationPipelineConfig = Field(default_factory=NotificationPipelineConfig)
    presentation: NotificationPresentationConfig = Field(
        default_factory=NotificationPresentationConfig
    )
    media: NotificationMediaConfig = Field(default_factory=NotificationMediaConfig)

    @model_validator(mode="before")
    @classmethod
    def read_legacy_document(cls, value: Any):
        """Allow a v1 document to start so it can be migrated safely."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if data.get("public_base_url") == "{NGROK_URL}":
            ngrok_url = os.getenv("NGROK_URL", "").strip()
            data["public_base_url"] = ngrok_url or None
        data.setdefault("schema_version", 2)
        if "channels" not in data and "providers" in data:
            channels = data.pop("providers")
            if isinstance(channels, dict):
                channels = {
                    name: {
                        key: val
                        for key, val in dict(channel).items()
                        if key != "recipients"
                    }
                    | (
                        {
                            "recipients": [
                                {
                                    k: v
                                    for k, v in dict(recipient).items()
                                    if k != "cameras"
                                }
                                for recipient in channel.get("recipients", [])
                            ]
                        }
                        if isinstance(channel, dict) and "recipients" in channel
                        else {}
                    )
                    for name, channel in channels.items()
                }
            data["channels"] = channels
        channels = data.get("channels")
        if isinstance(channels, dict):
            zalo = channels.get("zalo")
            if isinstance(zalo, dict) and zalo.get("public_base_url"):
                data.setdefault("public_base_url", zalo.get("public_base_url"))
                zalo = dict(zalo)
                zalo.pop("public_base_url", None)
                channels = dict(channels)
                channels["zalo"] = zalo
                data["channels"] = channels
        data.pop("cooldown", None)
        data.pop("enabled_in_config", None)
        return data

    @model_validator(mode="after")
    def validate_rules(self):
        rule_ids = [rule.id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Notification rule IDs must be unique")

        recipients = {
            "telegram": {r.id for r in self.channels.telegram.recipients},
            "zalo": {r.id for r in self.channels.zalo.recipients},
        }
        for rule in self.rules:
            for channel in ("telegram", "zalo"):
                unknown = set(getattr(rule.destinations, channel)) - recipients[channel]
                if unknown:
                    raise ValueError(
                        f"Rule {rule.id} references unknown {channel} recipients: "
                        + ", ".join(sorted(unknown))
                    )
        return self
