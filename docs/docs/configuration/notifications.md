---
id: notifications
title: Notifications
---

import ConfigTabs from "@site/src/components/ConfigTabs";
import TabItem from "@theme/TabItem";
import NavPath from "@site/src/components/NavPath";

# Notifications

Frigate offers native notifications through WebPush, Telegram, and Zalo. All
providers use the same camera enablement, cooldown, and suspension policy.
WebPush uses the [WebPush Protocol](https://web.dev/articles/push-notifications-web-push-protocol)
and [VAPID spec](https://tools.ietf.org/html/draft-thomson-webpush-vapid).

:::info

Push notifications require internet access from the Frigate server to the browser vendor's push service (e.g., Google FCM, Mozilla autopush). See [Network Requirements](/frigate/network_requirements#push-notifications) for details.

:::

## Setting up Notifications

In order to use notifications the following requirements must be met:

- Frigate must be accessed via a secure `https` connection ([see the authorization docs](/configuration/authentication)).
- A supported browser must be used. Currently Chrome, Firefox, and Safari are known to be supported.
- In order for notifications to be usable externally, Frigate must be accessible externally.
- For iOS devices, some users have also indicated that the Notifications switch needs to be enabled in iOS Settings --> Apps --> Safari --> Advanced --> Features.

### Configuration

Enable notifications and fill out the required fields.

Optionally, change the default cooldown period for notifications. The cooldown can also be overridden at the camera level.

Notifications will be prevented if either:

- The global cooldown period hasn't elapsed since any camera's last notification
- The camera-specific cooldown period hasn't elapsed for the specific camera

#### Global notifications

<ConfigTabs>
<TabItem value="ui">

1. Navigate to <NavPath path="Settings > Notifications > Notifications" />.
   - Set **Email** to your email address
   - Enable notifications for the desired cameras

</TabItem>
<TabItem value="yaml">

```yaml
notifications:
  enabled: true
  email: "johndoe@gmail.com"
  cooldown: 10 # wait 10 seconds before sending another notification from any camera
  providers:
    webpush:
      enabled: true
    telegram:
      enabled: true
      recipients:
        - id: security_team
          name: Security Team
          chat_id: "-100123456789"
          cameras: [doorbell]
    zalo:
      enabled: true
      public_base_url: https://camera.example.com
      media_url_ttl: 300
      recipients:
        - id: operators
          name: Operators
          chat_id: "123456789"
          cameras: [doorbell]
  delivery:
    max_attempts: 5
    initial_backoff: 5
    max_backoff: 300
    retention_days: 7
    max_pending: 5000
```

</TabItem>
</ConfigTabs>

#### Per-camera notifications

<ConfigTabs>
<TabItem value="ui">

1. Navigate to <NavPath path="Settings > Camera configuration > Notifications" /> and select the desired camera.
   - Set **Enable notifications** to on
   - Set **Cooldown period** to the desired number of seconds to wait before sending another notification from this camera (e.g. `30`)

</TabItem>
<TabItem value="yaml">

```yaml
cameras:
  doorbell:
    ...
    notifications:
      enabled: True
      cooldown: 30 # wait 30 seconds before sending another notification from the doorbell camera
      providers: [webpush, telegram, zalo]
```

</TabItem>
</ConfigTabs>

If `providers` is omitted for a camera, only WebPush is selected. An empty
recipient `cameras` list grants that recipient all notification-enabled cameras.
Provider and recipient enablement must also be on.

Telegram and Zalo tokens are secrets and are never stored in YAML. Set them in
the Frigate process environment:

```dotenv
FRIGATE_TELEGRAM_BOT_TOKEN=
FRIGATE_ZALO_BOT_TOKEN=
```

Telegram uploads event snapshots directly. Zalo receives a short-lived signed
snapshot URL. Without `public_base_url`, Zalo remains available in degraded
text/link mode.

### Registration

Once notifications are enabled, press the `Register for Notifications` button on all devices that you would like to receive notifications on. This will register the background worker. After this Frigate must be restarted and then notifications will begin to be sent.

## Supported Notifications

Native notifications support review alerts, semantic triggers, camera monitoring
alerts, test notifications, and finalized car events with a recognized license
plate.

:::note

Currently, only Chrome supports images in notifications. Safari and Firefox will only show a title and message in the notification.

:::

## Reduce Notification Latency

Different platforms handle notifications differently, some settings changes may be required to get optimal notification delivery.

### Android

Most Android phones have battery optimization settings. To get reliable Notification delivery the browser (Chrome, Firefox) should have battery optimizations disabled. If Frigate is running as a PWA then the Frigate app should have battery optimizations disabled as well.
