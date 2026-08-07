---
id: notifications
title: Notifications
---

# Notifications

Frigate uses one rule engine for WebPush, Telegram, and Zalo. The complete
notification document is stored under `notifications` in the active Frigate
YAML. In this deployment that file is `deploy/config.yaml`; the dashboard saves
directly to that file and hot-reloads the notification client.

## Configuration

```yaml
notifications:
  schema_version: 2
  enabled: true
  email: admin@example.com
  channels:
    webpush:
      enabled: true
    telegram:
      enabled: true
      recipients:
        - id: security_team
          name: Security Team
          chat_id: "{FRIGATE_TELEGRAM_CHAT_ID}"
          enabled: true
    zalo:
      enabled: true
      public_base_url: https://camera.example.com
      media_url_ttl: 300
      recipients:
        - id: operators
          name: Operators
          chat_id: "{FRIGATE_ZALO_CHAT_ID}"
          enabled: true
  rules:
    - id: door_alert
      name: Door alert
      enabled: true
      event: alert
      filters:
        cameras: [doorbell]
        labels: [person]
        zones: [front_porch]
      destinations:
        webpush: true
        telegram: [security_team]
        zalo: [operators]
      cooldown: 30
  delivery:
    max_attempts: 5
    initial_backoff: 5
    max_backoff: 300
    retention_days: 7
    max_pending: 5000
```

An empty `filters.cameras` list means every camera. Rules, rather than camera
configuration or recipient configuration, decide which event goes to which
destination. Cooldown is isolated by rule, camera, channel, and recipient.

Supported rule events are:

- `alert`: a review first reaches Alert severity.
- `object_detected`: a confirmed tracked object appears or first enters a
  matching zone.
- `license_plate`: a car passage ends with a normalized plate.
- `face_recognized`: face identity and media have been committed.
- `camera_offline`: the detect stream stays offline for 30 seconds.
- `camera_online`: a camera recovers after a notified offline transition.
- `semantic_trigger`: a semantic trigger with the notification action.
- `camera_monitoring`: a camera monitoring/VLM Watch result.

The camera Notifications page is read-only and shows effective rules. Its
Suspend action is runtime state and blocks every selected channel for that
camera.

## Secrets and provider behavior

Tokens are never stored in YAML or returned by the API. Set them in the process
environment:

```dotenv
FRIGATE_TELEGRAM_BOT_TOKEN=
FRIGATE_ZALO_BOT_TOKEN=
```

Telegram uploads the snapshot directly. Zalo uses a short-lived signed media
URL; without `public_base_url`, it sends text/link notifications in degraded
mode. Telegram and Zalo deliveries use the durable SQLite outbox and resume
after restart.

## WebPush registration

WebPush requires HTTPS and a supported browser. Register each browser/device
from Settings → Notifications. Browser push services also require outbound
internet access. Chrome supports notification images; Safari and Firefox may
show only title and message.

## Migration

Legacy `providers`, recipient camera lists, and per-camera notification blocks
are backed up and automatically converted once to schema v2. Migration creates
only rules matching the former Alert, Semantic, Monitoring, and social LPR
behavior; new object, face, and camera-health rules remain opt-in.
