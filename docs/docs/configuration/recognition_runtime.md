---
id: recognition_runtime
title: Recognition Runtime
---

Face and license-plate recognition run locally by default. Set `recognition.runtime`
to `external` to move model inference, voting, and track session state into the
dedicated recognition container. External mode is fail-closed: Frigate does not
run a local model or resubmit a rejected job when the service is unavailable.

```yaml
recognition:
  runtime: external
  endpoint: recognition:50051
  deadline: 5
  observation_capacity: 128
  control_capacity: 64
  outcome_capacity: 128
  shutdown_drain: 10
  tls:
    ca: /config/recognition-tls/ca.crt
    certificate: /config/recognition-tls/client.crt
    key: /config/recognition-tls/client.key
    server_name: recognition
```

The service container needs its server certificate and key, the client CA, and
an allowlist containing the Frigate client certificate identity. A non-loopback
service bind without mTLS is rejected at startup. There is no plaintext fallback.

Frigate remains the owner of tracking, raw evidence lifetime, Event publication,
and face snapshot/attempt media. The service owns the recognition models, voting
state, track sessions, and face classifier library. Each service start creates a
new epoch; Frigate discards outcomes from an earlier epoch.

The Compose `recognition` profile builds the `recognition-runtime` image target.
It intentionally publishes no host port. Provide the certificate files under
`config/recognition-tls` before enabling the profile:

```bash
docker compose --profile recognition up --build recognition
```
