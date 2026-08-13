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
  job_deadline: 30
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

`deadline` bounds connection and RPC operations. `job_deadline` bounds an accepted
observation, including bounded queue wait and inference. A full queue is rejected
explicitly; Frigate does not block, evict older work, or retry through a local model.

Frigate remains the owner of tracking, raw evidence lifetime, Event publication,
and face snapshot/attempt media. The service owns the recognition models, voting
state, track sessions, and face classifier library. Face detection, crop geometry,
bbox rendering, LPR processing, voting, and publication decisions use the same
modules as local mode; the runtime changes only evidence/job transport and where
the synchronous core executes. Each service start creates a new epoch; Frigate
discards outcomes from an earlier epoch.

The Compose `external-recognition` profile starts the `camera-recognition` image.
It intentionally publishes no host port. `deploy/run.ps1` selects the profile from
`recognition.runtime`; there is no command-line switch and the two topologies never
run together. Set `RECOGNITION_TLS_DIR` to a host directory containing `ca.crt`,
`server.crt`, `server.key`, `client.crt`, and `client.key`.

Use the bounded overlay build so the cached base image is reused, then start the
runtime selected in the config:

```powershell
.\deploy\run.ps1 build
.\deploy\run.ps1 doctor
.\deploy\run.ps1 start
```

In local mode, only the `frigate` container is expected. In external mode,
`camera-recognition` must also be running and healthy, while Frigate logs must not
show local Face or LPR model initialization.

Runtime evidence is producer-owned. When capture is enabled, the shared Face code
returns the original recognition frame, a producer-rendered person/face bbox frame,
and the exact classifier crop. The report validator verifies hashes and geometry; it
does not synthesize records or draw replacement boxes.

The repository uses one parameterized runtime harness. Run the normal local topology
with `tools/tests/e2e/run_platform_runtime_test.py`; the external convenience entrypoint
`tools/tests/e2e/run_external_recognition_runtime_test.py` calls the same harness with
`--topology recognition`.
