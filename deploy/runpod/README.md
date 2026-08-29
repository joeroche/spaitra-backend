# RunPod Deploy Surface

Active RunPod deployment helpers:

- `bootstrap.sh` builds or repairs the Pod runtime and starts supervised services.
- `resume_and_benchmark.sh` is the preferred pod-resume operator path. It stops
  expensive services during repair, runs idempotent bootstrap, repairs OCR
  dependencies only when needed, downloads only missing overlay-lost assets such
  as `depth_pro.pt`, validates runtime, and can start the benchmark with
  `RUN_BENCHMARK=1`.
- `preflight.sh` checks GPU, storage, tooling, and auth prerequisites.
- `validate_runtime.sh` verifies ownership, health, ports, and supervisor state.
- `spaitra-ctl` is the operator wrapper around `supervisorctl`.

Typical on-demand pod resume:

```bash
cd /opt/spaitra/backend-copy
RUN_BENCHMARK=1 bash deploy/runpod/resume_and_benchmark.sh
```

For setup-only repair, omit `RUN_BENCHMARK=1`. The script writes resumable
state to `/opt/spaitra/logs/runpod/resume_status.env` and the benchmark log path
there when a run is started.
