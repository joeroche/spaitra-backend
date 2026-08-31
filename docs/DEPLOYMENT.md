# Deployment

Spaitra's current deployment target is a GPU container with persistent storage
and supervisor-managed processes. This guide intentionally omits live hosts,
credentials, private paths, and one-time migration details.

## Services

| Process | Purpose | Default local listener |
| --- | --- | --- |
| Core | Flask API and Socket.IO | `127.0.0.1:5000` |
| OCR | PaddleOCR HTTP service | `127.0.0.1:8001` |
| Ollama | Local query model | `127.0.0.1:11434` |
| Ingress | Optional HTTPS/WSS tunnel or reverse proxy | operator-defined |

The core process must use one worker because loaded model and session state are
process-local. Core and OCR use separate virtual environments.

## Persistent layout

Keep every stateful or expensive asset on the provider's persistent volume:

```text
<persistent-root>/
  .env
  .ocr.env
  backend/
  venv-core/
  venv-ocr/
  cache/
  checkpoints/
  models/
  data/
  logs/
  benchmark-images/
  benchmark-baselines/
```

Do not place secrets, databases, raw benchmark assets, or model caches in Git.
Do not rely on a container overlay filesystem for long-lived state.

## Setup

From the repository on the target host:

```bash
bash deploy/runpod/preflight.sh
bash deploy/runpod/bootstrap.sh
bash deploy/runpod/validate_runtime.sh
```

The bootstrap creates or repairs the isolated environments, installs the shared
package with the appropriate extras, prepares persistent directories, and
starts supervised services. Review `deploy/env.example` and
`deploy/ocr.env.example`, then place real values in operator-managed files
outside the repository.

Model access must be accepted before weight setup for gated Hugging Face models.
Downloaded weights and caches stay on persistent storage.

## Resume and benchmark

The on-demand resume path is:

```bash
bash deploy/runpod/resume_and_benchmark.sh
```

To request the full benchmark after runtime repair and validation:

```bash
RUN_BENCHMARK=1 BENCHMARK_THREADS=4 \
  bash deploy/runpod/resume_and_benchmark.sh
```

The benchmark must use a persistent absolute `BASELINE_ROOT`. Baseline
directories are append-only and must record code SHA, settings, command, split,
and artifact paths.

## Validation gates

Before treating a deployment as ready, verify:

1. GPU visibility and available memory.
2. Persistent volume availability and write permissions.
3. Required model weights and caches.
4. OCR and core health endpoints.
5. Supervisor status for required processes.
6. Lightweight unit and API suites.
7. Live WebSocket and audio checks with an explicitly supplied test URL.
8. Two full benchmark runs with matching accuracy metrics.

Model-backed tests, live audio checks, and benchmarks run on the active GPU
server. Do not substitute the retired legacy server without explicit approval.

## Security and data boundaries

- Supply API keys, encryption keys, Hugging Face tokens, and SSH credentials at
  runtime; never commit them.
- Require the live WebSocket test URL through environment or CLI configuration.
- Keep operator access documentation private.
- Use HTTPS/WSS ingress for public clients.
- Back up the database, exemplars, environment files, and baseline artifacts
  outside the container volume.
- Treat raw benchmark images and recordings as private until content, metadata,
  and rights reviews are complete.

## Legacy compatibility

Some top-level service and cutover helpers remain for compatibility with older
deployments. They are not the recommended container path; use the supervised
container flow above for current deployments.
