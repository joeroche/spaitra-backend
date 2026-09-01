# Spaitra Backend Architecture

Spaitra is a voice-first assistive vision backend for teaching, recognizing,
locating, and asking questions about personal objects. This document describes
the stable system shape. Benchmark execution details live in
[Matcher Tuning](ACCURACY_TUNING.md), and infrastructure setup lives in
[Deployment](DEPLOYMENT.md). The exact HTTP, Socket.IO, and state-machine
boundary with the companion client lives in
[Backend-Client Protocol](CLIENT_PROTOCOL.md).

## Runtime boundaries

The project uses two Python services because its primary model stacks have
different dependency requirements:

| Service | Responsibilities | Main stack |
| --- | --- | --- |
| Core | API, WebSockets, detection, embeddings, depth, retrieval, VLM, speech, persistence | Flask, Socket.IO, Torch |
| OCR | Text recognition behind a small HTTP boundary | FastAPI, PaddleOCR |

Shared application code is under `src/visual_memory`. The service entrypoints in
`services/` contain process setup rather than duplicate business logic.

## Request flow

- **Client input**
  - HTTP carries remember, scan, find, ask, item, feedback, and settings
    requests.
  - Socket.IO carries voice turns, session state, navigation, and narration.
- **API and session layer**
  - Validates input and preserves the backend-authoritative interaction state.
  - Routes work into the Remember, Scan, Find, or Ask path.
- **Inference pipelines**
  - Detection and DINOv3 provide visual candidates and representations.
  - The OCR service and CLIP add text evidence when a crop is text-like.
  - Direction and optional depth add spatial context after matching.
- **Memory and response**
  - SQLite supplies personal prototypes, feedback, settings, and sightings.
  - The backend returns structured results and concise narration.

## Core workflows

### Remember

The remember pipeline receives an image and a user label, detects or refines an
object crop, measures quality, extracts visual and optional text features, and
stores a memory record. Successful teaches persist the source image needed for
later inspection and verification. Multiple prototypes per label are supported
and bounded by settings.

Before a new label is stored, the pipeline can report high-similarity memories
as possible aliases. It does not merge labels automatically.

### Scan

The scan pipeline proposes objects, creates visual and OCR-aware embeddings,
retrieves ranked personal-memory candidates, applies the configured decision
policy, estimates direction and optional depth, records sightings, and returns
accessible narration.

Retrieval, optional verification, and decision policy are separate internal
stages. This lets the benchmark ask two different questions:

- Did retrieval surface the correct memory in top-k?
- Did the product policy accept, express uncertainty, or reject it?

That boundary prevents a strict threshold from hiding weak retrieval or low
coverage.

### Find and ask

Sightings record where an item was last observed so `find` can answer location
questions. `ask` routes scene and item questions through the available visual
language model while retaining object-memory context when relevant.

### Voice

The Socket.IO session layer coordinates recording, transcription, intent
routing, onboarding, pending image/location states, focused-item navigation,
and narration. HTTP endpoints remain available for direct clients and testing.

## Model and evidence stack

| Stage | Implementation |
| --- | --- |
| Prompt-free detection | YOLOE |
| Prompt-guided detection | Grounding DINO |
| Visual representation | DINOv3, 1,024 dimensions |
| Text representation | CLIP text encoder, 512 dimensions |
| OCR | PaddleOCR service |
| Depth | Apple Depth Pro |
| Speech recognition | Whisper large-v3-turbo |
| Visual questions | Moondream2 |
| Query parsing | Llama 3.2 through Ollama |

Visual and text vectors form a 1,536-dimensional combined representation. OCR
contribution scales with OCR confidence; missing or weak text evidence does not
receive the same weight as clean text.

Model sources and non-uniform third-party terms are listed in
[LICENSES.md](LICENSES.md).

## Personalization and storage

SQLite stores items, embeddings, OCR metadata, sightings, settings, and
feedback. Raw stored embeddings remain unchanged. The projection head is
trained from feedback and applied at match time, keeping personalization
reversible and preserving the original memory representation.

The feedback flow uses the scan cache to connect a correction with the exact
query embedding that produced the result. Training data uses positive and
negative relationships rather than overwriting the user's stored label.

## Decision and safety posture

Spaitra speaks object identity to blind users, so false positives matter. A
system that avoids errors by refusing most matches is also not useful. The
evaluation therefore reports precision beside coverage, correct accept rate,
abstention, false-positive rate, and top-k recall.

The top-1/top-2 margin remains available as evidence but is not a nonzero hard
gate by default. In a small personal database, a hard margin can silence similar
items before feedback has a chance to separate them. Near-duplicate wallets,
bottles, and receipts remain explicit hard cases.

## Benchmark protocol

The vision dataset contains 120 images: 10 labels photographed under all 12
combinations of three distances, two lighting levels, and two background
conditions.

The accuracy-hardening split is fixed by `benchmarks/split_manifest.json`:

- seed: 42;
- training: 60 images, 6 per label;
- test: 60 images, 6 per label;
- both partitions contain mixed capture conditions.

This is not a distance-held-out split. The older benchmark path that used
near-field reference images and farther test images is documented as a legacy
protocol in `benchmarks/BENCHMARK_SPEC.md`; it must not be confused with the
seeded hardening split.

Each full run can produce per-image results, top-k traces, failure taxonomy,
operating-point sweeps, fixed false-positive budget summaries, and latency
slices. The private raw images are not committed.

The last recorded full-stack frozen run completed in May 2026 on a dedicated
GPU container, but the on-demand endpoint later became unavailable. Current
final metrics require a fresh server rerun; local documentation does not claim
that historical results are current.

## Deployment shape

The current target is a GPU container with persistent storage. The repository,
virtual environments, model caches, weights, database, logs, benchmark assets,
and baselines live on the persistent volume. Supervisor manages core, OCR,
Ollama, and optional ingress processes because the container does not provide
`systemd`.

Operational addresses, credentials, SSH keys, private data, and machine-specific
paths belong in local environment files or an operator secret store, never in
committed documentation.

## Observability

Application logs are structured JSON records with timestamps, levels, modules,
events, and optional performance or memory fields. `visual_memory.utils.logparse`
provides filtering, summaries, and export without coupling the runtime to an
external logging service.

## Code map

```text
src/visual_memory/api/          routes, narration, and voice sessions
src/visual_memory/engine/       model wrappers and inference utilities
src/visual_memory/pipelines/    remember and scan orchestration
src/visual_memory/database/     SQLite storage
src/visual_memory/learning/     feedback and projection-head training
src/visual_memory/benchmarks/   benchmark, traces, and baseline comparison
src/visual_memory/tests/        unit, API, integration, and live checks
services/core/                  core process entrypoints
services/ocr/                   OCR process entrypoints
deploy/runpod/                  current container deployment scripts
```

## Known constraints

- Full model validation is server-only and depends on private model and dataset
  assets.
- The committed voice manifest references recordings that are not currently
  present.
- Similar objects remain the main retrieval and decision-policy challenge.
- OCR-heavy objects depend on legible text under the captured conditions.
- The project source license is still an explicit publication decision.
