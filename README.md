# Spaitra Backend

Built by Joe Roche as the backend and ML infrastructure for Spaitra, a
voice-first assistive vision system developed for TSA Software Development
2026.

This repository is the post-competition continuation of that backend. The
submitted version remains preserved in the
[original TSA repository](https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026).

## Pipeline evidence

![Authentic object-detection pipeline output](media/pipeline/scan-result-cropped.jpg)

This privacy-cropped output from the original demo pipeline shows the detector
operating on a real multi-object scene. It is authentic system output, not a
mock product screen or a claim about benchmark accuracy.

## What I built

Spaitra turns camera and voice input into a personal object memory. The backend
spans:

- prompt-free and prompt-guided object detection;
- DINOv3 image embeddings and CLIP text embeddings;
- OCR-aware retrieval and feedback-driven projection-head personalization;
- depth estimation, scene questions, and object-specific questions;
- remember, scan, find, ask, feedback, settings, and item-management APIs;
- a real-time Socket.IO voice state machine with Whisper transcription;
- SQLite persistence, sightings, structured logs, and benchmark tooling;
- separate Torch and PaddleOCR services for incompatible inference stacks.

## By the numbers

- 120 vision benchmark images
- 10 distinct personal objects
- 12 capture conditions per object: 3 distances x 2 lighting levels x 2 backgrounds
- Seeded 60/60 train-test split: 6 training and 6 test images per label
- 1,536-dimensional combined representation: 1,024 visual + 512 text features
- 2 isolated inference services: core and OCR
- 120-case voice evaluation manifest: 80 core + 40 extended cases

The voice manifest is committed, but its referenced recordings are not present
in this repository. Voice-result claims remain unverified until those assets are
recovered and rerun.

## System flow

```text
camera + voice
      |
      v
Flask / Socket.IO API ----> Whisper / intent routing
      |
      v
YOLOE or Grounding DINO ----> object crops ----> Depth Pro
      |
      +----> DINOv3 visual embedding
      +----> PaddleOCR ----> CLIP text embedding
                         |
                         v
              retrieval + personalization
                         |
                         v
             match, uncertainty, narration
```

The core API and OCR service run in separate environments so Torch-heavy vision
models and PaddleOCR can be deployed without forcing their dependency stacks
into one process. Shared application code lives under `src/visual_memory`.

## Backend-client boundary

The companion mobile client captures camera and microphone input, plays local
TTS and haptics, and renders the backend's current mode. The backend owns model
inference, persistence, intent routing, session transitions, and narration.
Clients consume a server-driven Socket.IO event stream rather than duplicating
those decisions. See the [backend-client protocol](docs/CLIENT_PROTOCOL.md) for
the HTTP surface, events, and state contract.

## Evaluation design

The accuracy-hardening benchmark uses the immutable
[`split_manifest.json`](benchmarks/split_manifest.json): seed 42, six training
and six test images for every label. Both halves contain mixed distance,
lighting, and background conditions. It is not a distance-held-out split.

The harness records top-k recall, accepted precision, accepted match rate,
correct accept rate, false-positive rate, abstention, per-condition slices, and
latency. Operating points are judged under explicit false-positive budgets so a
high-precision result cannot hide a system that rejects nearly everything.

The full raw benchmark images remain private because the source files contain
location metadata and some content requires privacy review. The committed CSV,
split, specification, and tooling preserve the evaluation design without
publishing those assets. See [Benchmark Specification](benchmarks/BENCHMARK_SPEC.md),
[Accuracy Tuning](docs/ACCURACY_TUNING.md), and
[Hard Cases Workflow](benchmarks/HARD_CASES.md).

![One object captured across all twelve controlled conditions](media/evaluation/12-condition-grid.jpg)

The publication-safe grid above uses one authentic object across all twelve
distance, lighting, and background combinations. The private archive retains
the complete 120-image corpus.

## Technical decisions

### Preserve raw memories while personalizing at match time

Stored embeddings remain reproducible inputs. A learned projection head is
applied during matching, allowing feedback-driven personalization without
irreversibly rewriting the memory database.

### Separate retrieval evidence from product policy

The benchmark distinguishes whether the correct object appears in top-k from
whether the decision stage accepts it. Thresholds, coverage, and false-positive
budgets are evaluated together instead of treating one accuracy number as the
whole system.

### Treat OCR confidence as evidence quality

OCR text contribution scales with OCR confidence. Low-confidence text cannot
receive the same influence as a clean label or receipt, and visual and text
signals are retained separately for failure analysis.

### Keep similar items visible as hard cases

Wallet-versus-wallet, bottle-versus-bottle, and receipt-versus-receipt errors
remain in the benchmark. The hard-case workflow extracts those confusions from
frozen traces rather than removing them or globally raising thresholds until
the system becomes silent.

## Code highlights

- [Scan retrieval pipeline](src/visual_memory/pipelines/scan_mode/pipeline.py)
- [Remember and personalization input pipeline](src/visual_memory/pipelines/remember_mode/pipeline.py)
- [Projection-head training](src/visual_memory/learning/projection_head.py)
- [Real-time voice session](src/visual_memory/api/voice_session.py)
- [Backend-client protocol](docs/CLIENT_PROTOCOL.md)
- [Full benchmark harness](src/visual_memory/benchmarks/full_benchmark.py)

## Repository layout

```text
services/core/             Flask and Socket.IO runtime
services/ocr/              PaddleOCR HTTP service
src/visual_memory/api/     HTTP and real-time interfaces
src/visual_memory/engine/  model wrappers and multimodal inference
src/visual_memory/pipelines/ remember and scan workflows
src/visual_memory/learning/ feedback and personalization
src/visual_memory/database/ persistence
src/visual_memory/benchmarks/ evaluation and reporting
deploy/runpod/             current container deployment surface
```

## Run locally

Accept access for the gated DINOv3 and Grounding DINO models before setup.

```bash
python3 -m venv .venv-core
source .venv-core/bin/activate
pip install -e ".[core]"
hf auth login
python setup_weights.py
```

Create OCR separately:

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install -e ".[ocr]"
python -m services.ocr.run
```

Then start the core service from the core environment:

```bash
export OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
python -m services.core.run
```

Model-backed tests and benchmarks require the documented server runtime and
private assets. Lightweight test entrypoints are listed in
[`src/visual_memory/tests/scripts/TESTING.md`](src/visual_memory/tests/scripts/TESTING.md).

## Deployment and limitations

The current deployment uses persistent container storage, isolated core/OCR
environments, and supervisor-managed services. See
[Deployment](docs/DEPLOYMENT.md) and [Architecture](docs/ARCHITECTURE.md).

Known limits:

- similar personal items remain difficult under distance, blur, or weak text;
- OCR-dependent objects need readable text for reliable discrimination;
- final current performance numbers require a fresh full-stack server rerun;
- raw image and audio datasets are intentionally absent from the public tree;
- a project source license has not yet been selected.

Third-party model terms are tracked in [LICENSES.md](docs/LICENSES.md), with
citations in [CITATIONS.bib](docs/CITATIONS.bib). Those notices do not license
this repository's own source code.
