# Spaitra

**A voice-first system that learns and recalls the objects that matter to one
person.**

Generic recognition can identify a wallet. Spaitra is designed to learn which
wallet is yours, recognize it in a new scene, remember where it was seen, and
answer through voice. It combines personal-object retrieval, spatial context,
and persistent memory for blind and low-vision users.

I built Spaitra's backend and ML system: the vision pipelines, multimodal
retrieval and personalization, voice state machine, persistence, evaluation
tooling, and GPU deployment. The complete application began as a team project
for the 2026 TSA Software Development event; the mobile client was shared team
work.

## What It Does

- **Teach**
  - The user names an object and photographs it.
  - Grounding DINO isolates the named object.
  - DINOv3 and optional OCR/CLIP features create a reusable personal memory.
- **Scan**
  - YOLOE proposes candidate regions without assuming which object is present.
  - Each crop is compared with the user's stored object memories.
  - Accepted matches include direction, optional distance, and accessible
    narration.
- **Ask**
  - Sightings answer questions such as "Where did I leave my wallet?"
  - OCR and the visual-language path handle documents, objects, and scene
    questions.

## Personal-Object Matching

- **Candidate generation**
  - YOLOE finds broad regions during Scan.
  - Grounding DINO uses the supplied label to isolate an object during Teach.
- **Multimodal representation**
  - DINOv3 produces a 1,024-dimensional visual embedding for each crop.
  - Text-like crops are sent to PaddleOCR.
  - OCR confidence controls the contribution of a 512-dimensional CLIP text
    embedding.
  - The normalized visual and text slots form a 1,536-dimensional memory.
- **Retrieval and decision**
  - Cosine similarity ranks stored prototypes by label.
  - Label-aware thresholds and an optional top-1/top-2 margin determine whether
    the best result is accepted or rejected.
  - Accepted regions are deduplicated, spatially ordered, and optionally passed
    through Depth Pro before narration.
- **Personalization**
  - Correct and incorrect feedback retains the query-memory relationship that
    produced the result.
  - Hard-negative mining forms triplets from similar objects.
  - A residual projection head adapts matching without overwriting the original
    embeddings in SQLite.

## Real Pipeline Evidence

![YOLOE candidate regions in a multi-object desk scene](media/pipeline/scan-result-cropped.jpg)

The boxes above are real YOLOE proposals from a multi-object demo scene. They
are intentionally unlabeled: the detector finds possible objects, while the
personal-memory pipeline determines identity.

The evaluation set tests the matcher beyond one favorable scene:

- 120 images across 10 personal objects;
- 12 conditions per object;
  - 1, 3, and 6 feet;
  - bright and dim lighting;
  - clean and cluttered backgrounds;
- a fixed seed-42 split;
  - 60 images for reference construction and projection-head training;
  - 60 held-out test images;
  - six train and six test images per label with no file overlap.

![One personal object photographed across all twelve controlled conditions](media/evaluation/12-condition-grid.jpg)

The benchmark records retrieval rank, accepted precision, coverage,
false-positive rate, abstention, condition slices, failure categories, and
latency. This makes the tuning target visible: whether the correct memory was
missing from the candidate list, or retrieved successfully but rejected or
misclassified by the final decision policy.

The historical full-stack run showed strong top-k retrieval headroom but weak
final acceptance precision. That result drove the current separation between
retrieval, verification, and decision stages. It is evidence from an earlier
checkpoint, not a claim about current performance. See the
[benchmark specification](benchmarks/BENCHMARK_SPEC.md) and
[matcher-tuning history](docs/ACCURACY_TUNING.md).

## System

- **Mobile client**
  - Captures camera frames and microphone audio.
  - Plays local TTS, haptics, and the interface state sent by the backend.
  - Communicates through HTTP and Socket.IO.
- **Core service**
  - Flask API and backend-authoritative Socket.IO session state.
  - Grounding DINO, YOLOE, DINOv3, CLIP, Depth Pro, Whisper, and visual-language
    inference.
  - Deterministic command routing with bounded Ollama interpretation when
    wording needs disambiguation.
- **OCR service**
  - FastAPI boundary around PaddleOCR.
  - Runs separately because the Torch and Paddle dependency stacks conflicted
    during integration.
- **Persistence**
  - SQLite stores prototypes, OCR metadata, feedback, projection weights,
    settings, and last-seen sightings.

The backend owns inference, persistence, intent routing, session transitions,
and narration decisions. The [client protocol](docs/CLIENT_PROTOCOL.md)
documents the server-driven event stream and the exact mobile boundary.

## Engineering Highlights

- Instance retrieval that distinguishes a specific taught object from other
  members of the same category.
- Conditional OCR that adds text evidence when useful without paying its full
  cost on every crop.
- Multiple prototypes per label for different viewpoints.
- Reversible feedback learning through a residual projection head and
  hard-negative triplet training.
- State-aware Whisper prompts built from known labels, recent rooms, and the
  current interaction mode.
- Model lifecycle controls that trade GPU memory between mutually exclusive
  Teach and Scan workloads.
- Structured logging, bounded scan caches, reproducible benchmark artifacts,
  and persistent GPU-container deployment.

## Code Tour

- [Scan pipeline](src/visual_memory/pipelines/scan_mode/pipeline.py): proposals,
  embeddings, selective OCR, retrieval, decisions, depth, and narration
- [Teach pipeline](src/visual_memory/pipelines/remember_mode/pipeline.py):
  prompt-guided crops, quality checks, representations, and prototype storage
- [Combined representation](src/visual_memory/engine/embedding/embed_combined.py):
  normalized visual/text slots and confidence-weighted OCR contribution
- [Projection head](src/visual_memory/learning/projection_head.py) and
  [trainer](src/visual_memory/learning/trainer.py): residual metric adaptation
  and triplet learning
- [Voice session](src/visual_memory/api/voice_session.py) and
  [Socket.IO routes](src/visual_memory/api/routes/voice_ws.py): real-time
  interaction state
- [Full benchmark](src/visual_memory/benchmarks/full_benchmark.py): fixed splits,
  retrieval traces, operating points, failure analysis, and latency
- [Architecture](docs/ARCHITECTURE.md): runtime boundaries, model lifecycle,
  storage, and API flow

## Run Locally

Accept access to the gated DINOv3 and Grounding DINO checkpoints before setup.

```bash
python3 -m venv .venv-core
source .venv-core/bin/activate
pip install -e ".[core]"
hf auth login
python setup_weights.py
```

Create the isolated OCR environment:

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
pip install -e ".[ocr]"
python -m services.ocr.run
```

Then start the core service:

```bash
export OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
python -m services.core.run
```

Model-backed execution requires a GPU and local checkpoints. Lightweight test
entrypoints are listed in [Testing](src/visual_memory/tests/scripts/TESTING.md),
with runtime details in [Deployment](docs/DEPLOYMENT.md).

## Current Constraints

- Fine-grained matching remains difficult at distance, under blur, and when
  discriminating text is unreadable.
- The current verifier stage is a placeholder. Local-feature verification,
  larger embeddings, or a learned reranker could improve precision, but the
  project was constrained by GPU memory and per-frame inference cost.
- Current performance requires a fresh full-stack benchmark run; historical
  results are retained only to explain the tuning process.
- Full execution depends on gated checkpoints and a GPU-capable environment.
- The repository has no source-code license, and its model stack includes
  component-specific terms tracked in [LICENSES.md](docs/LICENSES.md).

## Project History

This repository continues the backend after the TSA event. The frozen 2026
competition submission remains in the
[original team repository](https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026).
