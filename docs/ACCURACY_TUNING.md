# Tuning the Personal-Object Matcher

Spaitra is an advanced multimodal ML pipeline built under a practical inference
budget. Its tuning work focused on making personal-object retrieval more useful
without turning every camera frame into an unbounded cascade of large models.

This document records how the matcher reached its current design, what the
historical evaluation revealed, and which methods could improve it further.

## How the Matcher Evolved

### 1. Visual instance retrieval

The first matching path used a detector crop and a visual embedding to compare
a scene region with stored object examples. DINOv3 became the visual backbone
because the task is instance retrieval: separating one wallet, receipt, bottle,
or key set from visually related objects.

### 2. OCR-aware multimodal memories

Visual appearance alone was not enough for receipts, labels, and similar
containers. PaddleOCR and a CLIP text encoder added a second evidence channel:

- DINOv3 supplies a 1,024-dimensional visual slot.
- CLIP supplies a 512-dimensional text slot when OCR finds useful text.
- Both slots are normalized before the final 1,536-dimensional representation
  is normalized again.
- OCR confidence scales the text contribution so weak text cannot dominate a
  strong visual match.

### 3. Feedback-based metric adaptation

User corrections were connected to the exact query and memory embeddings that
produced a scan result. Correct and incorrect relationships became triplets for
a small residual projection head:

- it begins as an identity transform;
- hard-negative mining selects the closest confusing negative;
- recent feedback can receive more training weight;
- its influence ramps with the amount of available feedback;
- raw stored embeddings remain unchanged, so personalization is reversible.

### 4. Measurable retrieval and latency

The benchmark grew into a controlled 120-image dataset with fixed splits,
negative examples, per-stage timing, top-k traces, and failure categories. OCR
batching and text-likelihood gating reduced unnecessary work, while normalized
embedding and benchmark-parity checks kept offline evaluation aligned with the
runtime pipeline.

### 5. Multiple prototypes and explicit decisions

Teach can retain several views of one label instead of collapsing every example
into one average. Retrieval scores stored examples, groups them by label, and
then applies a label-aware decision policy.

The scan path now exposes three separate stages:

- **Retrieve:** rank personal memories and retain top-k evidence.
- **Verify:** preserve a boundary for a future local-feature or patch-level
  check; the current implementation passes the candidate through unchanged.
- **Decide:** apply label-aware thresholds and an optional top-1/top-2 margin.

This separation matters because retrieval and acceptance can fail for different
reasons.

## Current Pipeline

- **Teach representation**
  - Grounding DINO crop refinement
  - Image-quality checks
  - DINOv3 visual embedding
  - Selective PaddleOCR and CLIP text embedding
  - Bounded prototype storage in SQLite
- **Scan representation**
  - YOLOE candidate proposals
  - Batched DINOv3 embeddings
  - Conditional OCR on text-like crops
  - The same normalized multimodal representation used during Teach
- **Matching**
  - Optional projection applied to queries and stored prototypes
  - Cosine ranking grouped by label
  - Baseline, personalized, and document threshold settings
  - Optional margin evidence
  - Deduplication, direction, depth, sightings, and narration after acceptance

## What the Historical Run Showed

A full-stack run recorded in May 2026 reported:

- top-3 recall: 0.55
- top-5 recall: 0.65
- top-10 recall: 1.00
- accepted precision: 0.30
- accepted match rate: 1.00
- abstention rate: 0.00

These are historical results, not current performance claims. The correct label
appearing in the top ten for every test query showed that the representation
retained useful retrieval signal. Accepting every query with low precision
showed that the decision policy did not yet convert that signal into a safe
operating point.

That diagnosis changed the tuning priority from replacing the entire retrieval
stack to improving calibration, uncertainty handling, and candidate
verification.

## Evaluation Strategy

The benchmark reports retrieval and product behavior separately:

- **Retrieval quality**
  - top-1 accuracy
  - top-3, top-5, and top-10 recall
  - true-item rank and ranked-candidate traces
- **Decision quality**
  - accepted precision
  - accepted match rate, which acts as coverage
  - correct accept rate
  - holdout false-positive rate
  - abstention rate
- **Operating behavior**
  - per-label and per-condition slices
  - precision/coverage and risk/coverage frontiers
  - fixed false-positive budget summaries
  - latency p50/p95 for changed stages
  - failure categories such as retrieval miss, bad crop, OCR disagreement,
    near-duplicate confusion, false reject, and false accept

Precision must be reported with coverage. A matcher that avoids errors only by
rejecting nearly every query is not useful.

## Hard-Case Workflow

Near-duplicate wallets and receipts are kept as explicit hard cases rather than
removed from the main evaluation. After a benchmark run:

```bash
scripts/run_hard_cases.sh
```

The workflow reads the result and trace artifacts, then writes a focused
Markdown and JSON report. A different archived run can be inspected with:

```bash
python scripts/build_hard_cases.py \
  --artifacts-dir /opt/spaitra/accuracy_hardening_baselines/main/frozen_baseline
```

The frozen trace does not contain raw OCR text. A `text_signal_failure` label
therefore means text evidence was enabled but the final match was wrong; it does
not prove that OCR transcription itself failed.

## Controlled Degradation

The robustness harness derives blur, JPEG compression, and Gaussian-noise
variants from the private benchmark images. It can produce per-level curve data
and optional SVG plots without changing the fixed 60/60 evaluation split.

```bash
PYTHONPATH=src python3 -m visual_memory.benchmarks.create_degraded \
  --dataset benchmarks/dataset.csv \
  --images benchmarks/images \
  --output benchmarks/degraded

PYTHONPATH=src python3 -m visual_memory.benchmarks.degradation_curves \
  --dataset-degraded benchmarks/dataset_degraded.csv \
  --results-csv benchmarks/results.csv
```

No degradation result should be presented without recording the code SHA,
source split, noise seed, parameters, and output artifacts together.

## Improvements Within the Inference Budget

These methods use evidence the current pipeline already computes:

- calibrate separate baseline, personalized, and document thresholds;
- choose operating points against explicit false-positive budgets;
- use the top-1/top-2 margin as uncertainty evidence instead of a blanket hard
  gate;
- improve prototype selection and weighting across stored viewpoints;
- use crop quality, OCR agreement, and retrieval rank in a lightweight decision
  model;
- collect targeted feedback from near-duplicate and low-margin cases.

They are the most practical next steps because they can improve the decision
layer without loading another large model for every candidate.

## Improvements With More Compute

The current `Verify` stage is intentionally an extension point. With a larger
inference budget, it could add:

- DINO patch-token comparison for the top few candidates;
- LightGlue or another local-feature verifier for texture and geometry;
- a learned reranker over query, prototype, OCR, quality, and margin signals;
- a larger embedding backbone or a small ensemble for the hardest object pairs.

These methods may improve fine-grained precision, but they also add GPU memory,
latency, and deployment complexity. The project therefore demonstrates the
full tuning process—measurement, diagnosis, controlled changes, and explicit
next experiments—without claiming that every high-cost extension belongs in
the current runtime.

## Reproducing a New Result

Before publishing new performance numbers:

1. Use the fixed split in `benchmarks/split_manifest.json`.
2. Record the code SHA, settings, hardware, command, and artifact directory.
3. Keep threshold selection separate from final holdout reporting.
4. Run the full stack twice and confirm deterministic accuracy metrics.
5. Report retrieval, decision, latency, and condition slices together.

The complete dataset and scoring contract are in
[Benchmark Specification](../benchmarks/BENCHMARK_SPEC.md).

## Methodology Sources

- [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [scikit-learn precision-recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Selective Classification](https://arxiv.org/abs/1705.08500)
- [Calibration](https://proceedings.mlr.press/v70/guo17a.html)
- [Google DELF retrieval](https://research.google/pubs/large-scale-image-retrieval-with-attentive-deep-local-features/)
- [Hard-negative metric learning](https://arxiv.org/abs/2007.12749)
