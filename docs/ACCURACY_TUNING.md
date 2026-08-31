# Accuracy Tuning

This document records the project-level tuning process for scan/retrieval
accuracy.

## Current State

The current scan stack uses YOLOE proposals, DINOv3 image embeddings, CLIP text
embeddings for OCR, a combined 1536-dimensional vector, optional ProjectionHead
personalization, deduplication, and confidence-proportional OCR text weighting.
The configurable top-1/top-2 margin gate is zero by default; the margin remains
available in traces for tuning.

The benchmark dataset is a 120-image instance-level set with 10 labels across
distance, lighting, and background conditions. Its accuracy-hardening split is
fixed at seed 42 with 60 training and 60 test images, six of each per label.
Both partitions contain mixed capture conditions; this is not a
distance-held-out split.

A full-stack run and repeat comparison were recorded in May 2026. That run
reported:
- `top_3_recall=0.55`
- `top_5_recall=0.65`
- `top_10_recall=1.00`
- `accepted_precision=0.30`
- `accepted_match_rate=1.00`
- `correct_accept_rate=0.30`
- `abstention_rate=0.00`

These numbers describe that historical run, not the current code. The on-demand
endpoint later became unavailable, so any new performance report requires a
fresh server run with the code SHA, settings, hardware, command, split, and
artifact paths recorded together.

The historical interpretation was that retrieval coverage was strong
(`top_10_recall=1.00`) while decision-policy precision was weak. That result
points to decision and verifier work before replacement of the retrieval stack.

## Tuning Goal

Optimize for high useful recognition under explicit false-positive budgets.
False positives matter because the app speaks object identity to blind users,
but a system that avoids mistakes by rejecting most matches is also not useful.
Near-identical items such as similar wallets, same-brand bottles, or receipts
with similar OCR are expected hard cases. They should be measured, reduced, and
surfaced as uncertainty where appropriate, not used as a reason to drive recall
or accepted-match rate toward zero.

## Required Metrics

Every matching or decision-policy run should report:

- `top_1_accuracy`
- `top_3_recall` and `top_10_recall`
- true-item rank and optional MRR/mAP-style retrieval metrics when top-k traces
  are available
- `accepted_precision`
- `accepted_match_rate`
- `correct_accept_rate`
- `holdout_fp_rate`
- `abstention_rate`
- per-label and per-condition metrics
- latency p50/p95 for changed stages
- calibration metrics such as ECE or Brier score if probability-like confidence
  is used
- fixed-FP-budget summaries, for example best useful result at FP budgets 0.05
  and 0.10
- precision/recall and risk/coverage operating-point data

`accepted_match_rate` is the coverage metric for this project. It must be
reported beside accepted precision so high-precision, low-coverage runs cannot
look successful.

## Workflow

1. Freeze the dataset split before tuning.
2. Keep training, calibration, and holdout roles separate.
3. Sweep thresholds and verifier cutoffs broadly enough to show the operating
   frontier.
4. Pick thresholds on training/calibration data or cross-validation.
5. Report final numbers once on the pinned holdout split.
6. Compare retrieval, verifier, and decision-policy bottlenecks separately.
7. Keep hard negatives and near-duplicate cases in a separate hard-cases set.
8. Reject changes that improve false positives only by collapsing coverage.

Reports should use specific failure labels instead of treating every mistake as
the same kind of false positive. At minimum, distinguish retrieval misses,
threshold false rejects, near-duplicate confusion, OCR collision or disagreement,
bad crops, lighting or distance failures, verifier false accepts/rejects, and
over-abstention.

## Controlled degradation tooling

The repository also retains an earlier robustness harness for controlled image
degradation. `create_degraded.py` can derive blur, JPEG compression, and
Gaussian-noise variants from the private benchmark images, while
`degradation_curves.py` groups matching results by degradation level and writes
curve data and optional SVG figures.

This tooling is separate from the fixed 60/60 hardening result. The generated
images and curve outputs are not committed, and no degradation result should be
claimed until the current code SHA, source split, random seed for noise,
parameters, and output artifacts are captured together. The default generator
currently covers blur radii 1-5, JPEG quality 30/50/70/90, and Gaussian noise
standard deviations 0.01/0.02/0.05/0.10.

```bash
PYTHONPATH=src python3 -m visual_memory.benchmarks.create_degraded \
  --dataset benchmarks/dataset.csv \
  --images benchmarks/images \
  --output benchmarks/degraded

PYTHONPATH=src python3 -m visual_memory.benchmarks.degradation_curves \
  --dataset-degraded benchmarks/dataset_degraded.csv \
  --results-csv benchmarks/results.csv
```

## Next Validation Gate

- Restore or replace the active GPU benchmark runtime.
- Rerun the frozen configuration and verify deterministic accuracy metrics.
- Complete the pending OCR-agreement variant benchmark and hard-case report.
- Close remaining preprocessing, memory-quality, multi-prototype, and
  decision-stage parity checks against the frozen reference.
- Publish a final operating point only after a current full-stack run.

## Methodology Sources

- scikit-learn threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- scikit-learn precision-recall: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- Google ML ROC/AUC guidance: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- NIST FRVT verification reporting: https://pages.nist.gov/frvt/html/frvt11.html
- Selective Classification: https://arxiv.org/abs/1705.08500
- Calibration: https://proceedings.mlr.press/v70/guo17a.html
- Google DELF retrieval: https://research.google/pubs/large-scale-image-retrieval-with-attentive-deep-local-features/
- Google Landmarks Dataset v2: https://research.google/pubs/google-landmarks-dataset-v2-a-large-scale-benchmark-for-instance-level-recognition-and-retrieval/
- Hard negative metric learning: https://arxiv.org/abs/2007.12749
