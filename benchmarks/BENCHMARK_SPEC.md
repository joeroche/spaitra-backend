# Benchmark Specification

Spaitra's benchmark measures personal-object retrieval and final acceptance
under controlled changes in distance, lighting, and background. Raw images are
private; the dataset manifest, fixed split, scoring contract, and evaluation
code are retained in the repository.

## Dataset

- 120 images
- 10 distinct personal-object labels
- 12 conditions per label
  - distances: 1, 3, and 6 feet
  - lighting: bright and dim
  - backgrounds: clean and cluttered

The labels include two wallets and two receipts as distinct objects. Confusing
one wallet or receipt for the other is a false positive, not partial credit.

Files use this naming convention:

```text
{label}_{distance}ft_{lighting}_{background}.jpg
```

## Capture Protocol

To keep conditions comparable:

1. Mark fixed camera distances at 1, 3, and 6 feet.
2. Use one clean surface and one naturally cluttered surface.
3. Capture all bright images before changing to the dim setup.
4. Use the rear camera without zoom or flash.
5. Keep object orientation and camera orientation consistent.
6. Tap the object to focus before each capture.
7. Redact private receipt text before adding images to the benchmark set.
8. Validate the complete set before running inference.

```bash
python -m visual_memory.benchmarks.check_dataset
```

## Fixed Split

The accuracy-hardening split is defined by `benchmarks/split_manifest.json`:

- seed: 42
- training: 60 images, six per label
- test: 60 images, six per label
- no file overlap
- mixed distances, lighting, and backgrounds in both partitions

This is not a distance-held-out split. The manifest is authoritative and should
not be regenerated when comparing tuning changes.

An older benchmark path uses one near-field reference image per label, all
1-foot images for projection training, and the 3- and 6-foot images for testing.
Results from that legacy protocol must be labeled separately from the fixed
60/60 split.

## Scoring

- **Correct accept:** the accepted label matches the ground-truth object.
- **False accept:** the system accepts the wrong label or accepts a known label
  for a no-match distractor.
- **Reject:** no candidate satisfies the decision policy.
- **Uncertain:** the system abstains without counting the query as correct or as
  a false accept.
- **Top-k hit:** the correct label appears among the first `k` retrieved labels,
  regardless of the final decision.

## Required Metrics

- top-1 accuracy
- top-3, top-5, and top-10 recall
- accepted precision
- accepted match rate
- correct accept rate
- holdout false-positive rate
- abstention rate
- per-label and per-condition results
- latency p50/p95
- ranked-candidate traces and failure categories
- operating points at false-positive budgets of 0.05 and 0.10

Accepted precision and accepted match rate must be reported together so that a
high-precision, low-coverage policy cannot appear complete.

## Hard Cases and Distractors

The benchmark keeps near-duplicate wallets and receipts in the primary result.
Additional negative inputs include unrelated objects and same-category objects
that were never taught to the system.

After a full run, the focused hard-case report can be generated with:

```bash
scripts/run_hard_cases.sh
```

The tuning rationale and hard-case interpretation are documented in
[Tuning the Personal-Object Matcher](../docs/ACCURACY_TUNING.md).

## Running the Benchmark

Fast smoke test without depth or OCR:

```bash
python -m visual_memory.benchmarks.full_benchmark \
  --dataset benchmarks/dataset.csv \
  --images benchmarks/images \
  --seed 42 --no-depth --no-ocr --epochs 5
```

Full model-backed run:

```bash
bash scripts/run_benchmark.sh
```

The full run requires the private images, gated model weights, and a
GPU-capable environment.

## Artifacts

Each run can produce:

- `benchmarks/results.csv`: per-query metrics and latency
- `benchmarks/results.json`: run metadata and aggregate results
- `benchmarks/inference_traces.csv`: ranked candidates and component scores
- `benchmarks/failure_taxonomy.json`: categorized failures
- `benchmarks/projection_head_bench.pt`: projection weights trained for the run
- `benchmarks/hard_cases_report.md`: focused review of difficult queries

Archived baselines are append-only. A publishable result must record its code
SHA, settings, hardware, command, fixed split, and artifact paths together.
