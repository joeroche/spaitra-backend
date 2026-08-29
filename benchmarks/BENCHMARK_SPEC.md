# Benchmark Specification

## Dataset

120 images across 10 object labels, 12 conditions per label.

### Labels (10)

| Label | Coarse category |
|---|---|
| glasses_prescription | personal |
| keys_house | personal |
| keys_safe | personal |
| magnesium_bottle | container |
| receipt_eye_doctor | paper |
| receipt_salon | paper |
| sunglasses_sun | personal |
| wallet_trifold | personal |
| wallet_zipper | personal |
| water_bottle | container |

### Condition matrix (12 per label)

Each label is photographed under every combination of:

- Distance: 1ft, 3ft, 6ft
- Lighting: bright, dim
- Background: clean, messy

Total: 3 x 2 x 2 = 12 conditions x 10 labels = 120 images.

File naming convention: `{label}_{distance}ft_{lighting}_{background}.{ext}`

### Label count note

The plan originally mentioned "8 object types." The actual dataset has 10 labels.
Two wallet variants (wallet_trifold, wallet_zipper) and two receipt variants
(receipt_salon, receipt_eye_doctor) are each treated as distinct objects. No label
collapse or merging is applied. The benchmark evaluates 10-way retrieval.

---

## Train / Test Split

Defined in `benchmarks/split_manifest.json` (generated with seed=42, train_per_label=6).

- 60 train images (6 per label): reference DB construction + projection head training
- 60 test images (6 per label): retrieval evaluation

The split is random per label (seed=42). It is NOT stratified by condition, so each
label's train set may include a mix of distances, lighting, and backgrounds. The
manifest is the authoritative definition. Do not regenerate it or change the seed.

### Legacy benchmark split (pre-hardening)

The existing `full_benchmark.py` uses a different split:
- Reference DB: 1ft_bright_clean image per label (1 per label)
- Projection head training: all 1ft images (4 per label)
- Test: all 3ft and 6ft images (8 per label)

The new split_manifest.json is for the accuracy hardening workflow. Both splits
coexist. The hardening plan uses split_manifest.json. Existing benchmark runs
continue to use the legacy split until full_benchmark.py is updated (Step 0.2).

---

## Scoring Rules

### Correct match

A query is a correct match if the system returns the true label as the accepted
result (decision = "accept") and that label matches the query's ground truth label.

### False positive (FP)

A query is a false positive if the system accepts a label that does not match
the ground truth label. This includes:
- Accepting any wrong label (cross-category or within-category)
- Accepting a match when the correct answer is "no match" (distractor queries)

### False negative / abstention

A query is a false negative if:
- The true label was above threshold but rejected by the margin gate
  (failure mode: margin_false_reject)
- The true label was below threshold (failure mode: threshold_false_reject or
  retrieval_miss)

A query that produces decision = "uncertain" is counted separately from
"reject." Uncertain counts toward abstention_rate but is not a FP or FN.

### Uncertain handling

For queries where the system outputs decision = "uncertain":
- Not counted as correct_accept
- Not counted as false_accept
- Counted in abstention_rate
- Reported separately in accepted_risk analysis

### Top-k partial credit

Top-k recall (Recall@k) counts a query as a hit if the true label appears
anywhere in the top-k candidates, regardless of the accept/reject decision.
This measures retrieval quality independent of decision policy.

### Similar-item ambiguity policy

Near-identical items (wallet_zipper vs wallet_trifold, receipt_salon vs
receipt_eye_doctor) are treated as distinct objects. A match of wallet_zipper
when the true label is wallet_trifold is a FP, not an ambiguous case.
The two wallet variants and two receipt variants are the primary hard cases.
They are measured separately in hard_cases analysis but are not excluded from
the main FP rate.

---

## Metrics Definitions

| Metric | Definition |
|---|---|
| top_1_accuracy | fraction of test queries where accepted label == true label |
| top_3_recall | fraction where true label is in top-3 candidates (pre-decision) |
| top_5_recall | fraction where true label is in top-5 candidates |
| top_10_recall | fraction where true label is in top-10 candidates |
| accepted_precision | correct_accepts / all_accepts |
| accepted_match_rate | all_accepts / total_queries (coverage) |
| correct_accept_rate | correct_accepts / total_queries |
| holdout_fp_rate | false_accepts / total_queries |
| abstention_rate | (uncertain + reject) / total_queries |
| accepted_risk | 1 - accepted_precision |
| safety_score | see full_benchmark.py composite formula |

---

## FP Budget Operating Points

Primary targets for calibration and operating-point selection:

- holdout_fp_rate <= 0.05 (strict)
- holdout_fp_rate <= 0.10 (production default)

For each budget, report: best threshold and margins, accepted_precision,
accepted_match_rate, correct_accept_rate, and latency p50/p95.

---

## Artifact Paths

All benchmark runs write to `BASELINE_ROOT`:

- Local default: `{repo_root}/benchmarks/baselines/accuracy_hardening/`
- Server default: `/opt/spaitra/accuracy_hardening_baselines/`

The frozen baseline (first full run after Phase 0 is complete) is saved under:
`{BASELINE_ROOT}/main/frozen_baseline/`

Do not modify or delete any baseline directory. Append only.
