# Benchmark Capture and Run Guide

---

## Objects (10 total)

| ID | What to shoot |
|----|---------------|
| wallet_zipper | Zipper-pattern wallet |
| wallet_trifold | Trifold wallet |
| sunglasses_sun | Sunglasses -- flat on surface, both lenses visible |
| glasses_prescription | Prescription glasses -- flat on surface |
| receipt_eye_doctor | Eye doctor receipt -- flat, no folds, full text visible |
| receipt_salon | Salon receipt -- flat, no folds, full text visible |
| magnesium_bottle | Magnesium supplement bottle -- label facing camera |
| water_bottle | Water bottle -- label facing camera |
| keys_house | House keys -- spread in loose fan on surface |
| keys_safe | Safe keys -- spread in loose fan on surface |

---

## Naming

```
{object_id}_{distance}ft_{lighting}_{background}.jpg
```

Examples:
```
wallet_zipper_1ft_bright_clean.jpg
sunglasses_sun_6ft_dim_messy.jpg
receipt_salon_3ft_bright_messy.jpg
```

All 120 images go in `benchmarks/images/`.

---

## Setup Before You Start

1. Mark three floor positions with tape: 1ft (~30cm), 3ft (~91cm), 6ft (~183cm)
2. Set up one clean surface (plain table or floor)
3. Set up one messy area nearby (desk with items, counter with clutter)
4. Have a lamp ready for dim sessions (desk lamp off to the side, NOT a flash)
5. Line up all 10 objects so you can grab each one quickly

---

## Shoot 4 Blocks -- Change Lighting Only Once

Group by lighting to minimize setup changes. Do all 60 bright shots first, then all 60 dim shots.

### Block 1: BRIGHT + CLEAN (30 shots, ~15 min)
Bright overhead lights. Plain surface. Move through all 10 objects at each distance.

| Object | 1ft | 3ft | 6ft |
|--------|-----|-----|-----|
| wallet_zipper | [ ] | [ ] | [ ] |
| wallet_trifold | [ ] | [ ] | [ ] |
| sunglasses_sun | [ ] | [ ] | [ ] |
| glasses_prescription | [ ] | [ ] | [ ] |
| receipt_eye_doctor | [ ] | [ ] | [ ] |
| receipt_salon | [ ] | [ ] | [ ] |
| magnesium_bottle | [ ] | [ ] | [ ] |
| water_bottle | [ ] | [ ] | [ ] |
| keys_house | [ ] | [ ] | [ ] |
| keys_safe | [ ] | [ ] | [ ] |

### Block 2: BRIGHT + MESSY (30 shots, ~15 min)
Same lighting. Move to messy surface. Same order.

| Object | 1ft | 3ft | 6ft |
|--------|-----|-----|-----|
| wallet_zipper | [ ] | [ ] | [ ] |
| wallet_trifold | [ ] | [ ] | [ ] |
| sunglasses_sun | [ ] | [ ] | [ ] |
| glasses_prescription | [ ] | [ ] | [ ] |
| receipt_eye_doctor | [ ] | [ ] | [ ] |
| receipt_salon | [ ] | [ ] | [ ] |
| magnesium_bottle | [ ] | [ ] | [ ] |
| water_bottle | [ ] | [ ] | [ ] |
| keys_house | [ ] | [ ] | [ ] |
| keys_safe | [ ] | [ ] | [ ] |

### Block 3: DIM + CLEAN (30 shots, ~15 min)
Close blinds, turn off overhead, single lamp at an angle. Back to clean surface.

| Object | 1ft | 3ft | 6ft |
|--------|-----|-----|-----|
| wallet_zipper | [ ] | [ ] | [ ] |
| wallet_trifold | [ ] | [ ] | [ ] |
| sunglasses_sun | [ ] | [ ] | [ ] |
| glasses_prescription | [ ] | [ ] | [ ] |
| receipt_eye_doctor | [ ] | [ ] | [ ] |
| receipt_salon | [ ] | [ ] | [ ] |
| magnesium_bottle | [ ] | [ ] | [ ] |
| water_bottle | [ ] | [ ] | [ ] |
| keys_house | [ ] | [ ] | [ ] |
| keys_safe | [ ] | [ ] | [ ] |

### Block 4: DIM + MESSY (30 shots, ~15 min)
Same dim lighting. Messy surface.

| Object | 1ft | 3ft | 6ft |
|--------|-----|-----|-----|
| wallet_zipper | [ ] | [ ] | [ ] |
| wallet_trifold | [ ] | [ ] | [ ] |
| sunglasses_sun | [ ] | [ ] | [ ] |
| glasses_prescription | [ ] | [ ] | [ ] |
| receipt_eye_doctor | [ ] | [ ] | [ ] |
| receipt_salon | [ ] | [ ] | [ ] |
| magnesium_bottle | [ ] | [ ] | [ ] |
| water_bottle | [ ] | [ ] | [ ] |
| keys_house | [ ] | [ ] | [ ] |
| keys_safe | [ ] | [ ] | [ ] |

---

## Camera Tips

- Rear camera, no zoom
- Tap object on screen to focus before each shot
- Portrait or landscape -- pick one, stay consistent
- No flash (defeats the dim condition)
- Same object orientation throughout all shots for that object

---

## Rename and Transfer

After shooting all 120:
1. Rename files on device to match naming convention (use a batch rename app or do it on computer)
2. Copy all to `benchmarks/images/`
3. Verify: `ls benchmarks/images/ | wc -l` should be 120

---

## Receipt Redaction

Run once per receipt before the benchmark.

```bash
python -m visual_memory.benchmarks.redact_receipt eye_doctor
python -m visual_memory.benchmarks.redact_receipt salon
```

At each prompt:
- **Redact>** Type one string per line to redact (your name, card last 4, phone number). Blank line to finish.
- **GT>** Type the receipt text after redaction (provider name, items, totals). Type `END` on its own line when done.

The script runs OCR on all 12 images for each receipt and blacks out matching regions.
At 6ft dim conditions, OCR may miss some regions -- script warns you which images were skipped.

Ground truth files saved to: `benchmarks/ground_truth/receipt_eye_doctor.txt` and `receipt_salon.txt`

---

## Validate Before Running

```bash
python -m visual_memory.benchmarks.check_dataset
```

Shows missing images grouped by object, and checks for receipt ground truth files.

---

## Running the Benchmark

### Fast smoke test (no depth, no OCR, ~5-10 min)
```bash
python -m visual_memory.benchmarks.full_benchmark \
    --dataset benchmarks/dataset.csv \
    --images benchmarks/images \
    --seed 42 --no-depth --no-ocr --epochs 5
```

### Full run (~2-4 hours)
```bash
python -m visual_memory.benchmarks.full_benchmark \
    --dataset benchmarks/dataset.csv \
    --images benchmarks/images \
    --seed 42
```

Or use the one-command runner which also archives artifacts:
```bash
bash scripts/run_benchmark.sh
```

### With calibrated focal length (better depth accuracy)
iPhone 15 Plus: f_px = (6.24 / 8.64) * imageWidthPx. At 4032px wide: f_px = 2912.
```bash
python -m visual_memory.benchmarks.full_benchmark \
    --dataset benchmarks/dataset.csv \
    --images benchmarks/images \
    --seed 42 --focal-length 2912
```

Negative images (from `tests/input_images/`) are evaluated automatically for false positive rate.
No extra action needed.

---

## Output Files

| File | Description |
|------|-------------|
| `benchmarks/results.csv` | One row per test image -- all metrics + latencies |
| `benchmarks/results.json` | Same data + metadata + negative FP results |
| `benchmarks/inference_traces.csv` | Per-query top-k candidates, component sims, failure class |
| `benchmarks/failure_taxonomy.json` | Failure category counts and rates |
| `benchmarks/projection_head_bench.pt` | Trained head weights from this run |

---

## Generate BENCHMARKS.md

```bash
python -m visual_memory.benchmarks.format_results
```

Writes `BENCHMARKS.md` at project root with:
- Per-label retrieval table (baseline vs personalized, delta)
- Per-label detection table
- Per-label depth accuracy table
- Per-condition breakdown (all 12 conditions)
- Latency table (mean/min/max per phase, outliers flagged)
- False positive table (negative images, baseline vs personalized)

---

## Reading the Results

Key things to check in BENCHMARKS.md:

**Accuracy delta** -- how much the projection head improved retrieval.
Positive = head helped. Near zero = not enough training data or classes too visually similar.

**Mean sim gap** -- per-image average cosine similarity improvement. Even +0.01 is meaningful.

**Triplet loss** -- convergence. >0.15 = underfitting (add more varied images).

**Detection rate by condition** -- expect drops at 6ft dim messy. >70% overall is healthy.

**Depth % error** -- without focal length: ~27%. With calibrated f_px: ~20-25%.

**FP delta** -- false positive change after training. Should be zero or negative (head should not increase FPs). If positive, head is overfitting.

**Latency outliers** -- OCR on receipts will be slowest. GDINO at 6ft may be slower.

**Top-k recall** -- top_3_recall and top_10_recall indicate retrieval headroom. A large gap
between top_1 and top_3 means the threshold/decision stage is the bottleneck, not the embedder.

**Failure taxonomy** -- see failure_taxonomy.json for category breakdown. threshold_miss and
near_miss cases are addressable by Branch A threshold tuning. retrieval_miss cases require
embedding or preprocessing fixes.
