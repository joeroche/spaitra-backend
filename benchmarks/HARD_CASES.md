# Hard Cases Workflow

The hard-cases set is a small review surface extracted from the frozen full
benchmark. It is for retrieval and decision-policy work, not more OCR threshold
tuning.

Files:

- `benchmarks/hard_cases/`: copied image files for selected cases.
- `benchmarks/hard_cases_manifest.json`: case metadata, source paths, and
  category counts.
- `benchmarks/hard_cases_dataset.csv`: the selected image subset for inspection
  and ad hoc experiments.
- `scripts/run_hard_cases.sh`: writes a mini-report from benchmark results.

Typical use after a benchmark run:

```bash
scripts/run_hard_cases.sh
```

This reads `benchmarks/results.json` and `benchmarks/inference_traces.csv`, then
writes:

- `benchmarks/hard_cases_report.md`
- `benchmarks/hard_cases_report.json`

To generate the set from a different archived run:

```bash
python scripts/build_hard_cases.py \
  --artifacts-dir /opt/spaitra/accuracy_hardening_baselines/main/frozen_baseline
```

The frozen trace does not include raw OCR text. Cases tagged
`text_signal_failure` mean OCR was allowed and text likelihood was high, but the
final match was still wrong. Treat those as text-integration or decision-policy
cases unless a later trace includes raw OCR text proving an OCR recognition issue.
