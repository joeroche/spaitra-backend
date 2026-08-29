"""Centralized baseline artifact writer and comparator.

All benchmark runs write through this module so artifacts are consistent
across main, wt/global-hardening, and wt/verifier.

Usage from scripts/run_benchmark.sh:
    python -c "from visual_memory.benchmarks.baseline_writer import write_metrics_summary; write_metrics_summary('OUT_DIR')"

Usage for branch comparison:
    python -m visual_memory.benchmarks.baseline_writer compare \\
        {BASELINE_ROOT}/main/frozen_baseline \\
        {BASELINE_ROOT}/baseline-global-hardening/20260413_143022_a1b2c3d

Step 0.2 (extend benchmark metrics) must fill in write_metrics_summary() with
the new metric fields once full_benchmark.py produces them.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_BASELINE_ROOT = (
    _PROJECT_ROOT / "benchmarks" / "baselines" / "accuracy_hardening"
)

# Metrics whose improvement direction is "higher is better"
_HIGHER_IS_BETTER = {
    "personalized_accuracy",
    "accepted_precision",
    "accepted_match_rate",
    "correct_accept_rate",
    "top_1_accuracy",
    "top_3_recall",
    "top_5_recall",
    "top_10_recall",
    "safety_score",
}
# Metrics whose improvement direction is "lower is better"
_LOWER_IS_BETTER = {
    "holdout_fp_rate",
    "abstention_rate",
    "lat_retrieve_p50",
    "lat_retrieve_p95",
    "lat_embed_p50",
    "lat_embed_p95",
}
# Regression threshold: flag if delta exceeds this fraction of the baseline value
_REGRESSION_FLAG_DELTA = 0.02  # 2 percentage points absolute


def write_baseline(
    branch: str,
    run_id: str,
    config: Dict[str, Any],
    results_csv_path: Path,
    results_json_path: Path,
    baseline_root: Optional[Path] = None,
    metrics: Optional[Dict[str, Any]] = None,
    failure_taxonomy: Optional[Dict[str, Any]] = None,
    inference_traces_path: Optional[Path] = None,
    top_k: Optional[Dict[str, Any]] = None,
    confusion: Optional[Dict[str, Any]] = None,
    operating_points: Optional[List[Dict[str, Any]]] = None,
    fixed_fp_budget: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write all baseline artifacts for one benchmark run.

    Returns the run directory path.
    """
    root = Path(baseline_root) if baseline_root else _LOCAL_BASELINE_ROOT
    run_dir = root / branch / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.json -- includes Settings snapshot, git SHA, branch, worktree, command
    config_path = run_dir / "config.json"
    config["branch"] = branch
    config["run_id"] = run_id
    config["baseline_root"] = str(root)
    config["written_at"] = datetime.now(timezone.utc).isoformat()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Copy primary results
    if results_csv_path and Path(results_csv_path).exists():
        shutil.copy2(results_csv_path, run_dir / "results.csv")
    if results_json_path and Path(results_json_path).exists():
        shutil.copy2(results_json_path, run_dir / "results.json")

    # Copy inference traces
    if inference_traces_path and Path(inference_traces_path).exists():
        shutil.copy2(inference_traces_path, run_dir / "inference_traces.csv")

    # Write optional structured artifacts
    if metrics is not None:
        with open(run_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

    if failure_taxonomy is not None:
        with open(run_dir / "failure_taxonomy.json", "w") as f:
            json.dump(failure_taxonomy, f, indent=2, default=str)

    if top_k is not None:
        with open(run_dir / "top_k_analysis.json", "w") as f:
            json.dump(top_k, f, indent=2, default=str)

    if confusion is not None:
        with open(run_dir / "confusion_pairs.json", "w") as f:
            json.dump(confusion, f, indent=2, default=str)

    if operating_points is not None:
        import csv as _csv
        if operating_points:
            op_path = run_dir / "operating_points.csv"
            fieldnames = list(operating_points[0].keys())
            with open(op_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(operating_points)

    if fixed_fp_budget is not None:
        with open(run_dir / "fixed_fp_budget.json", "w") as f:
            json.dump(fixed_fp_budget, f, indent=2, default=str)

    # Create/update branch/latest symlink
    latest_link = root / branch / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_id)

    return run_dir


def write_metrics_summary(run_dir: str | Path) -> Dict[str, Any]:
    """Parse results.json in run_dir and write metrics_summary.json.

    Called by scripts/run_benchmark.sh after the benchmark completes.
    Reads hardening_metrics from metadata (added in Phase 0.2) and computes
    accuracy from results rows.
    """
    run_dir = Path(run_dir)
    results_json = run_dir / "results.json"
    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(results_json) as f:
        data = json.load(f)

    metrics: Dict[str, Any] = {}
    metadata = data.get("metadata") or {}

    # Basic accuracy from results rows
    rows = data.get("results") or []
    n = max(len(rows), 1)
    if rows:
        metrics["personalized_accuracy"] = round(
            sum(r.get("personalized_correct", 0) for r in rows) / n, 4
        )
        metrics["baseline_accuracy"] = round(
            sum(r.get("baseline_correct", 0) for r in rows) / n, 4
        )
        metrics["top_1_accuracy"] = metrics["personalized_accuracy"]

    # Phase 0.2 hardening metrics from metadata block
    hardening = metadata.get("hardening_metrics") or {}
    for key in (
        "top_3_recall",
        "top_5_recall",
        "top_10_recall",
        "accepted_match_rate",
        "accepted_precision",
        "correct_accept_rate",
        "abstention_rate",
        "safety_score",
        "latency_p50_retrieve_pe",
        "latency_p95_retrieve_pe",
    ):
        if key in hardening:
            metrics[key] = hardening[key]
    # Normalize latency key names to match _LOWER_IS_BETTER set
    if "latency_p50_retrieve_pe" in metrics:
        metrics["lat_retrieve_p50"] = metrics.pop("latency_p50_retrieve_pe")
    if "latency_p95_retrieve_pe" in metrics:
        metrics["lat_retrieve_p95"] = metrics.pop("latency_p95_retrieve_pe")

    # Holdout FP rate from fp_holdout section
    fp_holdout = data.get("fp_holdout") or []
    if fp_holdout:
        n_hold = max(len(fp_holdout), 1)
        metrics["holdout_fp_rate"] = round(
            sum(int(r.get("personalized_fp", 0)) for r in fp_holdout) / n_hold, 4
        )

    # Per-label/distance/lighting breakdowns from hardening_metrics
    for key in ("per_label_accuracy", "per_distance_accuracy", "per_lighting_accuracy"):
        if key in hardening:
            metrics[key] = hardening[key]
        elif key in data:
            metrics[key] = data[key]

    out_path = run_dir / "metrics_summary.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


def compare_baselines(
    baseline_a: str | Path,
    baseline_b: str | Path,
) -> Dict[str, Any]:
    """Compare two baseline run directories. Return a regression report.

    Loads metrics_summary.json from each run. If it is absent, falls back
    to results.json summary block.

    Returns a dict with:
      - run_a, run_b: absolute paths
      - regressions: list of {metric, value_a, value_b, delta, direction}
        for metrics that regressed beyond _REGRESSION_FLAG_DELTA
      - improvements: same structure for metrics that improved
      - neutral: metrics within tolerance
      - missing: metrics present in one run but not the other
    """
    a_dir = Path(baseline_a)
    b_dir = Path(baseline_b)

    def _load_metrics(d: Path) -> Dict[str, float]:
        summary = d / "metrics_summary.json"
        if summary.exists():
            with open(summary) as f:
                raw = json.load(f)
        else:
            results = d / "results.json"
            if not results.exists():
                raise FileNotFoundError(f"No metrics_summary.json or results.json in {d}")
            with open(results) as f:
                data = json.load(f)
            raw = data.get("summary") or data.get("metrics") or {}
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    metrics_a = _load_metrics(a_dir)
    metrics_b = _load_metrics(b_dir)

    all_keys = set(metrics_a) | set(metrics_b)
    regressions: List[Dict] = []
    improvements: List[Dict] = []
    neutral: List[Dict] = []
    missing: List[str] = []

    for key in sorted(all_keys):
        if key not in metrics_a or key not in metrics_b:
            missing.append(key)
            continue

        a_val = metrics_a[key]
        b_val = metrics_b[key]
        delta = b_val - a_val

        entry = {"metric": key, "value_a": a_val, "value_b": b_val, "delta": delta}

        if abs(delta) < _REGRESSION_FLAG_DELTA:
            neutral.append(entry)
        elif key in _HIGHER_IS_BETTER:
            if delta < 0:
                regressions.append(entry)
            else:
                improvements.append(entry)
        elif key in _LOWER_IS_BETTER:
            if delta > 0:
                regressions.append(entry)
            else:
                improvements.append(entry)
        else:
            neutral.append(entry)  # direction unknown, treat as neutral

    report = {
        "run_a": str(a_dir.resolve()),
        "run_b": str(b_dir.resolve()),
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "regressions": regressions,
        "improvements": improvements,
        "neutral": neutral,
        "missing": missing,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
    }
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print(f"Comparing:")
    print(f"  A: {report['run_a']}")
    print(f"  B: {report['run_b']}")
    print()

    if report["regressions"]:
        print(f"REGRESSIONS ({len(report['regressions'])}):")
        for r in report["regressions"]:
            print(f"  {r['metric']}: {r['value_a']:.4f} -> {r['value_b']:.4f}  ({r['delta']:+.4f})")
    else:
        print("No regressions.")

    if report["improvements"]:
        print(f"\nImprovements ({len(report['improvements'])}):")
        for r in report["improvements"]:
            print(f"  {r['metric']}: {r['value_a']:.4f} -> {r['value_b']:.4f}  ({r['delta']:+.4f})")

    if report["neutral"]:
        print(f"\nNeutral ({len(report['neutral'])}):")
        for r in report["neutral"]:
            print(f"  {r['metric']}: {r['value_a']:.4f} -> {r['value_b']:.4f}  ({r['delta']:+.4f})")

    if report["missing"]:
        print(f"\nMissing in one run: {report['missing']}")


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1] != "compare":
        print("Usage: python -m visual_memory.benchmarks.baseline_writer compare <run_a_dir> <run_b_dir>")
        sys.exit(1)
    report = compare_baselines(sys.argv[2], sys.argv[3])
    _print_report(report)
    out = Path(sys.argv[2]).parent / "comparison_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out}")
    sys.exit(1 if report["regression_count"] > 0 else 0)


if __name__ == "__main__":
    main()
