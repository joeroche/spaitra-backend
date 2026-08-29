#!/usr/bin/env python3
"""Generate a mini-report for hard cases from a benchmark results file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_traces(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image"]: row for row in csv.DictReader(f)}


def _result_by_image(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("image")): row for row in data.get("results", [])}


def _score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def build_report(
    manifest_path: Path,
    results_json: Path,
    traces_csv: Path,
    output_md: Path,
    output_json: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = _result_by_image(results_json)
    traces = _read_traces(traces_csv)

    rows = []
    category_totals: dict[str, dict[str, int]] = {}
    for case in manifest.get("cases", []):
        image = str(case.get("image", ""))
        result = results.get(image, {})
        trace = traces.get(image, {})
        category = str(case.get("category", "unknown"))
        correct = int(result.get("personalized_correct", 0) or 0)
        matched = trace.get("personalized_match_label") or result.get("holdout_personalized_match") or ""
        similarity = trace.get("personalized_similarity") or result.get("personalized_similarity")
        row = {
            "category": category,
            "image": image,
            "true_label": case.get("true_label", result.get("label", "")),
            "expected_problem_label": case.get("predicted_label", ""),
            "current_match_label": matched,
            "current_similarity": similarity,
            "current_correct": bool(correct),
            "failure_class": trace.get("failure_class", case.get("failure_class", "")),
            "reason": case.get("reason", ""),
        }
        rows.append(row)
        total = category_totals.setdefault(category, {"total": 0, "passed": 0})
        total["total"] += 1
        total["passed"] += correct

    summary = {
        "manifest": str(manifest_path),
        "results_json": str(results_json),
        "traces_csv": str(traces_csv) if traces_csv.exists() else "",
        "case_count": len(rows),
        "category_totals": category_totals,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Hard Cases Mini Report",
        "",
        f"- Results: `{results_json}`",
        f"- Cases: {len(rows)}",
        "",
        "## Category Pass Rates",
        "",
    ]
    for category, totals in sorted(category_totals.items()):
        total = max(totals["total"], 1)
        rate = totals["passed"] / total
        lines.append(f"- {category}: {totals['passed']}/{totals['total']} ({rate:.1%})")
    lines.extend(["", "## Cases", ""])
    lines.append("| Category | Image | True | Expected Problem | Current Match | Sim | Correct |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {category} | {image} | {true_label} | {expected_problem_label} | "
            "{current_match_label} | {sim} | {correct} |".format(
                category=row["category"],
                image=row["image"],
                true_label=row["true_label"],
                expected_problem_label=row["expected_problem_label"],
                current_match_label=row["current_match_label"],
                sim=_score(row["current_similarity"]),
                correct="yes" if row["current_correct"] else "no",
            )
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": summary, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/hard_cases_manifest.json"),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("benchmarks/results.json"),
    )
    parser.add_argument(
        "--traces-csv",
        type=Path,
        default=Path("benchmarks/inference_traces.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmarks/hard_cases_report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/hard_cases_report.json"),
    )
    args = parser.parse_args()
    report = build_report(
        manifest_path=args.manifest,
        results_json=args.results_json,
        traces_csv=args.traces_csv,
        output_md=args.output_md,
        output_json=args.output_json,
    )
    print(f"Hard cases report written: {args.output_md}")
    for category, totals in sorted(report["summary"]["category_totals"].items()):
        total = max(totals["total"], 1)
        print(f"  {category}: {totals['passed']}/{totals['total']} ({totals['passed'] / total:.1%})")


if __name__ == "__main__":
    main()
