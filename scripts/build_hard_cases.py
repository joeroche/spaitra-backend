#!/usr/bin/env python3
"""Build the hard-cases review set from a benchmark run."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


SAME_FAMILY_GROUPS = {
    "wallets": {"wallet_trifold", "wallet_zipper"},
    "bottles": {"magnesium_bottle", "water_bottle"},
    "keys": {"keys_house", "keys_safe"},
    "receipts": {"receipt_eye_doctor", "receipt_salon"},
    "eyewear": {"glasses_prescription", "sunglasses_sun"},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        filtered = (line for line in f if not line.lstrip().startswith("#"))
        return list(csv.DictReader(filtered))


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _family_for(label: str) -> str:
    for family, labels in SAME_FAMILY_GROUPS.items():
        if label in labels:
            return family
    return ""


def _case(
    category: str,
    row: dict[str, Any],
    result: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    rank = _as_int(row, "top_k_rank")
    return {
        "category": category,
        "image": row.get("image", ""),
        "true_label": row.get("label", result.get("label", "")),
        "predicted_label": row.get("personalized_match_label", ""),
        "similarity": round(_as_float(row, "personalized_similarity"), 6),
        "top_k_rank": rank if rank > 0 else None,
        "failure_class": row.get("failure_class", ""),
        "text_likelihood": result.get("text_likelihood"),
        "should_skip_ocr": result.get("should_skip_ocr"),
        "distance_bucket": result.get("distance_bucket"),
        "lighting_bucket": result.get("lighting_bucket"),
        "cleanliness_bucket": result.get("cleanliness_bucket"),
        "reason": reason,
    }


def _add_cases(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    results_by_image: dict[str, dict[str, Any]],
    category: str,
    reason: str,
    limit: int,
) -> None:
    seen = {(c["category"], c["image"]) for c in cases}
    for row in rows:
        image = row.get("image", "")
        key = (category, image)
        if key in seen:
            continue
        cases.append(_case(category, row, results_by_image.get(image, {}), reason))
        seen.add(key)
        if len([c for c in cases if c["category"] == category]) >= limit:
            break


def build_hard_cases(
    artifacts_dir: Path,
    dataset_csv: Path,
    images_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    hard_dataset_path: Path,
) -> dict[str, Any]:
    traces = _read_csv(artifacts_dir / "inference_traces.csv")
    dataset_rows = _read_csv(dataset_csv)
    results_data = json.loads((artifacts_dir / "results.json").read_text(encoding="utf-8"))
    results_by_image = {
        str(row.get("image")): row for row in results_data.get("results", [])
    }
    trace_by_image = {str(row.get("image")): row for row in traces}

    cases: list[dict[str, Any]] = []

    false_positives = sorted(
        [r for r in traces if r.get("failure_class") == "wrong_label_match"],
        key=lambda r: _as_float(r, "personalized_similarity"),
        reverse=True,
    )
    _add_cases(
        cases,
        false_positives,
        results_by_image,
        "top_false_positive",
        "Highest-similarity accepted wrong match from frozen baseline.",
        10,
    )

    near_misses = sorted(
        [r for r in traces if 2 <= _as_int(r, "top_k_rank") <= 5],
        key=lambda r: (_as_int(r, "top_k_rank", 99), -_as_float(r, "personalized_similarity")),
    )
    _add_cases(
        cases,
        near_misses,
        results_by_image,
        "near_miss_rank_2_to_5",
        "True label was close in top-k but not rank 1.",
        10,
    )

    text_signal_failures = []
    for image, result in results_by_image.items():
        trace = trace_by_image.get(image)
        if not trace:
            continue
        if int(result.get("personalized_correct", 0) or 0):
            continue
        if bool(result.get("should_skip_ocr")):
            continue
        if _as_float(result, "text_likelihood") < 0.30:
            continue
        text_signal_failures.append(trace)
    text_signal_failures.sort(
        key=lambda r: (
            -_as_float(results_by_image.get(str(r.get("image")), {}), "text_likelihood"),
            -_as_float(r, "personalized_similarity"),
        )
    )
    _add_cases(
        cases,
        text_signal_failures,
        results_by_image,
        "text_signal_failure",
        "OCR was allowed and text likelihood was high, but the match was still wrong.",
        10,
    )

    same_family_rows = []
    for row in traces:
        true_family = _family_for(str(row.get("label", "")))
        pred_family = _family_for(str(row.get("personalized_match_label", "")))
        if true_family and pred_family == true_family and row.get("label") != row.get("personalized_match_label"):
            same_family_rows.append(row)
    same_family_rows.sort(key=lambda r: _family_for(str(r.get("label", ""))))
    _add_cases(
        cases,
        same_family_rows,
        results_by_image,
        "same_family_hard_negative",
        "Known same-family object confusion that should improve without crushing recall.",
        10,
    )

    no_top_10 = [
        r for r in traces
        if _as_int(r, "top_k_rank") <= 0 or _as_int(r, "top_k_rank") > 10
    ]
    _add_cases(
        cases,
        no_top_10,
        results_by_image,
        "no_top_10_miss",
        "True label was absent from top 10.",
        5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_images: set[str] = set()
    for case in cases:
        image = str(case["image"])
        src = images_dir / image
        if src.exists():
            shutil.copy2(src, output_dir / image)
            copied_images.add(image)

    selected_images = {str(case["image"]) for case in cases}
    dataset_subset = [row for row in dataset_rows if row.get("image") in selected_images]
    with open(hard_dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "label", "ground_truth_distance_ft", "dino_prompt"],
        )
        f.write("# Hard-cases subset generated from frozen baseline traces.\n")
        writer.writeheader()
        writer.writerows(dataset_subset)

    metadata = results_data.get("metadata") or {}
    summary = {
        "source_artifacts_dir": str(artifacts_dir),
        "source_dataset_csv": str(dataset_csv),
        "images_dir": str(images_dir),
        "unique_images": len(selected_images),
        "copied_images": len(copied_images),
        "case_count": len(cases),
        "categories": {},
        "empty_categories": [],
        "alias_candidates": metadata.get("alias_candidates") or [],
        "notes": [
            "Raw OCR text was not present in frozen inference traces; text_signal_failure uses text_likelihood and OCR gating fields instead.",
            "OCR threshold tuning is closed; these cases are for retrieval and decision-policy work.",
        ],
    }
    for category in [
        "top_false_positive",
        "near_miss_rank_2_to_5",
        "text_signal_failure",
        "same_family_hard_negative",
        "no_top_10_miss",
    ]:
        count = sum(1 for c in cases if c["category"] == category)
        summary["categories"][category] = count
        if count == 0:
            summary["empty_categories"].append(category)

    manifest = {
        "version": 1,
        "summary": summary,
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Benchmark run directory containing results.json and inference_traces.csv.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/dataset.csv"))
    parser.add_argument("--images", type=Path, default=Path("benchmarks/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/hard_cases"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/hard_cases_manifest.json"),
    )
    parser.add_argument(
        "--hard-dataset",
        type=Path,
        default=Path("benchmarks/hard_cases_dataset.csv"),
    )
    args = parser.parse_args()

    manifest = build_hard_cases(
        artifacts_dir=args.artifacts_dir,
        dataset_csv=args.dataset,
        images_dir=args.images,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        hard_dataset_path=args.hard_dataset,
    )
    summary = manifest["summary"]
    print(f"Hard cases written: {args.manifest}")
    print(f"Unique images copied: {summary['copied_images']}/{summary['unique_images']}")
    for category, count in summary["categories"].items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
