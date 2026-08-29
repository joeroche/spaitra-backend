"""Audit the quality of items in the visual memory database.

Produces benchmarks/memory_audit.json with per-label quality metrics:
  exemplar_count, has_ocr, avg_blur, avg_luminance, has_alias_warning,
  reference_crop_present.

Blur and luminance require image files to be present at image_path.
If files are missing, those fields are null.

Usage:
    python -m visual_memory.benchmarks.check_memory_quality
    python -m visual_memory.benchmarks.check_memory_quality --db data/memory.db
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARKS_DIR = _PROJECT_ROOT / "benchmarks"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit visual memory database quality")
    p.add_argument("--db", type=Path, default=None, help="Path to memory.db (default: data/memory.db)")
    p.add_argument("--out", type=Path, default=_BENCHMARKS_DIR / "memory_audit.json",
                   help="Output path for audit JSON")
    p.add_argument("--alias-threshold", type=float, default=0.92,
                   help="Cosine similarity threshold for alias detection")
    return p.parse_args()


def _blur_and_luminance(image_path: str):
    """Return (blur_score, luminance) for image at path, or (None, None) if unavailable."""
    if not image_path:
        return None, None
    p = Path(image_path)
    if not p.exists():
        return None, None
    try:
        from visual_memory.utils.quality_utils import blur_score, mean_luminance
        from visual_memory.utils.image_utils import load_image
        img = load_image(str(p))
        return blur_score(img), mean_luminance(img)
    except Exception:
        return None, None


def _audit(db_path: Optional[Path], alias_threshold: float) -> dict:
    from visual_memory.config.settings import Settings
    from visual_memory.database.store import MemoryStore
    from visual_memory.utils.similarity_utils import detect_alias_candidates

    settings = Settings()
    resolved_db = db_path or Path(settings.db_path)
    if not resolved_db.is_absolute():
        resolved_db = _PROJECT_ROOT / resolved_db

    store = MemoryStore(str(resolved_db))
    all_items = store.get_all_items()
    meta_items = store.get_items_metadata()

    meta_by_id = {m["id"]: m for m in meta_items}

    by_label: dict = defaultdict(list)
    for item in all_items:
        item_meta = meta_by_id.get(item["id"], {})
        item["image_path"] = item_meta.get("image_path", "")
        by_label[item["label"]].append(item)

    # Build embedding list for alias detection
    db_embeddings = [(item["label"], item["combined_embedding"]) for item in all_items]
    alias_pairs = detect_alias_candidates(db_embeddings, threshold=alias_threshold)
    alias_labels = {la for la, lb, _ in alias_pairs} | {lb for la, lb, _ in alias_pairs}

    per_label = {}
    for label, items in sorted(by_label.items()):
        blurs = []
        lums = []
        has_ocr = False
        reference_crop_present = False

        for item in items:
            if item.get("ocr_text", "").strip():
                has_ocr = True
            b, lum = _blur_and_luminance(item.get("image_path", ""))
            if b is not None:
                blurs.append(b)
            if lum is not None:
                lums.append(lum)
            if item.get("image_path", "") and Path(item["image_path"]).exists():
                reference_crop_present = True

        per_label[label] = {
            "exemplar_count": len(items),
            "has_ocr": has_ocr,
            "avg_blur": round(sum(blurs) / len(blurs), 2) if blurs else None,
            "avg_luminance": round(sum(lums) / len(lums), 2) if lums else None,
            "reference_crop_present": reference_crop_present,
            "has_alias_warning": label in alias_labels,
        }

    alias_list = [
        {"label_a": la, "label_b": lb, "max_sim": round(sim, 4)}
        for la, lb, sim in alias_pairs
    ]

    return {
        "db_path": str(resolved_db),
        "total_labels": len(by_label),
        "total_exemplars": len(all_items),
        "alias_threshold": alias_threshold,
        "alias_candidates": alias_list,
        "per_label": per_label,
    }


def main() -> None:
    args = _parse_args()
    result = _audit(args.db, args.alias_threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Memory audit written to: {args.out}")
    print(f"  Labels: {result['total_labels']}")
    print(f"  Exemplars: {result['total_exemplars']}")
    if result["alias_candidates"]:
        print(f"  Alias candidates: {len(result['alias_candidates'])} pairs")
        for a in result["alias_candidates"]:
            print(f"    {a['label_a']} <-> {a['label_b']}: {a['max_sim']:.3f}")
    else:
        print("  No alias candidates detected.")
    for label, metrics in result["per_label"].items():
        flags = []
        if not metrics["has_ocr"]:
            flags.append("no_ocr")
        if metrics["exemplar_count"] < 3:
            flags.append(f"few_exemplars({metrics['exemplar_count']})")
        if metrics["has_alias_warning"]:
            flags.append("alias_warning")
        blur_str = f"{metrics['avg_blur']:.1f}" if metrics["avg_blur"] is not None else "n/a"
        lum_str = f"{metrics['avg_luminance']:.1f}" if metrics["avg_luminance"] is not None else "n/a"
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        print(f"  {label}: {metrics['exemplar_count']} exemplars, blur={blur_str}, lum={lum_str}{flag_str}")


if __name__ == "__main__":
    main()
