"""Similarity utilities for embedding search and box filtering."""

from typing import Optional, Tuple, List, Dict, Any, Callable
import torch
import torch.nn as nn
import re

_cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
_DOCUMENT_LABEL_PATTERN = re.compile(
    r"\b("
    r"receipt|document|invoice|bill|statement|form|contract|letter|memo|note|"
    r"paper|passport|license|id|card|menu|ticket|coupon|label|text"
    r")s?\b",
    flags=re.IGNORECASE,
)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity between two (1, dim) tensors."""
    return _cos_sim(a, b)

def find_match(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    Return best match above similarity threshold.
    Linear search over database.
    """

    if not database_embeddings:
        return None, 0.0

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    best_path = None
    best_similarity = -1.0

    for path, db_embedding in database_embeddings:
        sim = cosine_similarity(query_embedding, db_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_path = path

    if best_similarity < threshold:
        return None, 0.0

    return best_path, best_similarity.item()


def find_match_dynamic_threshold(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold_for_path: Callable[[str], float],
    margin_for_path: Optional[Callable[[str], float]] = None,
    score_aggregation_mode: str = "max",
) -> Tuple[Optional[str], float, float]:
    """Return best match where each candidate can have its own threshold and top1-top2 margin.

    Candidates with the same label are grouped; per-label scores are aggregated before
    labels are ranked against each other.

    score_aggregation_mode:
      "max"       -- highest exemplar similarity (default; matches prior implicit behavior)
      "top_2_avg" -- mean of top-2 exemplar similarities; falls back to max with < 2 exemplars

    Margin is the gap between best and second-best label aggregated scores.
    """
    if not database_embeddings:
        return None, 0.0, 0.0

    # Compute per-exemplar similarities grouped by label.
    by_label: Dict[str, Any] = {}
    for label, db_embedding in database_embeddings:
        threshold = float(threshold_for_path(label))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        margin = float(margin_for_path(label)) if margin_for_path is not None else 0.0
        if margin < 0.0:
            raise ValueError("Margin must be >= 0.")
        sim = float(cosine_similarity(query_embedding.detach(), db_embedding.detach()))
        if label not in by_label:
            by_label[label] = {"sims": [], "threshold": threshold, "margin": margin}
        by_label[label]["sims"].append(sim)

    def _aggregate(sims: List[float]) -> float:
        sims_sorted = sorted(sims, reverse=True)
        if score_aggregation_mode == "top_2_avg" and len(sims_sorted) >= 2:
            return (sims_sorted[0] + sims_sorted[1]) / 2.0
        return sims_sorted[0]

    label_scores: List[Tuple[float, str, float, float]] = [
        (_aggregate(data["sims"]), label, data["threshold"], data["margin"])
        for label, data in by_label.items()
    ]

    eligible = [row for row in label_scores if row[0] >= row[2]]
    if not eligible:
        return None, 0.0, 0.0

    eligible.sort(key=lambda row: row[0], reverse=True)
    best_agg_sim, best_label, _best_threshold, best_margin = eligible[0]
    second_agg_sim = max(
        (row[0] for row in label_scores if row[1] != best_label),
        default=0.0,
    )
    similarity_margin = best_agg_sim - second_agg_sim

    if similarity_margin < best_margin:
        return None, 0.0, similarity_margin

    return best_label, best_agg_sim, similarity_margin


def find_top_k(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k (label, similarity) pairs ranked by similarity descending.

    No thresholding or margin gating. Use this for ranked candidate retrieval
    before applying decision policy.
    """
    if not database_embeddings:
        return []
    ranked = [
        (label, float(cosine_similarity(query_embedding.detach(), db_emb.detach())))
        for label, db_emb in database_embeddings
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:k]


def find_match_with_top_k(
    query_embedding: torch.Tensor,
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold_fn: Callable[[str], float],
    margin_fn: Optional[Callable[[str], float]] = None,
    k: int = 5,
    score_aggregation_mode: str = "max",
) -> Tuple[Optional[str], float, float, List[Tuple[str, float]]]:
    """Return (matched_label_or_None, best_similarity, margin, top_k_list).

    top_k_list is [(label, sim)] for the top k candidates regardless of the
    accept/reject decision. The match result applies the same threshold and
    margin logic as find_match_dynamic_threshold.
    """
    top_k = find_top_k(query_embedding, database_embeddings, k=k)
    matched_label, best_sim, margin = find_match_dynamic_threshold(
        query_embedding, database_embeddings, threshold_fn, margin_fn,
        score_aggregation_mode=score_aggregation_mode,
    )
    return matched_label, best_sim, margin, top_k


def ocr_text_agreement(text_a: str, text_b: str, min_token_len: int = 2) -> float:
    """Return fraction of tokens in text_a that appear in text_b.

    Tokens are lowercased, whitespace-split, and filtered to min_token_len characters.
    Returns 0.0 when either input is empty or produces no valid tokens.
    Use this as an independent OCR signal in retrieval scoring (Step 1.5).
    """
    def _tokens(s: str):
        return {t.lower() for t in s.split() if len(t) >= min_token_len}

    tokens_a = _tokens(text_a or "")
    tokens_b = _tokens(text_b or "")
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def detect_alias_candidates(
    database_embeddings: List[Tuple[str, torch.Tensor]],
    threshold: float = 0.92,
) -> List[Tuple[str, str, float]]:
    """Find label pairs whose embeddings are suspiciously similar (alias detection).

    Returns a list of (label_a, label_b, max_sim) tuples for each pair of distinct
    labels where the maximum pairwise cosine similarity exceeds threshold.
    Detection only -- no merging, no side effects.

    For k labels each with n exemplars, compares all cross-label pairs.
    O(k^2 * n^2) -- suitable for small personal memory databases only.
    """
    by_label: dict = {}
    for label, emb in database_embeddings:
        by_label.setdefault(label, []).append(emb)

    labels = sorted(by_label.keys())
    candidates: List[Tuple[str, str, float]] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            max_sim = -1.0
            for emb_a in by_label[la]:
                for emb_b in by_label[lb]:
                    sim = float(cosine_similarity(
                        (emb_a.unsqueeze(0) if emb_a.dim() == 1 else emb_a).detach(),
                        (emb_b.unsqueeze(0) if emb_b.dim() == 1 else emb_b).detach(),
                    ))
                    if sim > max_sim:
                        max_sim = sim
            if max_sim >= threshold:
                candidates.append((la, lb, round(max_sim, 4)))

    return sorted(candidates, key=lambda x: x[2], reverse=True)


def is_document_like_label(label: str) -> bool:
    """Return True when label likely refers to a document/text-heavy object."""
    if not label:
        return False
    normalized = str(label).replace("_", " ").replace("-", " ")
    return _DOCUMENT_LABEL_PATTERN.search(normalized) is not None

def iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area
    return 0.0 if union == 0 else inter_area / union

def deduplicate_matches(
    matches: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Remove overlapping or duplicate detections.
    Keeps highest similarity match first.
    """

    if not matches:
        return []

    matches = sorted(matches, key=lambda m: m["similarity"], reverse=True)
    kept = []

    for match in matches:
        duplicate = False

        for k in kept:
            if iou(match["box"], k["box"]) > iou_threshold:
                if match["label"] == k["label"]:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(match)

    return kept
