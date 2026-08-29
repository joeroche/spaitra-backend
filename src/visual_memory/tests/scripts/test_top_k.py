"""Unit tests for find_top_k and find_match_with_top_k."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from visual_memory.tests.scripts.test_harness import TestRunner
from visual_memory.utils.similarity_utils import find_top_k, find_match_with_top_k

_runner = TestRunner("top_k")


def _norm(seed: int, dim: int = 1536) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(1, dim), dim=1)


def test_find_top_k_known_good_match_at_rank_1():
    query = _norm(0)
    db = [("wallet", query.clone()), ("keys", _norm(1)), ("bottle", _norm(2))]
    result = find_top_k(query, db, k=3)
    assert result[0][0] == "wallet", f"expected wallet at rank 1, got {result[0][0]}"
    assert result[0][1] > 0.99, f"identical vector should have sim > 0.99, got {result[0][1]}"


def test_find_top_k_known_items_at_expected_ranks():
    query = _norm(0)
    # seed 0 is the query itself; pick seeds far from 0 for distinct distances
    db = [("a", _norm(10)), ("b", _norm(20)), ("c", _norm(30)), ("d", _norm(40))]
    result = find_top_k(query, db, k=4)
    labels = [r[0] for r in result]
    assert len(labels) == 4
    # similarities must be non-increasing
    sims = [r[1] for r in result]
    assert sims == sorted(sims, reverse=True), f"results not sorted: {sims}"


def test_find_top_k_empty_database_returns_empty():
    query = _norm(0)
    result = find_top_k(query, [], k=5)
    assert result == [], f"expected empty list, got {result}"


def test_find_top_k_k_larger_than_db_returns_full_db():
    query = _norm(0)
    db = [("a", _norm(1)), ("b", _norm(2))]
    result = find_top_k(query, db, k=10)
    assert len(result) == 2, f"expected 2 results, got {len(result)}"


def test_find_top_k_k_1_returns_single_best():
    query = _norm(0)
    db = [("wallet", query.clone()), ("keys", _norm(99))]
    result = find_top_k(query, db, k=1)
    assert len(result) == 1
    assert result[0][0] == "wallet"


def test_find_match_with_top_k_accept_returns_top_k():
    query = _norm(0)
    db = [("wallet", query.clone()), ("keys", _norm(1)), ("bottle", _norm(2))]
    matched, sim, margin, top_k = find_match_with_top_k(
        query, db, threshold_fn=lambda _: 0.5, k=3
    )
    assert matched == "wallet"
    assert sim > 0.99
    assert len(top_k) == 3
    assert top_k[0][0] == "wallet"


def test_find_match_with_top_k_reject_still_returns_top_k():
    query = _norm(0)
    db = [("wallet", _norm(1)), ("keys", _norm(2))]
    # threshold so high nothing matches
    matched, sim, margin, top_k = find_match_with_top_k(
        query, db, threshold_fn=lambda _: 0.9999, k=2
    )
    assert matched is None
    assert sim == 0.0
    assert len(top_k) == 2, f"top_k should still have 2 entries when rejected, got {len(top_k)}"


def test_find_match_with_top_k_top_k_length_bounded_by_k():
    query = _norm(0)
    db = [("a", _norm(i)) for i in range(1, 8)]
    _, _, _, top_k = find_match_with_top_k(query, db, threshold_fn=lambda _: 0.0, k=5)
    assert len(top_k) == 5, f"top_k should be bounded by k=5, got {len(top_k)}"


for name, fn in [
    ("top_k:known_good_match_at_rank_1", test_find_top_k_known_good_match_at_rank_1),
    ("top_k:known_items_at_expected_ranks", test_find_top_k_known_items_at_expected_ranks),
    ("top_k:empty_database", test_find_top_k_empty_database_returns_empty),
    ("top_k:k_larger_than_db", test_find_top_k_k_larger_than_db_returns_full_db),
    ("top_k:k_1_returns_best", test_find_top_k_k_1_returns_single_best),
    ("match_with_top_k:accept_returns_top_k", test_find_match_with_top_k_accept_returns_top_k),
    ("match_with_top_k:reject_still_returns_top_k", test_find_match_with_top_k_reject_still_returns_top_k),
    ("match_with_top_k:top_k_bounded_by_k", test_find_match_with_top_k_top_k_length_bounded_by_k),
]:
    _runner.run(name, fn)

sys.exit(_runner.summary())
