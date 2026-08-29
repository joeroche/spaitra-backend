"""MatchCandidate dataclass for the retrieval stage of ScanPipeline.

Produced by the RETRIEVE stage and consumed by VERIFY (no-op now, verifier in Branch B)
and DECIDE (threshold + margin gating). Using a typed container here makes each stage
boundary explicit and verifier integration (Branch B) a clean drop-in.

Public API of ScanPipeline.run() does not change -- MatchCandidate is an internal type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MatchCandidate:
    """One retrieval candidate for a single detected crop.

    Attributes:
        label:       The matched label from the database (top-1 match above threshold).
                     None when no candidate passes the threshold.
        similarity:  Cosine similarity of the projected query to the best DB entry.
        margin:      Gap between top-1 and top-2 similarity (top-1 minus best competitor).
        rank:        1-indexed position of the true database entry in the ranked list.
                     0 when the item was not retrieved (rank > k or below threshold).
                     Only meaningful in benchmark context where true label is known.
        top_k:       [(label, similarity), ...] for the top-k candidates before
                     threshold gating, sorted by similarity descending.
        is_document: True when the query's OCR text or label hints at a text-heavy item.
                     Determines which threshold and margin settings apply.
        db_path:     Raw path key from database_embeddings used in ScanPipeline
                     (label for benchmark, path string for production).
    """

    label: Optional[str]
    similarity: float
    margin: float
    rank: int = 0
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    is_document: bool = False
    db_path: Optional[str] = None

    def accepted(self) -> bool:
        """True when a match was returned (label is not None)."""
        return self.label is not None
