"""Focused checks for recall-oriented candidate broadening and reranking."""
from pathlib import Path
import sys

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from retrieval_strategy import rank_rows


def rows():
    return [
        {
            "position": 0,
            "atom_id": "semantic-only",
            "search_text": "fast exports and report downloads",
            "topic": "reporting",
            "product_area": "analytics",
            "customer_segment": "small_business",
        },
        {
            "position": 1,
            "atom_id": "topic-match",
            "search_text": "enterprise administrators struggle during workspace setup",
            "topic": "onboarding",
            "product_area": "admin",
            "customer_segment": "enterprise",
        },
    ]


def test_recall_mode_broadens_beyond_semantic_shortlist():
    semantic_scores = np.array([0.90, 0.62], dtype="float32")
    ranked = rank_rows(
        "enterprise onboarding problems",
        rows(),
        semantic_scores,
        mode="recall",
        candidate_pool=1,
    )
    assert ranked[0].row["atom_id"] == "topic-match"
    assert {item.row["atom_id"] for item in ranked} == {"semantic-only", "topic-match"}


def test_pipeline_mode_preserves_hybrid_baseline_order():
    semantic_scores = np.array([0.90, 0.62], dtype="float32")
    ranked = rank_rows(
        "enterprise onboarding problems",
        rows(),
        semantic_scores,
        mode="pipeline",
    )
    assert ranked[0].row["atom_id"] == "semantic-only"


def test_candidate_pool_must_be_positive():
    try:
        rank_rows("onboarding", rows(), np.array([0.9, 0.6]), mode="recall", candidate_pool=0)
    except ValueError as error:
        assert "candidate_pool" in str(error)
    else:
        raise AssertionError("Expected candidate_pool validation to fail")


if __name__ == "__main__":
    test_recall_mode_broadens_beyond_semantic_shortlist()
    test_pipeline_mode_preserves_hybrid_baseline_order()
    test_candidate_pool_must_be_positive()
    print("retrieval strategy checks passed")
