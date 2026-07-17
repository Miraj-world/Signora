"""Unit checks for deterministic answer-quality evaluation."""
from pathlib import Path
import sys


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluate_answers import aggregate, deterministic_scores, scalar_filters


def test_scalar_filters_convert_benchmark_lists():
    assert scalar_filters({"filters": {"severity": ["high"]}}) == {"severity": "high"}


def test_citations_are_scored_against_predicate_not_only_canonical_ids():
    question = {
        "should_abstain": False,
        "relevance_rules": {"topic_any": ["offline_editing"], "severity_any": ["high"]},
    }
    answer = {
        "status": "answered",
        "citations": [{"atom_id": "atom_alternative"}],
        "retrieval": {"results": [{
            "atom_id": "atom_alternative",
            "topic": "offline_editing",
            "severity": "high",
            "source_type": "support_ticket",
        }]},
    }
    scores = deterministic_scores(question, answer)
    assert scores["citation_validity"] == 1.0
    assert scores["citation_predicate_precision"] == 1.0


def test_aggregate_keeps_abstention_separate():
    rows = [
        {
            "should_abstain": False,
            "deterministic": {
                "status_correct": True,
                "citation_presence": 1.0,
                "citation_validity": 1.0,
                "citation_predicate_precision": 1.0,
                "source_diversity": 0.5,
            },
            "judgment": {
                "citation_entailment": 2,
                "unsupported_claim_control": 2,
                "uncertainty_quality": 2,
                "recommendation_fact_separation": 2,
                "counterevidence_handling": 2,
            },
        },
        {
            "should_abstain": True,
            "deterministic": {
                "status_correct": True,
                "citation_presence": 1.0,
                "citation_validity": 1.0,
                "citation_predicate_precision": 1.0,
                "source_diversity": 1.0,
            },
            "judgment": None,
        },
    ]
    metrics = aggregate(rows)
    assert metrics["status_accuracy"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["citation_entailment"] == 1.0


if __name__ == "__main__":
    test_scalar_filters_convert_benchmark_lists()
    test_citations_are_scored_against_predicate_not_only_canonical_ids()
    test_aggregate_keeps_abstention_separate()
    print("answer quality evaluation checks passed")
