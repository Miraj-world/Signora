"""Checks for calibrated abstention and fail-closed citation handling."""
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from answer_generation import (CitationValidationError, GroundedClaim, GroundedDraft,
                               finalize_draft, generate_answer, load_threshold)


EVIDENCE = [{
    "atom_id": "atom_valid",
    "feedback_id": "fb_valid",
    "source_url": None,
    "statement": "Enterprise administrators struggle with SSO setup.",
}]


def test_valid_citations_are_enriched_from_retrieved_evidence():
    draft = GroundedDraft(
        status="answered",
        claims=[GroundedClaim(
            text="Administrators struggle with SSO setup.",
            cited_atom_ids=["atom_valid"],
        )],
        uncertainty="Evidence is limited to the retrieved feedback.",
    )
    answer = finalize_draft(draft, EVIDENCE)
    assert answer["status"] == "answered"
    assert answer["citations"][0]["feedback_id"] == "fb_valid"


def test_invented_citation_fails_closed():
    draft = GroundedDraft(
        status="answered",
        claims=[GroundedClaim(
            text="Unsupported claim.",
            cited_atom_ids=["atom_invented"],
        )],
        uncertainty="",
    )
    try:
        finalize_draft(draft, EVIDENCE)
    except CitationValidationError as error:
        assert "outside retrieved evidence" in str(error)
    else:
        raise AssertionError("Expected invented citation to fail validation")


def test_inline_citations_are_rendered_from_structured_claims():
    draft = GroundedDraft(
        status="answered",
        claims=[GroundedClaim(
            text="Supported claim.",
            cited_atom_ids=["atom_valid"],
        )],
        uncertainty="",
        recommendations=[GroundedClaim(
            text="Improve SSO setup.",
            cited_atom_ids=["atom_valid"],
        )],
    )
    answer = finalize_draft(draft, EVIDENCE)
    assert answer["answer"].endswith("[atom_valid]")
    assert answer["recommendations"][0].endswith("[atom_valid]")


def test_low_confidence_retrieval_abstains_without_generation_call():
    retrieval = {"should_abstain": True, "results": [], "top_score": 0.1}
    answer = generate_answer("Unknown topic?", retrieval)
    assert answer["status"] == "abstained"
    assert answer["citations"] == []


def test_recall_threshold_is_versioned():
    assert load_threshold("openai_small", "recall") == 0.3453946398364173


if __name__ == "__main__":
    test_valid_citations_are_enriched_from_retrieved_evidence()
    test_invented_citation_fails_closed()
    test_inline_citations_are_rendered_from_structured_claims()
    test_low_confidence_retrieval_abstains_without_generation_call()
    test_recall_threshold_is_versioned()
    print("answer generation checks passed")
