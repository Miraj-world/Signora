"""Evaluate grounded answers on held-out retrieval and abstention questions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from common import (DEFAULT_RESULTS_ROOT, EVAL_DIR, read_jsonl,
                    row_matches_relevance, write_json)


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from answer_generation import (DEFAULT_ANSWER_MODEL, evidence_prompt,
                               generate_answer, load_threshold)
from query_retrieval import retrieve
from retrieval_models import load_dotenv


QUALITY_FLOORS = {
    "status_accuracy": 0.90,
    "abstention_accuracy": 0.95,
    "citation_presence": 1.00,
    "citation_validity": 1.00,
    "citation_predicate_precision": 0.95,
    "citation_entailment": 0.90,
    "unsupported_claim_control": 0.90,
    "uncertainty_quality": 0.75,
    "recommendation_fact_separation": 0.75,
    "counterevidence_handling": 0.75,
}


class QualityJudgment(BaseModel):
    citation_entailment: int = Field(ge=0, le=2)
    unsupported_claim_control: int = Field(ge=0, le=2)
    uncertainty_quality: int = Field(ge=0, le=2)
    recommendation_fact_separation: int = Field(ge=0, le=2)
    counterevidence_handling: int = Field(ge=0, le=2)
    rationale: str


def scalar_filters(question: dict) -> dict:
    return {
        field: allowed[0]
        for field, allowed in (question.get("filters") or {}).items()
        if allowed
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def deterministic_scores(question: dict, answer: dict) -> dict:
    expected_status = "abstained" if question.get("should_abstain") else "answered"
    status_correct = answer["status"] == expected_status
    if answer["status"] != "answered":
        return {
            "status_correct": status_correct,
            "citation_presence": 1.0 if expected_status == "abstained" else 0.0,
            "citation_validity": 1.0 if expected_status == "abstained" else 0.0,
            "citation_predicate_precision": 1.0 if expected_status == "abstained" else 0.0,
            "source_diversity": 1.0 if expected_status == "abstained" else 0.0,
        }

    evidence_by_atom = {
        item["atom_id"]: item
        for item in answer["retrieval"].get("results", [])
    }
    cited_ids = [citation["atom_id"] for citation in answer.get("citations", [])]
    citation_validity = float(bool(cited_ids) and all(atom_id in evidence_by_atom for atom_id in cited_ids))
    relevant = [
        row_matches_relevance(evidence_by_atom[atom_id], question.get("relevance_rules"))
        for atom_id in cited_ids
        if atom_id in evidence_by_atom
    ]
    source_types = {
        evidence_by_atom[atom_id].get("source_type")
        for atom_id in cited_ids
        if atom_id in evidence_by_atom and evidence_by_atom[atom_id].get("source_type")
    }
    return {
        "status_correct": status_correct,
        "citation_presence": float(bool(cited_ids)),
        "citation_validity": citation_validity,
        "citation_predicate_precision": average([float(value) for value in relevant]),
        "source_diversity": min(1.0, len(source_types) / 2),
    }


def judge_answer(question: str, answer: dict, model: str) -> QualityJudgment:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for answer-quality evaluation.")

    from openai import OpenAI

    evidence = answer["retrieval"]["results"]
    candidate = {
        "answer": answer["answer"],
        "uncertainty": answer["uncertainty"],
        "recommendations": answer["recommendations"],
        "citations": answer["citations"],
    }
    instructions = (
        "You are evaluating a grounded Voice of Customer answer against the supplied retrieved evidence. "
        "Score each criterion from 0 to 2. Citation entailment: 2 when cited evidence directly supports every claim, "
        "1 for partial support, 0 for contradiction or missing support. Unsupported claim control: 2 when there are no "
        "unsupported claims, 1 for minor overreach, 0 for material fabrication. A relevant detail is not unsupported "
        "merely because the question did not explicitly request it; score only whether the supplied evidence supports it. "
        "Uncertainty quality: 2 when limitations "
        "are specific and calibrated, 1 when generic, 0 when absent or misleading. Recommendation/fact separation: "
        "2 when recommendations are clearly separate and evidence-linked or correctly omitted, 1 when somewhat mixed, "
        "0 when recommendations are presented as facts. Counterevidence handling: 2 when conflicts are represented or "
        "the answer appropriately avoids claiming consensus when no conflict is visible, 1 when incomplete, 0 when clear "
        "counterevidence is ignored. Judge only from the provided material."
    )
    prompt = evidence_prompt(question, evidence) + "\n\nCandidate answer:\n" + json.dumps(candidate, ensure_ascii=False)
    response = OpenAI().responses.parse(
        model=model,
        instructions=instructions,
        input=prompt,
        text_format=QualityJudgment,
    )
    if response.output_parsed is None:
        raise RuntimeError("Quality judge did not return structured output")
    return response.output_parsed


def aggregate(rows: list[dict]) -> dict:
    answerable = [row for row in rows if not row["should_abstain"]]
    abstention = [row for row in rows if row["should_abstain"]]
    judged = [row for row in answerable if row.get("judgment")]
    metrics = {
        "status_accuracy": average([float(row["deterministic"]["status_correct"]) for row in rows]),
        "answerable_accuracy": average([float(row["deterministic"]["status_correct"]) for row in answerable]),
        "abstention_accuracy": average([float(row["deterministic"]["status_correct"]) for row in abstention]),
        "citation_presence": average([row["deterministic"]["citation_presence"] for row in answerable]),
        "citation_validity": average([row["deterministic"]["citation_validity"] for row in answerable]),
        "citation_predicate_precision": average([row["deterministic"]["citation_predicate_precision"] for row in answerable]),
        "source_diversity": average([row["deterministic"]["source_diversity"] for row in answerable]),
    }
    for criterion in (
        "citation_entailment",
        "unsupported_claim_control",
        "uncertainty_quality",
        "recommendation_fact_separation",
        "counterevidence_handling",
    ):
        metrics[criterion] = average([row["judgment"][criterion] / 2 for row in judged])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citation-grounded answers on benchmark v2.")
    parser.add_argument("--answerable-limit", type=int, default=6)
    parser.add_argument("--abstention-limit", type=int, default=6)
    parser.add_argument("--answer-model", default=os.environ.get("SIGNORA_ANSWER_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--judge-model", default=os.environ.get("SIGNORA_JUDGE_MODEL", DEFAULT_ANSWER_MODEL))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_ROOT / "answer_quality_test.json"))
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    answerable = [
        question for question in read_jsonl(EVAL_DIR / "retrieval_questions_v2.jsonl")
        if question.get("split") == "test" and not question.get("should_abstain")
    ][:args.answerable_limit]
    abstention = [
        question for question in read_jsonl(EVAL_DIR / "abstention_questions_v2.jsonl")
        if question.get("split") == "test"
    ][:args.abstention_limit]

    rows = []
    for number, question in enumerate(answerable + abstention, 1):
        print(f"[{number}/{len(answerable) + len(abstention)}] {question['question_id']}")
        retrieval = retrieve(
            question["question"],
            top_k=args.top_k,
            mode="recall",
            filters=scalar_filters(question),
        )
        threshold = load_threshold(retrieval["embedding_profile"], "recall")
        retrieval["abstain_threshold"] = threshold
        retrieval["should_abstain"] = retrieval["top_score"] < threshold
        if retrieval["should_abstain"]:
            retrieval["results"] = []
        answer = generate_answer(question["question"], retrieval, model=args.answer_model)
        deterministic = deterministic_scores(question, answer)
        judgment = None
        if answer["status"] == "answered":
            judgment = judge_answer(question["question"], answer, args.judge_model).model_dump()
        rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "should_abstain": question.get("should_abstain", False),
            "answer": answer,
            "deterministic": deterministic,
            "judgment": judgment,
        })

    metrics = aggregate(rows)
    gate_checks = {name: metrics[name] >= floor for name, floor in QUALITY_FLOORS.items()}
    payload = {
        "evaluation_version": "answer_quality_v1",
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "answerable_count": len(answerable),
        "abstention_count": len(abstention),
        "quality_floors": QUALITY_FLOORS,
        "metrics": metrics,
        "gate_checks": gate_checks,
        "status": "pass" if all(gate_checks.values()) else "fail",
        "details": rows,
    }
    write_json(Path(args.output), payload)
    print(json.dumps({key: payload[key] for key in ("status", "metrics", "gate_checks")}, indent=2))
    print(f"Wrote {args.output}")
    if args.fail_on_gate and payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
