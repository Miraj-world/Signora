"""Grounded answer generation with deterministic citation validation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from retrieval_models import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "config" / "retrieval_policy.json"
DEFAULT_ANSWER_MODEL = "gpt-5.4-mini"
ABSTENTION_MESSAGE = "I do not have enough retrieved evidence to answer that reliably."


class GroundedDraft(BaseModel):
    status: Literal["answered", "abstained"]
    answer: str
    cited_atom_ids: list[str] = Field(default_factory=list)
    uncertainty: str
    recommendations: list[str] = Field(default_factory=list)


class CitationValidationError(ValueError):
    pass


def load_threshold(profile: str, mode: str, policy_path: Path = DEFAULT_POLICY_PATH) -> float:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    try:
        return float(policy["thresholds"][profile][mode])
    except KeyError as error:
        raise ValueError(
            f"No calibrated abstention threshold for profile={profile} mode={mode}. "
            "Pass --abstain-threshold or calibrate this retrieval configuration."
        ) from error


def evidence_prompt(query: str, evidence: list[dict]) -> str:
    blocks = []
    for item in evidence:
        context = (item.get("source_context") or "")[:1200]
        blocks.append(
            "\n".join([
                f"atom_id: {item['atom_id']}",
                f"feedback_id: {item.get('feedback_id')}",
                f"statement: {item.get('statement')}",
                f"context: {context}",
                f"metadata: product_area={item.get('product_area')}; topic={item.get('topic')}; "
                f"segment={item.get('customer_segment')}; severity={item.get('severity')}; "
                f"source_type={item.get('source_type')}",
            ])
        )
    return f"Question:\n{query}\n\nRetrieved evidence:\n\n" + "\n\n---\n\n".join(blocks)


def finalize_draft(draft: GroundedDraft, evidence: list[dict]) -> dict:
    if draft.status == "abstained":
        return {
            "status": "abstained",
            "answer": draft.answer or ABSTENTION_MESSAGE,
            "uncertainty": draft.uncertainty,
            "recommendations": [],
            "citations": [],
        }

    evidence_by_atom = {item["atom_id"]: item for item in evidence}
    cited_ids = list(dict.fromkeys(draft.cited_atom_ids))
    invalid = [atom_id for atom_id in cited_ids if atom_id not in evidence_by_atom]
    if invalid:
        raise CitationValidationError(f"Model cited atoms outside retrieved evidence: {invalid}")
    if not cited_ids:
        raise CitationValidationError("Answered response contained no citations")
    rendered_text = "\n".join([draft.answer, *draft.recommendations])
    inline_ids = set(re.findall(r"\[(atom_[A-Za-z0-9_-]+)\]", rendered_text))
    unreported = inline_ids - set(cited_ids)
    missing_inline = set(cited_ids) - inline_ids
    if unreported:
        raise CitationValidationError(f"Inline citations were omitted from cited_atom_ids: {sorted(unreported)}")
    if missing_inline:
        raise CitationValidationError(f"Cited atoms were not referenced inline: {sorted(missing_inline)}")

    citations = []
    for atom_id in cited_ids:
        item = evidence_by_atom[atom_id]
        citations.append({
            "atom_id": atom_id,
            "feedback_id": item.get("feedback_id"),
            "source_url": item.get("source_url"),
            "statement": item.get("statement"),
        })
    return {
        "status": "answered",
        "answer": draft.answer,
        "uncertainty": draft.uncertainty,
        "recommendations": draft.recommendations,
        "citations": citations,
    }


def abstained_answer(reason: str, retrieval: dict, model: str | None = None) -> dict:
    return {
        "status": "abstained",
        "answer": ABSTENTION_MESSAGE,
        "uncertainty": reason,
        "recommendations": [],
        "citations": [],
        "generation_model": model,
        "retrieval": retrieval,
    }


def generate_answer(query: str, retrieval: dict, model: str | None = None) -> dict:
    model = model or os.environ.get("SIGNORA_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL
    evidence = retrieval.get("results") or []
    if retrieval.get("should_abstain") or not evidence:
        return abstained_answer("Retrieval confidence was below the calibrated threshold.", retrieval, model)

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to the local .env file.")

    from openai import OpenAI

    instructions = (
        "You are Signora's grounded Voice of Customer analyst. Answer only from the retrieved evidence. "
        "Never introduce facts, counts, causes, or customer claims that are not present in that evidence. "
        "Every factual sentence in the answer must end with one or more exact atom citations like "
        "[atom_123]. Use only atom IDs shown in the evidence and return every used ID in cited_atom_ids. "
        "Clearly separate evidence from recommendations. If the evidence conflicts, say so. "
        "If it is insufficient to answer the question, return status=abstained. Keep the answer concise."
    )
    client = OpenAI()
    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=evidence_prompt(query, evidence),
        text_format=GroundedDraft,
    )
    draft = response.output_parsed
    if draft is None:
        return abstained_answer("The generation model did not return a valid structured answer.", retrieval, model)
    try:
        answer = finalize_draft(draft, evidence)
    except CitationValidationError as error:
        return abstained_answer(f"Citation validation failed: {error}", retrieval, model)
    answer["generation_model"] = model
    answer["retrieval"] = retrieval
    return answer
