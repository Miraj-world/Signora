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


class GroundedClaim(BaseModel):
    text: str
    cited_atom_ids: list[str] = Field(min_length=1)


class GroundedDraft(BaseModel):
    status: Literal["answered", "abstained"]
    claims: list[GroundedClaim] = Field(default_factory=list)
    caveats: list[GroundedClaim] = Field(default_factory=list)
    uncertainty: str
    recommendations: list[GroundedClaim] = Field(default_factory=list)


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
            "answer": ABSTENTION_MESSAGE,
            "uncertainty": draft.uncertainty,
            "recommendations": [],
            "citations": [],
        }

    evidence_by_atom = {item["atom_id"]: item for item in evidence}
    if not draft.claims:
        raise CitationValidationError("Answered response contained no claims")
    claim_groups = [*draft.claims, *draft.caveats, *draft.recommendations]
    cited_ids = list(dict.fromkeys(
        atom_id
        for claim in claim_groups
        for atom_id in claim.cited_atom_ids
    ))
    invalid = [atom_id for atom_id in cited_ids if atom_id not in evidence_by_atom]
    if invalid:
        raise CitationValidationError(f"Model cited atoms outside retrieved evidence: {invalid}")

    def render(claim: GroundedClaim) -> str:
        clean_text = re.sub(r"\[(atom_[A-Za-z0-9_-]+)\]", "", claim.text).strip()
        citations = " ".join(f"[{atom_id}]" for atom_id in dict.fromkeys(claim.cited_atom_ids))
        return f"{clean_text} {citations}"

    answer_parts = [render(claim) for claim in draft.claims]
    answer_parts.extend(f"Caveat: {render(claim)}" for claim in draft.caveats)
    answer_text = " ".join(answer_parts)
    recommendations = [render(claim) for claim in draft.recommendations]

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
        "answer": answer_text,
        "uncertainty": draft.uncertainty,
        "recommendations": recommendations,
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
        "Treat each atom statement as the support boundary for its citation. Source context may clarify wording, but "
        "do not use context to make a claim that the cited atom statement does not itself support. Prefer atoms whose "
        "statements directly answer the question. For questions asking for problems or issues, claims must be direct "
        "problem or symptom statements. Troubleshooting attempts, workarounds, support outcomes, requests for a fix, "
        "meta statements, and pilot counterexamples are not additional problems and must not appear in claims. "
        "Put only material counterexamples or limitations in caveats, with their own atom citations. "
        "When combining separate examples, say 'the retrieved examples include' instead of implying prevalence, consensus, "
        "cross-segment scope, or causation. Do not infer that an issue is unresolved, widespread, primary, or caused by "
        "something unless a cited atom explicitly says so. "
        "Return at most four direct factual points as separate claims items with only the exact atom IDs that directly "
        "support that point. Cite one atom per claim unless every cited atom statement independently expresses the same "
        "claim. "
        "Keep one claim per item; do not attach a bundle of citations to a paragraph or use an atom merely because it has "
        "the same topic metadata. Use only atom IDs shown in the evidence. "
        "Recommendations must be returned only in the recommendations field, each as a separate evidence-linked item; "
        "otherwise return an empty recommendations list. Do not offer follow-up "
        "work or say 'if you want'. If retrieved evidence is mixed or includes limitations, state that explicitly. "
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
