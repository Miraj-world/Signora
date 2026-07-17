"""HTTP API for Signora grounded Voice of Customer answers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from answer_generation import DEFAULT_POLICY_PATH, generate_answer, load_threshold
from query_retrieval import resolve_index_dir, retrieve
from retrieval_models import load_dotenv


class AnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    mode: Literal["pipeline", "recall"] = "recall"
    candidate_pool: int = Field(default=250, ge=1, le=2000)
    product_area: str | None = None
    customer_segment: str | None = None
    source_type: str | None = None
    target_product: str | None = None
    severity: str | None = None
    include_evidence: bool = False


def readiness() -> dict:
    load_dotenv()
    index_dir = resolve_index_dir(None)
    required = [
        index_dir / "index_manifest.json",
        DEFAULT_POLICY_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    key_configured = bool(os.environ.get("OPENAI_API_KEY"))
    ready = not missing and key_configured
    return {
        "status": "ready" if ready else "not_ready",
        "index_dir": str(index_dir),
        "openai_api_key_configured": key_configured,
        "missing_files": missing,
    }


def answer_question(request: AnswerRequest) -> dict:
    filters = {
        "product_area": request.product_area,
        "customer_segment": request.customer_segment,
        "source_type": request.source_type,
        "target_product": request.target_product,
        "severity": request.severity,
    }
    retrieval = retrieve(
        request.query,
        top_k=request.top_k,
        mode=request.mode,
        candidate_pool=request.candidate_pool,
        filters=filters,
    )
    threshold = load_threshold(retrieval["embedding_profile"], request.mode)
    retrieval["abstain_threshold"] = threshold
    retrieval["should_abstain"] = retrieval["top_score"] < threshold
    if retrieval["should_abstain"]:
        retrieval["results"] = []
    answer = generate_answer(request.query, retrieval)

    response = {
        "status": answer["status"],
        "answer": answer["answer"],
        "uncertainty": answer["uncertainty"],
        "recommendations": answer["recommendations"],
        "citations": answer["citations"],
        "generation_model": answer["generation_model"],
        "retrieval": {
            "embedding_profile": retrieval["embedding_profile"],
            "mode": retrieval["retrieval_mode"],
            "top_score": retrieval["top_score"],
            "abstain_threshold": threshold,
            "should_abstain": retrieval["should_abstain"],
            "evidence_count": len(retrieval["results"]),
        },
    }
    if request.include_evidence:
        response["evidence"] = retrieval["results"]
    return response


app = FastAPI(
    title="Signora API",
    version="0.1.0",
    description="Citation-validated Voice of Customer retrieval and grounded answers.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    result = readiness()
    if result["status"] != "ready":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/v1/answers")
def answers(request: AnswerRequest) -> dict:
    state = readiness()
    if state["status"] != "ready":
        raise HTTPException(status_code=503, detail=state)
    try:
        return answer_question(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
