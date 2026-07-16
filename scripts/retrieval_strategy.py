"""Shared ranking strategies for benchmark and production retrieval."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


METADATA_FIELDS = (
    "topic",
    "subtopic",
    "product_area",
    "feedback_type",
    "severity",
    "customer_segment",
)


@dataclass(frozen=True)
class ScoredRow:
    score: float
    semantic: float
    keyword: float
    metadata: float
    row: dict


def tokenize(text: str | None) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", (text or "").lower())
        if len(token) > 1
    ]


def overlap_score(query_tokens: Counter, text: str | None) -> float:
    if not query_tokens:
        return 0.0
    text_counts = Counter(tokenize(text))
    if not text_counts:
        return 0.0
    overlap = sum(min(text_counts[token], count) for token, count in query_tokens.items())
    return overlap / max(1, sum(query_tokens.values()))


def metadata_topic_score(row: dict, query_tokens: Counter) -> float:
    if not query_tokens:
        return 0.0
    metadata_text = " ".join(str(row.get(field) or "") for field in METADATA_FIELDS)
    metadata_tokens = set(tokenize(metadata_text.replace("_", " ")))
    if not metadata_tokens:
        return 0.0
    overlap = set(query_tokens) & metadata_tokens
    return len(overlap) / max(1, min(len(set(query_tokens)), len(metadata_tokens)))


def rank_rows(
    query: str,
    rows: list[dict],
    semantic_scores,
    mode: str = "pipeline",
    candidate_pool: int = 250,
    semantic_weight: float = 0.75,
) -> list[ScoredRow]:
    """Rank filtered rows with the baseline pipeline or recall-oriented pass."""
    if mode not in {"dense", "pipeline", "recall"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be at least 1")

    query_tokens = Counter(tokenize(query))
    scored = []
    for row in rows:
        semantic = float(semantic_scores[row["position"]])
        keyword = overlap_score(query_tokens, row.get("search_text"))
        metadata = metadata_topic_score(row, query_tokens)
        if mode == "dense":
            score = semantic
        elif mode == "pipeline":
            score = semantic_weight * semantic + (1 - semantic_weight) * keyword
        else:
            score = 0.55 * semantic + 0.20 * keyword + 0.25 * metadata
        scored.append(ScoredRow(score, semantic, keyword, metadata, row))

    if mode != "recall":
        return sorted(scored, key=lambda item: item.score, reverse=True)

    semantic_pool = sorted(scored, key=lambda item: item.semantic, reverse=True)[:candidate_pool]
    keyword_pool = sorted(
        (item for item in scored if item.keyword > 0),
        key=lambda item: item.keyword,
        reverse=True,
    )[:candidate_pool]
    metadata_pool = sorted(
        (item for item in scored if item.metadata > 0),
        key=lambda item: item.metadata,
        reverse=True,
    )[:candidate_pool]

    broadened = {}
    for item in semantic_pool + keyword_pool + metadata_pool:
        key = item.row.get("atom_id") or item.row["position"]
        broadened[key] = item
    return sorted(broadened.values(), key=lambda item: item.score, reverse=True)
