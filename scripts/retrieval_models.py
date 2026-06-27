"""Embedding adapters used by the production retrieval pipeline.

The benchmark showed `text-embedding-3-small` is the practical hosted winner,
while `all-mpnet-base-v2` is the strongest local baseline.  This module keeps
those choices explicit so the retrieval index and query path use the same
model profile.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = "openai_small"
LOCAL_BASELINE_PROFILE = "mpnet"

MODEL_PROFILES = {
    "openai_small": {
        "name": "text-embedding-3-small",
        "backend": "openai",
        "role": "production_default",
        "benchmark_note": "Practical winner: strongest core retrieval metrics at lower cost than large.",
    },
    "openai_large": {
        "name": "text-embedding-3-large",
        "backend": "openai",
        "role": "hosted_comparison",
        "benchmark_note": "Useful comparison; led abstention in one run but costlier.",
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "backend": "sentence_transformers",
        "role": "local_baseline",
        "benchmark_note": "Strongest local baseline and strongest held-out local abstention result.",
    },
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "backend": "sentence_transformers",
        "role": "fast_local_baseline",
        "benchmark_note": "Fast local baseline; keyword fusion improved ranking.",
    },
}


def load_dotenv(path: Path | None = None) -> None:
    """Load a local .env without overwriting already-set environment values."""
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def normalize(values: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def resolve_profile(profile: str | None) -> tuple[str, dict]:
    slug = profile or os.environ.get("SIGNORA_EMBEDDING_PROFILE") or DEFAULT_PROFILE
    if slug not in MODEL_PROFILES:
        choices = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown embedding profile '{slug}'. Choose one of: {choices}")
    return slug, MODEL_PROFILES[slug]


def create_encoder(profile: str | None = None) -> tuple[str, dict, Callable[[list[str], str], np.ndarray]]:
    """Return `(slug, spec, encode)` for document/query embedding.

    The encode callable accepts a list of texts and a kind: "document" or
    "query".  All returned vectors are normalized float32 arrays so dot product
    is cosine similarity.
    """
    load_dotenv()
    slug, spec = resolve_profile(profile)
    backend = spec["backend"]
    name = spec["name"]

    if backend == "openai":
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or run with --profile mpnet for the local baseline."
            )
        client = OpenAI()

        def encode(texts: list[str], kind: str) -> np.ndarray:
            response = client.embeddings.create(input=texts, model=name)
            return normalize([row.embedding for row in response.data])

        return slug, spec, encode

    if backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(name)

        def encode(texts: list[str], kind: str) -> np.ndarray:
            return normalize(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))

        return slug, spec, encode

    raise AssertionError(f"Unsupported embedding backend: {backend}")
