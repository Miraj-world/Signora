"""Provider adapters.  All adapters expose the same document/query interface."""
import os

import numpy as np

from common import load_dotenv, normalize


MODELS = {
    "minilm": {"name": "all-MiniLM-L6-v2", "backend": "sentence_transformers"},
    "mpnet": {"name": "all-mpnet-base-v2", "backend": "sentence_transformers"},
    "openai_small": {"name": "text-embedding-3-small", "backend": "openai"},
    "openai_large": {"name": "text-embedding-3-large", "backend": "openai"},
    "cohere": {"name": "embed-english-v3.0", "backend": "cohere"},
    "google": {"name": "text-embedding-004", "backend": "google"},
    "voyage": {"name": "voyage-large-2", "backend": "voyage"},
}


def available_slugs():
    return list(MODELS)


def create_encoder(slug):
    load_dotenv()
    if slug not in MODELS:
        raise ValueError(f"unknown model {slug}; choose one of: {', '.join(available_slugs())}")
    spec = MODELS[slug]
    name, backend = spec["name"], spec["backend"]

    if backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        return spec, lambda texts, kind: normalize(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))
    if backend == "openai":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI()
        return spec, lambda texts, kind: normalize(np.array([row.embedding for row in client.embeddings.create(input=texts, model=name).data]))
    if backend == "cohere":
        import cohere
        if not os.environ.get("COHERE_API_KEY"):
            raise RuntimeError("COHERE_API_KEY is not set")
        client = cohere.Client(os.environ["COHERE_API_KEY"])
        def encode(texts, kind):
            result = client.embed(texts=texts, model=name, input_type="search_query" if kind == "query" else "search_document")
            values = result.embeddings.float_ if hasattr(result.embeddings, "float_") else result.embeddings
            return normalize(np.array(values))
        return spec, encode
    if backend == "google":
        import google.generativeai as genai
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        def encode(texts, kind):
            task = "retrieval_query" if kind == "query" else "retrieval_document"
            return normalize(np.array([genai.embed_content(model=f"models/{name}", content=text, task_type=task)["embedding"] for text in texts]))
        return spec, encode
    if backend == "voyage":
        import voyageai
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set")
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        return spec, lambda texts, kind: normalize(np.array(client.embed(texts, model=name, input_type="query" if kind == "query" else "document").embeddings))
    raise AssertionError(f"unsupported backend {backend}")
