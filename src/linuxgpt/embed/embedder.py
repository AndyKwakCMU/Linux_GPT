"""Wraps Ollama's batch embeddings endpoint for both chunk indexing and
query embedding, so both sides of retrieval always go through the exact
same model/call path."""

from __future__ import annotations

import ollama

from linuxgpt.config import settings

_client = ollama.Client(host=settings.ollama_host)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = _client.embed(model=settings.embedding_model, input=texts)
    return [list(vec) for vec in resp.embeddings]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
