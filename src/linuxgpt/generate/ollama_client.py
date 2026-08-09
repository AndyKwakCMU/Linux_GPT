"""Thin wrapper around Ollama chat for answer generation."""

from __future__ import annotations

import ollama

from linuxgpt.config import settings

_client = ollama.Client(host=settings.ollama_host)


def generate(messages: list[dict]) -> str:
    resp = _client.chat(
        model=settings.generation_model,
        messages=messages,
        options={"temperature": settings.generation_temperature},
    )
    return resp.message.content or ""
