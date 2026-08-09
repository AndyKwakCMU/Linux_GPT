"""Corrective retry loop: on verification failure, re-prompt the model
naming exactly which citations failed (or were missing/uncited entirely),
capped at settings.max_citation_retries. If still failing after the retry
budget, the caller (cli) surfaces the answer with failures visibly
flagged rather than hiding them -- for this audience (testing the tool's
integrity), visible failure is a feature, not something to paper over."""

from __future__ import annotations

from linuxgpt.config import settings
from linuxgpt.generate.chunk_ref import substitute_chunk_refs
from linuxgpt.generate.ollama_client import generate
from linuxgpt.ingest.chunk_merge import Chunk
from linuxgpt.verify.verifier import VerificationResult, verify_answer


def _failure_feedback(result: VerificationResult, unresolved_refs: list[int]) -> str:
    lines: list[str] = []
    if result.zero_citation_attempt:
        lines.append(
            "You made claims about the source code without citing a single [[CHUNK:n]] "
            "reference anywhere in your answer. Every factual claim about what the code "
            "does or contains (including struct field names) must cite the CHUNK it came "
            "from -- rewrite your answer so each claim references the chunk that supports it."
        )
    if result.uncited_blocks:
        lines.append(
            f"You wrote {len(result.uncited_blocks)} code block(s) without a valid "
            "[[CHUNK:n]] reference. Do not write source code directly -- reference a "
            "retrieved CHUNK by number instead, or state plainly that the relevant code "
            "isn't in the provided chunks."
        )
    for claim, reason in result.cited_failed:
        lines.append(
            f"Your citation for {claim.file_path} lines {claim.start_line}-{claim.end_line} "
            f"failed verification ({reason}). Only cite chunks exactly as given via [[CHUNK:n]]."
        )
    if unresolved_refs:
        lines.append(f"You referenced CHUNK numbers that don't exist: {unresolved_refs}.")
    return "\n".join(lines)


def generate_with_verification(
    messages: list[dict], chunks: list[Chunk]
) -> tuple[str, VerificationResult, int]:
    """Returns (rendered_answer, verification_result, attempts_used)."""
    conversation = list(messages)
    rendered = ""
    result: VerificationResult | None = None

    for attempt in range(settings.max_citation_retries + 1):
        raw = generate(conversation)
        rendered, unresolved = substitute_chunk_refs(raw, chunks)
        result = verify_answer(rendered, chunks_were_available=bool(chunks))

        if result.all_passed and not unresolved:
            return rendered, result, attempt + 1

        if attempt == settings.max_citation_retries:
            break

        conversation.append({"role": "assistant", "content": raw})
        feedback = _failure_feedback(result, unresolved)
        conversation.append(
            {"role": "user", "content": f"Citation check failed:\n{feedback}\nPlease rewrite your answer, fixing this."}
        )

    assert result is not None
    return rendered, result, settings.max_citation_retries + 1
