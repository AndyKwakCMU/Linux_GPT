"""Deterministic substitution of [[CHUNK:n]] references in a generated
answer with the actual verbatim chunk text + a citation header. Byte-
perfect quoting for anything retrieved is guaranteed by construction here
-- there is no verification failure mode for these references, only for
free-form quotes the model wrote directly instead (see verify/verifier.py,
which is the safety net for that case)."""

from __future__ import annotations

import re

from linuxgpt.ingest.chunk_merge import Chunk

# Matches a whole "reference cluster": a run of double-bracketed content
# joined by "], [" separators, e.g. "[[CHUNK:3]]", or the malformed-but-
# clearly-intentional multi-ref variant observed live from the model,
# "[[CHUNK:3], [CHUNK:5]]". Chunk numbers are extracted separately from
# whatever's inside via _CHUNK_NUM_RE, so any reasonable punctuation
# between them (", ", "] [", etc.) is tolerated without weakening
# anything -- a malformed reference still only resolves to real,
# already-retrieved chunks, never to arbitrary text.
_CHUNK_REF_RE = re.compile(r"\[\[[^\[\]]*(?:\][,\s]*\[[^\[\]]*)*\]\]")
_CHUNK_NUM_RE = re.compile(r"CHUNK\s*:\s*(\d+)", re.IGNORECASE)

# Fence language tag by chunk kind, so a prose doc excerpt doesn't render
# (and doesn't get syntax-highlighted) as if it were C source. Shared with
# prompts.py so chunks are shown to the model in the same fence style
# they'll be cited back in.
_FENCE_LANG_BY_KIND = {"doc": "rst", "lesson": "rst"}
_DEFAULT_FENCE_LANG = "c"


def fence_lang_for_kind(kind: str) -> str:
    return _FENCE_LANG_BY_KIND.get(kind, _DEFAULT_FENCE_LANG)


def safe_fence(text: str) -> str:
    """A backtick fence marker guaranteed longer than any backtick run
    already inside text -- CommonMark's own nested-fence convention (a
    fence of N backticks can only be closed by a run of >= N backticks).
    Needed because a lesson chunk can itself contain an embedded
    ```-fenced citation (confirmed live: a lesson section quoting real
    doc content, then re-quoted via [[CHUNK:n]], truncated at the inner
    fence when both used a plain triple-backtick)."""
    runs = re.findall(r"`+", text)
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)


def render_citation(chunk: Chunk) -> str:
    lang = fence_lang_for_kind(chunk.kind)
    fence = safe_fence(chunk.text)
    header = f"{fence}{lang} file={chunk.file_path} lines={chunk.start_line}-{chunk.end_line}"
    if chunk.review_status:
        header += f" status={chunk.review_status}"
    return f"{header}\n{chunk.text}\n{fence}"


def substitute_chunk_refs(answer: str, chunks: list[Chunk]) -> tuple[str, list[int]]:
    """Replaces every [[CHUNK:n]] (or multi-ref cluster like
    "[[CHUNK:3], [CHUNK:5]]") in answer with the verbatim cited source(s).
    Returns (rendered_answer, unresolved_indices): indices the model
    referenced that don't correspond to any retrieved chunk (1-indexed, as
    the model saw them) -- the caller should treat these as a citation
    failure worth a corrective retry."""
    unresolved: list[int] = []

    def repl(m: re.Match) -> str:
        numbers = [int(x) for x in _CHUNK_NUM_RE.findall(m.group(0))]
        if not numbers:
            return m.group(0)  # a "[[...]]" that isn't actually a CHUNK reference
        citations = []
        for idx in numbers:
            if 1 <= idx <= len(chunks):
                citations.append(render_citation(chunks[idx - 1]))
            else:
                unresolved.append(idx)
        if not citations:
            return m.group(0)  # leave visible rather than silently dropping it
        return "\n" + "\n\n".join(citations) + "\n"

    rendered = _CHUNK_REF_RE.sub(repl, answer)
    return rendered, unresolved
