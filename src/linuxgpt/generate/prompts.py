"""System + user prompt construction enforcing the citation contract:
every factual claim about the codebase is either a [[CHUNK:n]] reference
to retrieved source, or explicitly labeled general background knowledge.
"""

from __future__ import annotations

import yaml

from linuxgpt.config import settings
from linuxgpt.generate.chunk_ref import fence_lang_for_kind, safe_fence
from linuxgpt.ingest.chunk_merge import Chunk

ABSTAIN_MARKER = "NOT INDEXED:"

SYSTEM_PROMPT = """You are a Linux kernel source assistant, scoped to a pinned, indexed subset of the kernel (F2FS and its closest neighboring subsystems). Follow these rules strictly:

1. Every factual claim about what the indexed source code does, contains, or means MUST be grounded in one of the numbered CHUNKS provided in the user message. Reference it inline as [[CHUNK:n]] immediately after the claim it supports. Do not retype or paraphrase source code from memory -- reference the chunk instead of quoting it yourself.
1a. CHUNKS come in two kinds, both cited the same way via [[CHUNK:n]]: kind=doc chunks are excerpts from the kernel's own official documentation (Documentation/...) -- these are the right source for "what is X" / big-picture / design-rationale questions, and are just as citable as code. kind=function/struct/etc. chunks are exact source declarations. Prefer doc chunks for conceptual/overview questions and code chunks for "what exactly does this function/struct do" questions; use both together when a question needs both a concept and its implementation.
2. If you explain a general C, GNU C, or computer-hardware concept that is not itself a claim about this codebase (e.g. what `typeof` does in general, or how NAND flash erase blocks work in general), you may draw on the GENERAL BACKGROUND section if provided, and must label such an explanation as general background, not a source citation.
3. The CHUNKS you're given come from an automated search and may not actually be relevant to the question -- do not force an answer out of them. If none of the provided CHUNKS actually answer the question, do not guess and do not describe chunks that don't answer it: respond with exactly "__ABSTAIN_MARKER__" followed by a one-sentence reason, and nothing else. Never answer an unsupported question from prior training knowledge about the Linux kernel -- the kernel changes across versions and across subsystems, and only the pinned, indexed source below is authoritative here.
4. Never attach a [[CHUNK:n]] reference to a claim that chunk doesn't actually support.
5. It is fine, and preferred, to abstain (rule 3) rather than guess -- but only when the chunks genuinely don't cover the question. If a chunk defining the exact struct/function asked about is present, use it; don't abstain just because the question is broad.

Example of the expected style, given a CHUNK:3 defining a struct foo with a count field:

"The foo struct tracks a running total via its count field [[CHUNK:3]]."

Note the citation comes right after the specific claim it supports -- cite as many different CHUNK numbers as you have distinct claims, don't withhold a citation just because you're not quoting code verbatim.
""".replace("__ABSTAIN_MARKER__", ABSTAIN_MARKER)


def format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        lang = fence_lang_for_kind(c.kind)
        # A lesson chunk can itself contain an embedded ```-fenced
        # citation -- use a longer fence when needed (same convention as
        # generate/chunk_ref.render_citation) so the model sees the whole
        # chunk, not a version truncated at the first inner fence.
        fence = safe_fence(c.text)
        parts.append(
            f"CHUNK:{i}\n"
            f"file={c.file_path} lines={c.start_line}-{c.end_line} kind={c.kind} name={c.symbol_name or ''}\n"
            f"{fence}{lang}\n{c.text}\n{fence}"
        )
    return "\n\n".join(parts)


def _load_glossary() -> list[dict]:
    """Loads every glossary/*.yaml file (e.g. gnu_c_idioms.yaml,
    hardware_concepts.yaml) -- dropping a new curated background file in
    the directory is enough to make it available, no registration needed."""
    entries: list[dict] = []
    if not settings.glossary_dir.is_dir():
        return entries
    for path in sorted(settings.glossary_dir.glob("*.yaml")):
        entries.extend(yaml.safe_load(path.read_text()) or [])
    return entries


def select_glossary_entries(question: str, *, max_entries: int = 5) -> list[dict]:
    q_lower = question.lower()
    matches = [
        entry
        for entry in _load_glossary()
        if any(kw.lower() in q_lower for kw in entry.get("keywords", []))
    ]
    return matches[:max_entries]


def format_glossary(entries: list[dict]) -> str:
    return "\n\n".join(f"- {e['term']}: {e['explanation'].strip()}" for e in entries)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    kernel_version = chunks[0].kernel_version if chunks else "unknown"
    user_content = f"Indexed kernel version: {kernel_version}\n\n"

    if chunks:
        user_content += f"CHUNKS:\n{format_chunks(chunks)}\n\n"
    else:
        user_content += "CHUNKS: (none retrieved for this question)\n\n"

    glossary_entries = select_glossary_entries(question)
    if glossary_entries:
        user_content += (
            "GENERAL BACKGROUND (not source-grounded -- general C/GNU-C/hardware "
            f"reference only):\n{format_glossary(glossary_entries)}\n\n"
        )

    user_content += f"QUESTION: {question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
