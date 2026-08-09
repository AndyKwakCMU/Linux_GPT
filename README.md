# Linux_GPT (linuxgpt)

Citation-verified RAG Q&A over Linux kernel source, scoped to **F2FS** and its
closest neighboring subsystems. Built for Jaeguek Kim, the author of F2FS, so
he can test whether the tool's answers hold up against his own knowledge of
the code — every claim the model makes is checked byte-for-byte against the
pinned kernel source before it's shown as verified.

## Why this exists

Frontier LLMs hallucinate about kernel source contents. Rather than
fine-tuning a model on kernel source (which would still hallucinate and go
stale every release), this is retrieval-augmented generation: a small local
model retrieves and quotes real, indexed source with `[[CHUNK:n]]`
citations, and a verifier independently re-checks every citation against the
actual pinned files before the answer is labeled `VERIFIED`. Failure is
surfaced, not hidden — an answer that doesn't check out is shown as
`NOT FULLY VERIFIED` rather than silently cleaned up.

## Current status (as of 2026-08-09)

**Phase 1 is built and working end-to-end.** 25/25 pytest tests pass.

- Indexed content: `fs/f2fs/` from kernel v6.12, plus a curated set of real
  kernel `Documentation/*.rst` files (F2FS, VFS, journalling, path lookup,
  block layer, mm, NAND/MTD, DMA/cache/IRQ background), plus two
  hand-authored "lesson" `.rst` files that synthesize across those
  subsystems and are flagged `UNREVIEWED` until Jaeguek signs off on them.
- Everything runs locally: Ollama for embeddings + generation, LanceDB for
  the vector/FTS store. No cloud calls, no GPU required.
- VFS core, block layer, and `mm/` proper are **not yet indexed** — only
  their documentation is. The code paths for those subsystems are stubbed
  out in `kernel_manifest.yaml` as commented-out "Phase 2" entries.

### Known limitation (not a bug, still open)

The generation model (`qwen2.5-coder:7b-instruct-q4_K_M`) has an unreliable
compliance ceiling with the citation convention on compound/broad questions —
it sometimes abstains ("NOT INDEXED") even when the right chunk was
retrieved, or fabricates plausible-but-fictional details without attempting
a citation. The two-layer verifier reliably catches all of this and reports
it honestly; that's the safety property that matters. Fixing the underlying
compliance rate (not the verifier) is the open problem — candidates are a
larger model, Ollama structured-output/JSON mode, or a reranker ahead of
generation instead of raw RRF.

## Architecture

```
kernel_manifest.yaml (pins kernel v6.12 + which paths/docs to index)
        │
        ▼
  fetch (tarball + sha256 verify)  →  .kernel_src/
        │
        ▼
  chunk  ── tree-sitter + ctags hybrid, C-aware (functions/structs, not
        │    fixed-token windows) for code; RST-section-aware for docs
        │    and lessons (fence-safe: won't misparse quoted headings or
        │    nested ``` citations)
        ▼
  embed  (Ollama, nomic-embed-text)
        │
        ▼
  store  (LanceDB: hybrid vector + full-text search, .lancedb/)
        │
        ▼
  retrieve  (hybrid search + symbol-graph expansion to pull in
        │    related definitions)
        ▼
  generate  (Ollama, qwen2.5-coder:7b-instruct-q4_K_M, citation-enforced
        │    prompting via [[CHUNK:n]] tags)
        ▼
  verify  (re-checks every cited claim byte-for-byte against the pinned
        │   source tree; retries generation on failure up to a limit)
        ▼
  `linuxgpt ask` CLI  →  Answer + VERIFIED/NOT FULLY VERIFIED status +
                          footer manifest of exactly which files/lines
                          backed the answer
```

## Project layout

```
kernel_manifest.yaml         # pins kernel version + indexed paths/docs
src/linuxgpt/
  cli/main.py                 # `linuxgpt ingest`, `linuxgpt ask`
  config.py                   # all settings/paths/model names in one place
  ingest/                     # fetch, chunk (tree-sitter/ctags/doc), symbol graph
  embed/                      # Ollama embedding client
  store/                      # LanceDB store
  retrieve/                   # hybrid retrieval + symbol expansion
  generate/                   # prompt building, citation-tag rendering
  verify/                     # citation parsing + byte-level verification + retry
  glossary/                   # hand-authored concept glossaries (YAML)
  lessons/                    # hand-authored synthesis docs (UNREVIEWED until approved)
tests/                        # pytest suite (25 tests)
scripts/fetch_kernel_source.py
```

## Running it

Requires [Ollama](https://ollama.com) running locally with
`nomic-embed-text` and `qwen2.5-coder:7b-instruct-q4_K_M` pulled.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

linuxgpt ingest        # fetch v6.12, chunk, embed, index (idempotent; --force to redo)
linuxgpt ask "What is F2FS?"

pytest                 # 25 tests
```

## Scope discipline

Indexing is intentionally narrow — F2FS plus VFS/block/mm *only where they
directly touch F2FS's design* — not the full ~30M LOC kernel. This keeps the
index tractable and keeps every answer verifiable by someone who knows F2FS
well. Don't scope-creep to the full kernel tree; expand via
`kernel_manifest.yaml`'s commented-out Phase 2 entries deliberately, not by
default.

## Next steps

- Decide on and build a real interface (CLI is the current stopgap; a local
  web UI is deferred, not decided against).
- Improve citation compliance on compound questions (larger model /
  structured output / reranker — see "Known limitation" above).
- Get Jaeguek's review on the two `UNREVIEWED` lessons.
- Phase 2: uncomment and index VFS core, block layer, and `mm/` source
  (not just their docs) in `kernel_manifest.yaml`.
