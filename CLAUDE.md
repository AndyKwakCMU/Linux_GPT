# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`linuxgpt`: citation-verified RAG Q&A over Linux kernel source, scoped to
F2FS + closest neighboring subsystems (VFS/block/mm docs, not yet their
code). Built for Jaeguek Kim, the author of F2FS, to test answer integrity
against his own knowledge — see `README.md` for full project context,
architecture, and status.

## Commands

```bash
source .venv/bin/activate      # venv already exists at repo root
pip install -e ".[dev]"        # after dependency changes

linuxgpt ingest                # fetch pinned kernel (kernel_manifest.yaml), chunk, embed, index
linuxgpt ingest --force        # re-fetch/re-index even if already done
linuxgpt ask "<question>"      # citation-verified Q&A against the index

pytest                         # full suite (25 tests, ~instant)
pytest tests/test_lessons.py   # re-verifies every hand-authored lesson's citations
```

No linter/formatter is configured yet.

`pretrain/` (from-scratch piece training) has its **own venv** and no
dependency on `src/linuxgpt/` -- run it from inside `pretrain/`:

```bash
cd pretrain && source .venv/bin/activate   # torch, tokenizers, numpy, pyyaml
python build_corpus.py --piece f2fs        # pieces/f2fs.yaml -> data/f2fs/corpus.jsonl (needs .cache/linux-6.12.tar.xz)
python prepare_data.py --piece f2fs        # -> data/f2fs/{train,val}.bin
python train.py --piece f2fs --preset tiny # -> checkpoints/f2fs/{ckpt,ckpt_best}.pt, log.csv
python eval_probes.py --piece f2fs         # held-out functions vs bigram/unigram baselines
```

Requires a local Ollama daemon with `nomic-embed-text` and
`qwen2.5-coder:7b-instruct-q4_K_M` pulled. Everything runs locally — no
cloud calls, no GPU required.

## Architecture

`kernel_manifest.yaml` pins the kernel version and which paths/docs get
indexed → `ingest/` fetches + sha256-verifies the tarball into
`.kernel_src/`, then chunks it (tree-sitter+ctags hybrid for C code,
fence-safe RST-section splitting for docs/lessons) → `embed/` (Ollama) →
`store/` (LanceDB hybrid vector+FTS, `.lancedb/`) → `retrieve/` (hybrid
search + symbol-graph expansion) → `generate/` (citation-enforced prompting
via `[[CHUNK:n]]` tags) → `verify/` (re-checks every cited claim
byte-for-byte against the pinned source tree, retries generation on
failure). All settings/paths/model names live in `src/linuxgpt/config.py` —
change them there, not by hardcoding elsewhere.

Content types share one LanceDB table but differ in `kind`: `code` (F2FS
source), `doc` (real kernel `Documentation/*.rst`), `lesson`
(hand-authored synthesis in `src/linuxgpt/lessons/`, flagged `UNREVIEWED`
in CLI output until a human approves it). Lessons are staged into
`.kernel_src/lessons/` at ingest time so their citations resolve against
the same pinned root as real kernel content.

`pretrain/` trains small GPTs ("pieces") from scratch on kernel source --
see `pretrain/README.md`. A piece's training files are set by
`pretrain/pieces/<name>.yaml`, independent of `kernel_manifest.yaml`.

## Conventions and gotchas worth knowing before editing

- **Scope discipline**: indexing is deliberately narrow (F2FS + directly
  touching subsystems), not the full kernel tree. Don't expand scope beyond
  `kernel_manifest.yaml`'s Phase 2 entries without being asked — see
  README's "Scope discipline" section.
- **Don't weaken the verifier** to hide model compliance failures. An
  honest `NOT FULLY VERIFIED` is the intended, safety-relevant behavior for
  this tool's purpose — the open problem is generation-side compliance, not
  verification-side strictness. See README's "Known limitation" section.
- **Citation fence handling is load-bearing**: `generate/chunk_ref.py`'s
  `safe_fence()` and `verify/citation_parser.py`'s variable-length fence
  regex exist because lesson chunks can themselves contain fenced
  citations quoting other content — a fixed triple-backtick fence silently
  truncates in that case. Any new content type that can nest citations
  needs to go through the same fence-safe path.
- **Retrieval `top_k` is tuned low on purpose** (`retrieval_vector_top_k` /
  `retrieval_fts_top_k` = 6 in `config.py`): the 7B generation model
  reliably abstains when given ~10+ candidates even when the right one is
  present. Don't raise this without re-testing compliance, and don't
  "fix" abstention by prompt-tweaking alone without new evidence — already
  tried with diminishing returns.
- **Embeddings are not code-aware**: `nomic-embed-text` can rank
  lexically-similar-but-semantically-empty chunks (e.g. bare header-guard
  macros) above the real answer. Value-less `#define`s are already
  filtered at chunk time in `chunk_treesitter.py`; watch for similar
  surface-token noise if retrieval quality looks off elsewhere.
- **Pieces never bypass the verifier.** Anything a `pretrain/` model
  produces may steer retrieval or be shown labeled as unverified intuition;
  it never counts as a citation. Same reasoning as "don't weaken the
  verifier" above.
- **The shared tokenizer is frozen** (`pretrain/tokenizers/kernel-v6.12-16k.json`,
  committed). Retraining it makes every existing checkpoint incompatible;
  checkpoints record its sha256 and the scripts refuse mismatches. It is
  trained on the whole kernel tarball -- that is *not* an expansion of RAG
  indexing scope; scope discipline still applies to `kernel_manifest.yaml`
  and to what each piece's *weights* train on (`pieces/*.yaml`).
- `pretrain/kernel_tarball.py` deliberately duplicates
  `ingest/fetch.py`'s checksum + path-matching logic (pretrain stays
  standalone); keep them in sync.

## Next steps

See README's "Next steps" section (F2FS piece training + bridge, Phase 2
VFS/block/mm code indexing, interface decision, citation-compliance
improvements, lesson review) and `pretrain/README.md`'s "Roadmap".
