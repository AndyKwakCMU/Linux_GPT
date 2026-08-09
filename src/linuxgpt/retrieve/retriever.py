"""Retrieval: hybrid search over the chunk store, converted back into
Chunk objects, then expanded via the symbol graph so referenced
macro/type/struct/function definitions are pulled into context even when
hybrid search alone doesn't surface them (e.g. a helper the question
doesn't name directly).

The symbol graph needs the full chunk corpus in memory to build its
name index -- at Phase 1 scale (F2FS only, ~2k chunks) that's trivial;
Phase 2's ~4-6x larger corpus is still comfortably laptop-sized. Cached
per-process since it doesn't change within a single CLI invocation.
"""

from __future__ import annotations

from linuxgpt.config import settings
from linuxgpt.embed.embedder import embed_query
from linuxgpt.ingest.chunk_merge import Chunk
from linuxgpt.ingest.symbol_graph import SymbolGraph
from linuxgpt.store.lancedb_store import hybrid_search, open_chunks_table

_cache: dict[str, object] = {}


def _row_to_chunk(row: dict) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        kernel_version=row["kernel_version"],
        file_path=row["file_path"],
        kind=row["kind"],
        symbol_name=row["symbol_name"] or None,
        start_line=row["start_line"],
        end_line=row["end_line"],
        text=row["text"],
        low_confidence_parse=row["low_confidence_parse"],
        provenance=row["provenance"],
        review_status=row.get("review_status") or None,
    )


def _load_all_chunks() -> list[Chunk]:
    table = open_chunks_table()
    rows = table.to_arrow().to_pylist()
    return [_row_to_chunk(r) for r in rows]


def _get_symbol_graph() -> SymbolGraph:
    if "graph" not in _cache:
        _cache["graph"] = SymbolGraph(_load_all_chunks())
    return _cache["graph"]  # type: ignore[return-value]


def retrieve(question: str) -> list[Chunk]:
    """Returns an ordered list of Chunks: hybrid-search hits first (most
    relevant first, per LanceDB's RRF-merged relevance score), followed by
    symbol-graph expansion chunks."""
    query_vector = embed_query(question)
    hits = hybrid_search(
        question,
        query_vector,
        vector_k=settings.retrieval_vector_top_k,
        fts_k=settings.retrieval_fts_top_k,
    )
    seed_chunks = [_row_to_chunk(r) for r in hits]

    graph = _get_symbol_graph()
    already_have = {c.chunk_id for c in seed_chunks}
    # Only expand from code-kind chunks: expansion scans a chunk's raw
    # text for identifier-shaped tokens that match a real symbol name.
    # Doc chunks are natural-language prose, not code -- confirmed in
    # testing that f2fs.rst's prose mentioning "LFS" (the Log-structured
    # File System abbreviation) spuriously matched an unrelated F2FS enum
    # constant literally named LFS, injecting noise for a feature meant
    # for code cross-references.
    expansion_seeds = [c for c in seed_chunks if c.kind != "doc"]
    expanded = graph.expand(expansion_seeds, already_have, settings.symbol_expansion_max_chunks)

    return seed_chunks + expanded
