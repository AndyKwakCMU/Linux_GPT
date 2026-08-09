"""Embedded, persistent LanceDB store for chunks: hybrid vector + full-text
search, RRF-merged (LanceDB's default hybrid rerank), with no server
process to manage -- confirmed against a local probe table before wiring
in: `.search(query_type="hybrid").vector(v).text(q)` is the manual-hybrid
call shape needed here since we supply our own Ollama embeddings rather
than registering a LanceDB embedding function.

`create_fts_index` is deprecated in favor of `create_index(config=FTS())`
in lancedb 0.36, but the replacement's parameter shape for a plain
Table.create_index() didn't match a text column in testing -- the
deprecated call still works correctly, so it's used here with the warning
suppressed rather than chasing an unstable/unclear replacement signature.
"""

from __future__ import annotations

import warnings

import lancedb
from lancedb.pydantic import LanceModel, Vector

from linuxgpt.config import settings
from linuxgpt.ingest.chunk_merge import Chunk

EMBEDDING_DIM = 768  # nomic-embed-text output dimensionality


class ChunkRow(LanceModel):
    chunk_id: str
    kernel_version: str
    file_path: str
    kind: str
    symbol_name: str
    start_line: int
    end_line: int
    text: str
    low_confidence_parse: bool
    provenance: str
    review_status: str
    vector: Vector(EMBEDDING_DIM)


def _connect():
    settings.lancedb_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(settings.lancedb_dir))


def write_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Overwrites the chunks table with the given chunks + their
    precomputed embedding vectors (same order, same length)."""
    assert len(chunks) == len(vectors), "chunks and vectors must be aligned 1:1"
    db = _connect()
    rows = [
        {
            "chunk_id": c.chunk_id,
            "kernel_version": c.kernel_version,
            "file_path": c.file_path,
            "kind": c.kind,
            "symbol_name": c.symbol_name or "",
            "start_line": c.start_line,
            "end_line": c.end_line,
            "text": c.text,
            "low_confidence_parse": c.low_confidence_parse,
            "provenance": c.provenance,
            "review_status": c.review_status or "",
            "vector": vec,
        }
        for c, vec in zip(chunks, vectors)
    ]
    table = db.create_table(settings.chunks_table, data=rows, schema=ChunkRow, mode="overwrite")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        table.create_fts_index("text", replace=True)


def open_chunks_table():
    db = _connect()
    return db.open_table(settings.chunks_table)


def hybrid_search(query_text: str, query_vector: list[float], *, vector_k: int, fts_k: int) -> list[dict]:
    table = open_chunks_table()
    results = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .limit(max(vector_k, fts_k))
        .to_list()
    )
    return results
