"""Orchestrates the full ingest pipeline: fetch -> chunk -> embed -> store.
Kept as a single reusable entry point so the CLI's `ingest` command and any
ad hoc re-indexing script call the exact same path."""

from __future__ import annotations

import shutil

import yaml

from linuxgpt.config import settings
from linuxgpt.embed.embedder import embed_texts
from linuxgpt.ingest.chunk_doc import build_doc_chunks
from linuxgpt.ingest.chunk_merge import Chunk, build_chunks
from linuxgpt.ingest.fetch import fetch, load_manifest
from linuxgpt.store.lancedb_store import write_chunks

_EMBED_BATCH_SIZE = 128


def _load_lesson_review_status() -> dict[str, str]:
    manifest_path = settings.lessons_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    data = yaml.safe_load(manifest_path.read_text()) or {}
    return {filename: entry["status"] for filename, entry in data.items()}


def _stage_lessons_into(source_root) -> None:
    """Copies the git-tracked lesson .rst files into source_root/lessons/
    so their embedded citations resolve against the same pinned root as
    real code/doc citations -- see config.settings.lessons_dir."""
    dest = source_root / "lessons"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for path in settings.lessons_dir.glob("*.rst"):
        shutil.copy2(path, dest / path.name)


def ingest_all(*, force_fetch: bool = False) -> list[Chunk]:
    manifest = load_manifest()
    source_root = fetch(manifest, force=force_fetch)
    chunks = build_chunks(source_root, manifest.kernel_version)
    chunks += build_doc_chunks(source_root, manifest.kernel_version)

    _stage_lessons_into(source_root)
    chunks += build_doc_chunks(
        source_root,
        manifest.kernel_version,
        kind="lesson",
        scan_subdir="lessons",
        review_status_by_file=_load_lesson_review_status(),
    )

    vectors: list[list[float]] = []
    for i in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[i : i + _EMBED_BATCH_SIZE]
        vectors.extend(embed_texts([c.text for c in batch]))

    write_chunks(chunks, vectors)
    return chunks
