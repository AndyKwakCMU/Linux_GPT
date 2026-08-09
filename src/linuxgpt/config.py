"""Central settings for linuxgpt. All paths/model names live here so every
module (ingest, embed, store, retrieve, generate, verify, cli) reads from
one place instead of hardcoding values."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LINUXGPT_")

    manifest_path: Path = REPO_ROOT / "kernel_manifest.yaml"
    kernel_src_dir: Path = REPO_ROOT / ".kernel_src"
    lancedb_dir: Path = REPO_ROOT / ".lancedb"
    glossary_dir: Path = REPO_ROOT / "src" / "linuxgpt" / "glossary"
    # Hand-authored lesson .rst files + manifest.yaml (git-tracked source
    # of truth). Ingest copies these into kernel_src_dir/lessons/ so their
    # embedded citations resolve against the same pinned root as
    # everything else -- see ingest/pipeline.py.
    lessons_dir: Path = REPO_ROOT / "src" / "linuxgpt" / "lessons"

    ollama_host: str = "http://localhost:11434"
    generation_model: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    embedding_model: str = "nomic-embed-text"
    generation_temperature: float = 0.15

    # Confirmed live: at top_k=10/10 the qwen2.5-coder:7b-instruct-q4_K_M
    # model reliably abstained ("NOT INDEXED") on broad questions even
    # when the correct chunk was present in context, apparently
    # overwhelmed by ~10 mostly-irrelevant candidates; at 5/5 it
    # succeeded on the first attempt with the same underlying retrieval.
    # This is a small-model attention limitation, not a retrieval
    # correctness issue -- see project memory for the fuller finding.
    retrieval_vector_top_k: int = 6
    retrieval_fts_top_k: int = 6
    symbol_expansion_max_chunks: int = 6
    max_citation_retries: int = 2

    ctags_binary: str = "ctags"

    chunks_table: str = "chunks"


settings = Settings()
