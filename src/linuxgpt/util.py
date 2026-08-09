"""Shared helper for reading source files as 1-indexed line arrays. Used
identically at ingest time (to build chunk.text) and at verify time (to
re-check a citation against the pinned source) so the two can never drift
out of sync with each other."""

from __future__ import annotations

from pathlib import Path


def read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.split("\n")


def slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    """start_line/end_line are 1-indexed and inclusive."""
    return "\n".join(lines[start_line - 1 : end_line])
