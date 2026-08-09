"""Tests for the RST section chunker. Boundary behavior here was
spike-validated against the real f2fs.rst and mtdnand.rst (title section
correctly absorbs the intro paragraph before the first subsection; a
parent heading immediately followed by a subheading, with no body text of
its own, correctly produces a short heading-only section rather than
swallowing the child's content) -- these tests pin that behavior down as
regressions using a synthetic snippet mirroring that real structure.
"""

from pathlib import Path

from linuxgpt.ingest.chunk_doc import build_doc_chunks, find_sections
from linuxgpt.util import read_lines, slice_lines

_SAMPLE = """.. SPDX-License-Identifier: GPL-2.0

====================
Doc Title Here
====================

Some intro text before any subsection.
More intro text.

Background and Design issues
============================

Log-structured File System (LFS)
--------------------------------
LFS body text line 1.
LFS body text line 2.

Wandering Tree Problem
----------------------
Wandering tree body text.
"""


def test_title_section_absorbs_intro_paragraph() -> None:
    lines = _SAMPLE.split("\n")
    sections = find_sections(lines)
    title_section = sections[0]
    assert title_section.title == "Doc Title Here"
    assert title_section.level == 1
    text = slice_lines(lines, title_section.start_line, title_section.end_line)
    assert "Some intro text before any subsection." in text
    assert "More intro text." in text
    assert "Background" not in text  # doesn't bleed into the next section


def test_heading_only_parent_section_is_short_not_swallowing_child() -> None:
    lines = _SAMPLE.split("\n")
    sections = find_sections(lines)
    parent = next(s for s in sections if s.title == "Background and Design issues")
    child = next(s for s in sections if s.title == "Log-structured File System (LFS)")
    assert parent.end_line < child.start_line
    parent_text = slice_lines(lines, parent.start_line, parent.end_line)
    assert "LFS body text" not in parent_text


def test_subsection_levels_are_deeper_than_parent() -> None:
    lines = _SAMPLE.split("\n")
    sections = find_sections(lines)
    by_title = {s.title: s for s in sections}
    assert by_title["Doc Title Here"].level == 1
    assert by_title["Log-structured File System (LFS)"].level == 2
    assert by_title["Wandering Tree Problem"].level == 2


def test_build_doc_chunks_produces_byte_exact_verifiable_chunks(tmp_path: Path) -> None:
    docs_dir = tmp_path / "Documentation" / "filesystems"
    docs_dir.mkdir(parents=True)
    (docs_dir / "f2fs.rst").write_text(_SAMPLE)

    chunks = build_doc_chunks(tmp_path, "vtest")
    assert chunks
    assert all(c.kind == "doc" for c in chunks)
    assert all(c.file_path == "Documentation/filesystems/f2fs.rst" for c in chunks)

    lfs_chunk = next(c for c in chunks if c.symbol_name == "Log-structured File System (LFS)")
    on_disk_lines = read_lines(docs_dir / "f2fs.rst")
    expected = slice_lines(on_disk_lines, lfs_chunk.start_line, lfs_chunk.end_line)
    assert lfs_chunk.text == expected  # verification re-reads the same way -- must match exactly


def test_chunk_ids_are_unique_across_sections(tmp_path: Path) -> None:
    docs_dir = tmp_path / "Documentation"
    docs_dir.mkdir()
    (docs_dir / "f2fs.rst").write_text(_SAMPLE)

    chunks = build_doc_chunks(tmp_path, "vtest")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
