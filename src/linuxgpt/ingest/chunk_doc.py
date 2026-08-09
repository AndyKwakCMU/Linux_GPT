"""Splits an .rst documentation file into section-level chunks using
RST's title+underline convention, for the same byte-verifiable citation
pipeline used for C source: exact 1-indexed line ranges captured
directly, chunk.text always re-read from the file (via chunk_merge's
_make_chunk / linuxgpt.util), so verification is a trivial slice-and-
compare identical to code chunks -- only the chunk `kind` ("doc") differs.

An RST heading is a line of title text immediately followed by a line of
repeated punctuation from the RST adornment character set, at least as
long as the title (an optional matching overline above the title, as
used for a document's top-level title, is permitted but not required).
Nesting level is inferred from first-seen-order of distinct adornment
characters within the file, matching Sphinx's convention that level is
positional/consistent-per-document rather than a fixed global mapping of
character to level.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from linuxgpt.ingest.chunk_merge import Chunk
from linuxgpt.util import read_lines, slice_lines

_ADORNMENT_LINE_RE = re.compile(r'^([=\-~^"\'#*+.:_`])\1{2,}$')


@dataclass(frozen=True)
class DocSection:
    title: str
    level: int
    start_line: int  # 1-indexed, the title line itself
    end_line: int  # 1-indexed inclusive, last line before the next heading (or EOF)


def find_sections(lines: list[str]) -> list[DocSection]:
    """lines: 0-indexed list as returned by linuxgpt.util.read_lines, i.e.
    lines[i] is the text of 1-indexed line i+1."""
    n = len(lines)
    # (title_line_1indexed, adornment_char, title_text)
    headings: list[tuple[int, str, str]] = []

    in_fence = False
    for p in range(0, n):
        # A lesson can quote other RST content verbatim inside a fenced
        # ```block (e.g. citing a real doc section that itself contains a
        # heading) -- text inside a fence is literal, not structural, so
        # heading-adornment lines must be ignored while inside one.
        # Confirmed necessary: without this, an embedded citation's own
        # heading was misread as a heading of the outer lesson document,
        # corrupting section boundaries.
        if lines[p].startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or p == 0:
            continue
        m = _ADORNMENT_LINE_RE.match(lines[p].strip())
        if not m:
            continue
        title_candidate = lines[p - 1].strip()
        if not title_candidate:
            continue  # bare adornment line with nothing above it (e.g. an overline)
        underline_len = len(lines[p].strip())
        if underline_len < len(title_candidate):
            continue  # underline shorter than its title isn't a valid RST heading
        title_line_1indexed = p  # 0-indexed p-1 -> 1-indexed p
        headings.append((title_line_1indexed, m.group(1), title_candidate))

    level_order: list[str] = []
    for _, char, _ in headings:
        if char not in level_order:
            level_order.append(char)

    sections: list[DocSection] = []
    for idx, (title_line, char, title) in enumerate(headings):
        end_line = n if idx + 1 == len(headings) else headings[idx + 1][0] - 2
        end_line = max(end_line, title_line)
        sections.append(
            DocSection(title=title, level=level_order.index(char) + 1, start_line=title_line, end_line=end_line)
        )
    return sections


def build_doc_chunks(
    source_root: Path,
    kernel_version: str,
    *,
    kind: str = "doc",
    scan_subdir: str | None = None,
    review_status_by_file: dict[str, str] | None = None,
) -> list[Chunk]:
    """Mirrors chunk_merge.build_chunks' signature/root convention exactly
    (same source_root as the C chunker -- Documentation/ lives under the
    same pinned .kernel_src/ tree) so citation verification needs no
    changes: file_path is always relative to source_root, chunk.text is
    still a direct re-read of source lines.

    scan_subdir restricts which files get walked (e.g. "lessons") while
    file_path stays relative to the full source_root -- so a lesson's
    citation still resolves against the one pinned root everything else
    uses, it's just nested under a subdirectory of it.

    review_status_by_file (keyed by filename, e.g. "foo.rst") is used for
    kind="lesson": None for real code/doc content (the concept doesn't
    apply), a status string for hand-authored lesson files pending review.
    """
    review_status_by_file = review_status_by_file or {}
    scan_root = (source_root / scan_subdir) if scan_subdir else source_root
    chunks: list[Chunk] = []
    for path in sorted(scan_root.rglob("*.rst")):
        rel_path = path.relative_to(source_root).as_posix()
        review_status = review_status_by_file.get(path.name)
        lines = read_lines(path)
        for section in find_sections(lines):
            text = slice_lines(lines, section.start_line, section.end_line)
            chunk_id = hashlib.sha1(
                f"{kernel_version}|{rel_path}|{kind}|{section.title}|{section.start_line}|{section.end_line}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    kernel_version=kernel_version,
                    file_path=rel_path,
                    kind=kind,
                    symbol_name=section.title,
                    start_line=section.start_line,
                    end_line=section.end_line,
                    text=text,
                    low_confidence_parse=False,
                    provenance="rst-section",
                    review_status=review_status,
                )
            )
    return chunks
