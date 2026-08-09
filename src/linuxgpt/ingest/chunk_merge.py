"""Reconciles tree-sitter spans with the ctags symbol table into the final
list of Chunks, per the design in the approved plan: tree-sitter is
authoritative for span boundaries when it parsed cleanly; when a span
overlaps a parse ERROR/MISSING node, fall back to the matching ctags
symbol's line (or a brace-counting heuristic if ctags didn't report an end
line), and tag the chunk low_confidence_parse rather than trusting or
dropping it silently. Symbols ctags found that tree-sitter's targeted walk
never produces at all (notably ctags kind "variable" -- global data-
structure definitions like F2FS's `f2fs_sops`/`f2fs_fops` tables) are
added as ctags-only chunks so nothing indexable goes missing.

chunk.text is always the exact source lines [start_line, end_line] read
straight from the file via linuxgpt.util -- never a tree-sitter byte slice
-- so citation verification later is a trivial re-read-and-compare.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from linuxgpt.ingest.chunk_ctags import CtagsSymbol, run_ctags
from linuxgpt.ingest.chunk_treesitter import RawSpan, extract_spans
from linuxgpt.util import read_lines, slice_lines

SOURCE_SUFFIXES = (".c", ".h")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    kernel_version: str
    file_path: str  # posix path relative to the kernel_src root
    kind: str  # function | struct | union | enum | typedef | macro | variable | doc | lesson
    symbol_name: str | None
    start_line: int
    end_line: int
    text: str
    low_confidence_parse: bool
    provenance: str  # treesitter | treesitter_ctags_fallback | ctags_only | rst-section
    # None for real code/doc chunks (byte-identical to the pinned kernel,
    # no "review" concept applies). Lesson chunks (original synthesis I
    # authored, not verbatim kernel content) carry "unreviewed" until
    # Jaeguek approves them -- see lessons/manifest.yaml.
    review_status: str | None = None


def _brace_count_end_line(lines: list[str], start_line: int) -> int:
    """Starting at 1-indexed start_line, scans forward tracking brace depth
    (skipping string/char literals and // and /* */ comments) for the line
    where the first opened brace closes. Falls back to start_line itself
    (a single-line span) if no balanced brace is found before EOF."""
    depth = 0
    seen_open = False
    in_block_comment = False
    in_string = False
    in_char = False
    i = start_line - 1
    n = len(lines)
    while i < n:
        line = lines[i]
        j = 0
        length = len(line)
        while j < length:
            c = line[j]
            nxt = line[j + 1] if j + 1 < length else ""
            if in_block_comment:
                if c == "*" and nxt == "/":
                    in_block_comment = False
                    j += 2
                else:
                    j += 1
                continue
            if in_string:
                if c == "\\":
                    j += 2
                elif c == '"':
                    in_string = False
                    j += 1
                else:
                    j += 1
                continue
            if in_char:
                if c == "\\":
                    j += 2
                elif c == "'":
                    in_char = False
                    j += 1
                else:
                    j += 1
                continue
            if c == "/" and nxt == "*":
                in_block_comment = True
                j += 2
                continue
            if c == "/" and nxt == "/":
                break  # rest of line is a line comment
            if c == '"':
                in_string = True
                j += 1
                continue
            if c == "'":
                in_char = True
                j += 1
                continue
            if c == "{":
                depth += 1
                seen_open = True
                j += 1
                continue
            if c == "}":
                depth -= 1
                j += 1
                if seen_open and depth <= 0:
                    return i + 1  # 1-indexed
                continue
            j += 1
        i += 1
    return start_line


def _find_ctags_fallback(
    ctags_by_name: dict[str, list[CtagsSymbol]],
    symbol_name: str | None,
    kind: str,
    near_line: int,
) -> CtagsSymbol | None:
    """Picks the ctags candidate closest to near_line (the tree-sitter
    span's own start line), preferring matching kind. This matters for
    #ifdef/#else-guarded duplicate-named definitions (e.g. a real
    implementation vs. a stub under the same name, common in F2FS's
    CONFIG_UNICODE/CONFIG_QUOTA-gated code) -- ctags reports both, and
    proximity is the only signal that tells the two tree-sitter spans
    apart."""
    if symbol_name is None:
        return None
    candidates = ctags_by_name.get(symbol_name)
    if not candidates:
        return None
    same_kind = [c for c in candidates if c.kind == kind]
    pool = same_kind or candidates
    return min(pool, key=lambda c: abs(c.start_line - near_line))


def _chunk_id(kernel_version: str, file_path: str, kind: str, symbol_name: str | None, start: int, end: int) -> str:
    key = f"{kernel_version}|{file_path}|{kind}|{symbol_name}|{start}|{end}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    kernel_version: str,
    file_path: str,
    kind: str,
    symbol_name: str | None,
    start_line: int,
    end_line: int,
    lines: list[str],
    low_confidence: bool,
    provenance: str,
) -> Chunk:
    end_line = max(start_line, end_line)
    return Chunk(
        chunk_id=_chunk_id(kernel_version, file_path, kind, symbol_name, start_line, end_line),
        kernel_version=kernel_version,
        file_path=file_path,
        kind=kind,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        text=slice_lines(lines, start_line, end_line),
        low_confidence_parse=low_confidence,
        provenance=provenance,
    )


def _process_file(
    source_root: Path,
    path: Path,
    kernel_version: str,
    ctags_by_file: dict[str, list[CtagsSymbol]],
) -> list[Chunk]:
    rel_path = path.relative_to(source_root).as_posix()
    source_bytes = path.read_bytes()
    lines = read_lines(path)
    spans: list[RawSpan] = extract_spans(source_bytes)

    file_ctags = ctags_by_file.get(rel_path, [])
    ctags_by_name: dict[str, list[CtagsSymbol]] = defaultdict(list)
    for sym in file_ctags:
        ctags_by_name[sym.name].append(sym)

    chunks: list[Chunk] = []
    captured_keys: set[tuple[str, str | None]] = set()

    for span in spans:
        start_line, end_line = span.start_line, span.end_line
        low_confidence = span.has_error
        provenance = "treesitter"
        if span.has_error:
            fallback = _find_ctags_fallback(ctags_by_name, span.symbol_name, span.kind, span.start_line)
            if fallback is not None:
                start_line = fallback.start_line
                end_line = fallback.end_line or _brace_count_end_line(lines, fallback.start_line)
                provenance = "treesitter_ctags_fallback"
        chunks.append(
            _make_chunk(
                kernel_version, rel_path, span.kind, span.symbol_name,
                start_line, end_line, lines, low_confidence, provenance,
            )
        )
        captured_keys.add((span.kind, span.symbol_name))

    for sym in file_ctags:
        key = (sym.kind, sym.name)
        if key in captured_keys:
            continue
        end_line = sym.end_line or _brace_count_end_line(lines, sym.start_line)
        chunks.append(
            _make_chunk(
                kernel_version, rel_path, sym.kind, sym.name,
                sym.start_line, end_line, lines, True, "ctags_only",
            )
        )
        captured_keys.add(key)

    return chunks


def build_chunks(source_root: Path, kernel_version: str) -> list[Chunk]:
    ctags_symbols = run_ctags(source_root)
    ctags_by_file: dict[str, list[CtagsSymbol]] = defaultdict(list)
    for sym in ctags_symbols:
        ctags_by_file[sym.file_path].append(sym)

    source_files = sorted(
        p for p in source_root.rglob("*") if p.is_file() and p.suffix in SOURCE_SUFFIXES
    )

    chunks: list[Chunk] = []
    for path in source_files:
        chunks.extend(_process_file(source_root, path, kernel_version, ctags_by_file))
    return chunks
