"""Wraps universal-ctags to build a fast global symbol table over the
ingested kernel source: symbol name -> {kind, file, line, end}. Used by
chunk_merge as (a) a fallback span source when a tree-sitter declaration
overlaps a parse ERROR node, (b) a source of chunks for top-level symbols
tree-sitter's targeted node-kind walk doesn't cover at all (e.g. global
data-structure definitions like F2FS's `f2fs_sops`/`f2fs_fops` tables,
ctags kind "variable" -- frequently what people ask about, and otherwise
invisible to the function/struct/typedef/macro walk in chunk_treesitter),
and (c) the retrieval-time symbol graph for macro/type/struct expansion.

Confirmed against real fs/f2fs/ source: `--output-format=json` requires
universal-ctags (not the BSD ctags Xcode ships) and returns kind names as
long-form strings ("function", "struct", "variable", "macro", ...), one
JSON object per line.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from linuxgpt.config import settings

# ctags kind -> our chunk kind. Kinds not listed here (enumerator, member,
# ...) are field/constant-level detail, not standalone citable chunks.
KIND_MAP = {
    "function": "function",
    "struct": "struct",
    "union": "union",
    "enum": "enum",
    "typedef": "typedef",
    "macro": "macro",
    "variable": "variable",
}


@dataclass(frozen=True)
class CtagsSymbol:
    name: str
    kind: str  # our chunk kind, already mapped via KIND_MAP
    file_path: str  # relative to the scanned root, posix-style
    start_line: int
    end_line: int | None  # None if ctags didn't report an end line


def run_ctags(root: Path) -> list[CtagsSymbol]:
    cmd = [
        settings.ctags_binary,
        "-R",
        "--languages=C",
        "--output-format=json",
        "--fields=+ne",
        "-f",
        "-",
        str(root),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ctags binary '{settings.ctags_binary}' not found. Install universal-ctags "
            "(e.g. `brew install universal-ctags`) -- the macOS-bundled ctags is not "
            "compatible (no --output-format=json support)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ctags failed (exit {exc.returncode}): {exc.stderr}") from exc

    root_resolved = root.resolve()
    symbols: list[CtagsSymbol] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_type") != "tag":
            continue
        our_kind = KIND_MAP.get(rec.get("kind", ""))
        if our_kind is None:
            continue
        # ctags emits "path" relative to the process's cwd (i.e. prefixed
        # with whatever string `root` was passed as on the command line),
        # not relative to `root` itself -- resolve against cwd, then
        # relativize to root_resolved, rather than re-prepending root.
        path = Path(rec["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            rel_path = path.resolve().relative_to(root_resolved).as_posix()
        except ValueError:
            continue
        symbols.append(
            CtagsSymbol(
                name=rec["name"],
                kind=our_kind,
                file_path=rel_path,
                start_line=rec["line"],
                end_line=rec.get("end"),
            )
        )
    return symbols
