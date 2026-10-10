"""Builds one piece's training corpus from the pinned kernel tarball:
data/<piece>/corpus.jsonl, one record per top-level unit, with its source
provenance and its train/val assignment.

    python build_corpus.py --piece f2fs

Why units instead of whole files: F2FS is ~30 files, so a by-file 95/5
split leaves val as a single random file -- far too noisy to read a loss
curve from, and short enough to crash get_batch. A top-level unit is one
C definition (function, struct, inline helper) with whatever precedes it
(its comment block, #defines, #includes), or one RST section for docs.
Kernel style closes every top-level definition with `}` at column 0, so the
C split needs no parser.

Assignment is by hash of (path, unit text), so it's deterministic and
stable: re-running on the same tarball reproduces the same split, and
editing one function only moves that function.

Each record keeps path + start/end line so anything downstream (probes,
the future /embed bridge) can point back at exact pinned source, the same
way the RAG side's citations do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from kernel_tarball import PRETRAIN_ROOT, iter_members, load_piece, member_matches, verified_tarball

CODE_SUFFIXES = {".c", ".h"}
DOC_SUFFIXES = {".rst"}
# an RST underline/overline: one repeated punctuation char, 3+ long
_RST_ADORNMENT = re.compile(r"^([=\-~^\"#*+`:.'_])\1{2,}\s*$")


def split_c_units(lines: list[str]) -> list[tuple[int, int]]:
    """Returns (start, end) 0-based inclusive line ranges. A unit closes on a
    line that starts with `}` at column 0 (end of a top-level function,
    struct, enum or initializer) or `)` at column 0 (end of a multi-line
    top-level macro invocation, e.g. every TRACE_EVENT() in
    include/trace/events/f2fs.h); everything since the previous close --
    comments, macros, includes -- rides along with the next definition."""
    units, start = [], 0
    for i, line in enumerate(lines):
        if line.startswith(("}", ")")):
            units.append((start, i))
            start = i + 1
    if start < len(lines) and any(l.strip() for l in lines[start:]):
        units.append((start, len(lines) - 1))
    return units


def split_rst_units(lines: list[str]) -> list[tuple[int, int]]:
    """Starts a new unit at each section title: a non-blank line followed by
    an adornment line, plus its overline if it has one."""
    starts = [0]
    for i in range(1, len(lines)):
        prev = lines[i - 1]
        if _RST_ADORNMENT.match(lines[i]) and prev.strip() and not _RST_ADORNMENT.match(prev):
            title = i - 2 if i >= 2 and _RST_ADORNMENT.match(lines[i - 2]) else i - 1
            if title > starts[-1]:
                starts.append(title)
    return [(s, (starts[k + 1] - 1) if k + 1 < len(starts) else len(lines) - 1) for k, s in enumerate(starts)]


def assign_split(path: str, text: str, val_permille: int) -> str:
    h = int.from_bytes(hashlib.sha256(f"{path}\0{text}".encode()).digest()[:8], "big")
    return "val" if h % 1000 < val_permille else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--piece", default="f2fs")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--skip-verify", action="store_true", help="don't check the tarball sha256 against kernel.org (offline)")
    args = parser.parse_args()

    piece = load_piece(args.piece)
    tarball, tarball_sha = verified_tarball(piece.tarball_name, piece.checksums_url, skip_verify=args.skip_verify)
    val_permille = round(args.val_fraction * 1000)

    out_dir = PRETRAIN_ROOT / "data" / piece.name
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"

    matched_specs = set()
    n_files = 0
    counts = {"train": 0, "val": 0}
    chars = {"train": 0, "val": 0}
    with corpus_path.open("w") as out:
        for rel, data in iter_members(tarball):
            specs = [s for s in piece.paths if member_matches(rel, s)]
            if not specs:
                continue
            suffix = Path(rel).suffix
            if suffix in CODE_SUFFIXES:
                kind, splitter = "code", split_c_units
            elif suffix in DOC_SUFFIXES:
                kind, splitter = "doc", split_rst_units
            else:
                continue  # Makefile, Kconfig, etc. under fs/f2fs
            matched_specs.update(specs)
            n_files += 1
            lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
            for start, end in splitter(lines):
                text = "".join(lines[start : end + 1])
                if not text.strip():
                    continue
                split = assign_split(rel, text, val_permille)
                counts[split] += 1
                chars[split] += len(text)
                out.write(json.dumps({
                    "path": rel,
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "kind": kind,
                    "split": split,
                    "text": text,
                }) + "\n")

    unmatched = [s.path for s in piece.paths if s not in matched_specs]
    if unmatched:
        raise SystemExit(f"Piece manifest paths matched nothing in {tarball.name}: {unmatched}")

    meta = {
        "piece": piece.name,
        "kernel_version": piece.kernel_version,
        "tarball": piece.tarball_name,
        "tarball_sha256": tarball_sha,
        "val_fraction": args.val_fraction,
        "files": n_files,
        "units": counts,
        "chars": chars,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{piece.name}: {n_files} files -> {counts['train']} train / {counts['val']} val units "
          f"({chars['train']:,} / {chars['val']:,} chars) -> {corpus_path}")


if __name__ == "__main__":
    main()
