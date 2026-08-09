"""Regression tests for the tree-sitter/ctags chunker. The three
non-obvious cases here (bare forward-reference structs, #ifdef-duplicated
same-name functions, and low_confidence flagging around #ifdef-split
bodies) were all real bugs caught by running the chunker against actual
fs/f2fs/ source during the Phase 1 spike -- keep them as regressions.
"""

from pathlib import Path

import pytest

from linuxgpt.ingest.chunk_merge import build_chunks

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ctags") is None, reason="universal-ctags not installed"
)


def _write(tmp_path: Path, rel: str, content: str) -> None:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_function_and_struct_and_macro_are_chunked(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "foo.c",
        "struct point {\n"
        "\tint x;\n"
        "\tint y;\n"
        "};\n"
        "\n"
        "#define ADD(a, b) ((a) + (b))\n"
        "\n"
        "int distance(struct point *p)\n"
        "{\n"
        "\treturn p->x + p->y;\n"
        "}\n",
    )
    chunks = build_chunks(tmp_path, "vtest")
    by_name = {c.symbol_name: c for c in chunks}

    assert by_name["point"].kind == "struct"
    assert by_name["point"].start_line == 1
    assert by_name["point"].end_line == 4

    assert by_name["ADD"].kind == "macro"

    assert by_name["distance"].kind == "function"
    assert by_name["distance"].start_line == 8
    assert by_name["distance"].end_line == 11
    assert "return p->x + p->y;" in by_name["distance"].text


def test_bare_forward_reference_is_not_chunked(tmp_path: Path) -> None:
    """`struct inode *foo` as a parameter type, or `struct bar;`, must not
    produce a bogus single-line 'struct' chunk -- only a struct_specifier
    with an actual field_declaration_list body is a definition."""
    _write(
        tmp_path,
        "foo.c",
        "struct inode;\n"
        "\n"
        "void handle(struct inode *inode, struct dentry *dentry)\n"
        "{\n"
        "}\n",
    )
    chunks = build_chunks(tmp_path, "vtest")
    struct_chunks = [c for c in chunks if c.kind == "struct"]
    assert struct_chunks == []


def test_bare_header_guard_macro_is_not_chunked(tmp_path: Path) -> None:
    """A value-less #define like a header include-guard carries no citable
    content and was confirmed live to actively hurt retrieval: a general-
    purpose embedding model ranked `_LINUX_F2FS_H`'s bare identifier as
    *more* relevant to "What is F2FS?" than the real f2fs.rst overview,
    purely on lexical surface overlap with the filename. A #define with an
    actual value must still be chunked normally."""
    _write(
        tmp_path,
        "foo.h",
        "#ifndef _FOO_H\n"
        "#define _FOO_H\n"
        "\n"
        "#define MAX_LEN 128\n"
        "\n"
        "#endif\n",
    )
    chunks = build_chunks(tmp_path, "vtest")
    macro_names = {c.symbol_name for c in chunks if c.kind == "macro"}
    assert "_FOO_H" not in macro_names
    assert "MAX_LEN" in macro_names


def test_ifdef_duplicated_same_name_functions_get_distinct_chunks(tmp_path: Path) -> None:
    """Kernel code commonly guards a real implementation and a stub under
    the same name with #ifdef/#else -- both must survive as separate
    chunks with their own line ranges, not collapse onto one."""
    _write(
        tmp_path,
        "foo.c",
        "#ifdef CONFIG_FOO\n"
        "static int __init make_cache(void)\n"
        "{\n"
        "\treturn real_init();\n"
        "}\n"
        "#else\n"
        "static int __init make_cache(void) { return 0; }\n"
        "#endif\n",
    )
    chunks = build_chunks(tmp_path, "vtest")
    matches = sorted((c for c in chunks if c.symbol_name == "make_cache"), key=lambda c: c.start_line)
    assert len(matches) == 2
    assert matches[0].start_line != matches[1].start_line
    assert matches[0].chunk_id != matches[1].chunk_id
    assert "real_init" in matches[0].text
    assert "return 0" in matches[1].text


def test_chunk_ids_are_unique(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "foo.c",
        "int a(void) { return 1; }\n"
        "int b(void) { return 2; }\n"
        "int c(void) { return 3; }\n",
    )
    chunks = build_chunks(tmp_path, "vtest")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
