"""Tree-sitter based extraction of top-level C declarations (functions,
struct/union/enum specifiers, typedefs, object-like and function-like
macros) with exact line ranges captured directly from the parse tree.

Validated against real fs/f2fs/super.c: kernel constructs like `#ifdef`
splitting a function body, `__init`/`__exit` markers before a return type,
and bare macro-call statements (module_init(), MODULE_AUTHOR(), ...)
reliably produce ERROR/MISSING nodes *inside* an otherwise-correctly-spanned
function_definition. We flag any span whose subtree contains such a node
as low_confidence_parse rather than trusting it silently -- see
chunk_merge.reconcile() for the ctags-based fallback used when a span is
missing outright (e.g. swallowed into a broader ERROR region).
"""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_c as tsc
from tree_sitter import Language, Node, Parser

_LANGUAGE = Language(tsc.language())
_PARSER = Parser(_LANGUAGE)

_DECL_NODE_KINDS = {
    "function_definition": "function",
    "struct_specifier": "struct",
    "union_specifier": "union",
    "enum_specifier": "enum",
    "type_definition": "typedef",
    "preproc_function_def": "macro",
    "preproc_def": "macro",
}

_NAME_NODE_TYPES = {"identifier", "field_identifier", "type_identifier"}


@dataclass(frozen=True)
class RawSpan:
    kind: str
    symbol_name: str | None
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive
    has_error: bool


def _node_text(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _find_first_identifier(node: Node) -> Node | None:
    if node.type in _NAME_NODE_TYPES:
        return node
    for child in node.children:
        found = _find_first_identifier(child)
        if found is not None:
            return found
    return None


def _find_name(node: Node) -> str | None:
    name_field = node.child_by_field_name("name")
    if name_field is not None:
        return _node_text(name_field)
    declarator_field = node.child_by_field_name("declarator")
    if declarator_field is not None:
        ident = _find_first_identifier(declarator_field)
        if ident is not None:
            return _node_text(ident)
    # object/function-like macros: no name/declarator field, but the
    # identifier right after '#define' is the first identifier in the node.
    ident = _find_first_identifier(node)
    return _node_text(ident) if ident is not None else None


def _subtree_has_error(node: Node) -> bool:
    if node.type == "ERROR" or node.is_missing:
        return True
    for child in node.children:
        if _subtree_has_error(child):
            return True
    return False


def extract_spans(source: bytes) -> list[RawSpan]:
    """Walks the parse tree top-down. Whenever a node matches one of the
    target declaration kinds, records it and does NOT descend further --
    this both avoids capturing nested declarations (e.g. a local struct
    inside a function body) as separate top-level chunks, and naturally
    treats `typedef struct foo {...} foo_t;` as a single typedef chunk
    rather than also emitting its inner struct_specifier."""
    tree = _PARSER.parse(source)
    spans: list[RawSpan] = []

    def walk(node: Node) -> None:
        kind = _DECL_NODE_KINDS.get(node.type)
        if kind in ("struct", "union", "enum") and node.child_by_field_name("body") is None:
            # Bare forward reference / usage (e.g. `struct inode *foo`, or
            # `struct bar;`) rather than a definition -- nothing to chunk,
            # and nothing further to walk inside it.
            return
        if node.type == "preproc_def" and not any(c.type == "preproc_arg" for c in node.children):
            # A value-less #define (no preproc_arg child) is almost always
            # a header include-guard or feature-test marker, e.g.
            # `#define _LINUX_F2FS_H` -- confirmed live to be actively
            # harmful to index: a general-purpose embedding model ranks
            # its bare filename-derived identifier as *more* similar to
            # "What is F2FS?" than the real f2fs.rst overview, purely on
            # lexical surface overlap, burying the actually-useful chunk.
            # Zero citable content either way, so skip it entirely.
            return
        if kind is not None:
            spans.append(
                RawSpan(
                    kind=kind,
                    symbol_name=_find_name(node),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    has_error=_subtree_has_error(node),
                )
            )
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return spans
