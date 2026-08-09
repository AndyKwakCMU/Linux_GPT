"""Retrieval-time expansion graph: given a set of already-retrieved chunks,
find which additional macro/type/struct/function definitions they
reference (by scanning identifiers in their text against the chunk symbol
table) so those definitions get pulled into context too -- this is what
lets "what does container_of do here" resolve to the *pinned* definition
instead of the model reciting one from training data, per the plan's
GNU-C-dialect design.

Single-hop only: expansion resolves names referenced directly in a seed
chunk's text, not names referenced by the newly-added chunks in turn --
keeps context bounded for a laptop-scale model.
"""

from __future__ import annotations

import re
from collections import defaultdict

from linuxgpt.ingest.chunk_merge import Chunk

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

# C keywords and kernel-ubiquitous tokens that would otherwise pollute
# every expansion if left resolvable.
_STOPWORDS = frozenset(
    """
    if else for while do switch case default break continue return goto
    sizeof struct union enum typedef static const void volatile inline
    extern register auto int char long short unsigned signed float double
    NULL true false __init __exit __user __iomem __rcu __percpu likely
    unlikely BUG BUG_ON WARN_ON WARN_ON_ONCE
    """.split()
)


class SymbolGraph:
    def __init__(self, chunks: list[Chunk]):
        self._by_name: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            if c.symbol_name:
                self._by_name[c.symbol_name].append(c)

    def references_in(self, text: str) -> list[str]:
        """Identifiers referenced in text that resolve to a known chunk,
        in first-occurrence order, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for m in _IDENTIFIER_RE.finditer(text):
            tok = m.group(0)
            if tok in seen or tok in _STOPWORDS or tok not in self._by_name:
                continue
            seen.add(tok)
            out.append(tok)
        return out

    def resolve(self, name: str, prefer_file: str | None = None) -> list[Chunk]:
        """C name resolution is translation-unit scoped for static symbols,
        and even non-static same-name collisions across kernel files are
        common (e.g. multiple `alloc`/`reset` helpers) -- prefer a
        same-file match before falling back to any file."""
        candidates = self._by_name.get(name, [])
        if prefer_file:
            same_file = [c for c in candidates if c.file_path == prefer_file]
            if same_file:
                return same_file
        return candidates

    def expand(self, seed_chunks: list[Chunk], already_have: set[str], max_new: int) -> list[Chunk]:
        """Returns up to max_new additional chunks referenced directly by
        seed_chunks that aren't already covered by already_have (a set of
        chunk_ids -- callers should pre-seed this with the seed chunks'
        own IDs). Mutates already_have as it goes."""
        new: list[Chunk] = []
        for chunk in seed_chunks:
            for name in self.references_in(chunk.text):
                for candidate in self.resolve(name, prefer_file=chunk.file_path):
                    if candidate.chunk_id in already_have:
                        continue
                    new.append(candidate)
                    already_have.add(candidate.chunk_id)
                    if len(new) >= max_new:
                        return new
        return new
