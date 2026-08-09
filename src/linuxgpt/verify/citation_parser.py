"""Parses fenced code blocks out of a rendered answer and classifies each
one as a cited claim (fence info string matches our exact
`<lang> file=... lines=...` convention, for any language tag -- code
chunks render as `c`, doc chunks as `rst`) or an uncited block (anything
else -- a plain ```c/```rst with no location, or a hand-typed pseudo-
citation). A live Phase-1 test showed the base model will confidently
fabricate plausible-looking, entirely fictional code in a bare ```c fence
when it doesn't follow the [[CHUNK:n]] convention -- so "uncited" is
deliberately the default for anything that isn't in our exact format,
not just for blocks with zero location info at all. Accepting any
language tag here doesn't weaken that: `file=...lines=...` is still
mandatory for a block to count as cited.

Fence length is variable (3+ backticks, open/close must match via the
backreference below) rather than hardcoded triple-backtick, per
CommonMark's own nested-fence convention -- confirmed necessary live: a
lesson chunk can itself contain an embedded ```-fenced citation (a
lesson section quoting real doc content), and when that whole chunk gets
re-quoted via [[CHUNK:n]] (generate/chunk_ref.render_citation, which
picks a longer outer fence for exactly this case), a fixed-length parser
would incorrectly close on the inner fence and silently truncate the
outer citation's quoted text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"(?P<fence>`{3,})(?P<info>[^\n]*)\n(?P<body>.*?)\n(?P=fence)(?!`)", re.DOTALL)
_CITED_INFO_RE = re.compile(
    r"^(?P<lang>\w+)\s+file=(?P<file>\S+)\s+lines=(?P<start>\d+)-(?P<end>\d+)"
    r"(?:\s+status=(?P<status>\w+))?\s*$"
)


@dataclass(frozen=True)
class CitationClaim:
    file_path: str
    start_line: int
    end_line: int
    quoted_text: str
    lang: str = "c"  # the fence language tag, e.g. "c" or "rst" -- see generate/chunk_ref.fence_lang_for_kind
    # Set only for lesson-sourced claims (see Chunk.review_status). Purely
    # an additive trust signal for display -- byte-verification (below)
    # doesn't change based on this; an unreviewed lesson can still quote
    # its cited source exactly.
    review_status: str | None = None


def parse_fenced_blocks(rendered_answer: str) -> tuple[list[CitationClaim], list[str]]:
    cited: list[CitationClaim] = []
    uncited: list[str] = []
    for m in _FENCE_RE.finditer(rendered_answer):
        info = m.group("info").strip()
        body = m.group("body")
        cm = _CITED_INFO_RE.match(info)
        if cm:
            cited.append(
                CitationClaim(
                    file_path=cm.group("file"),
                    start_line=int(cm.group("start")),
                    end_line=int(cm.group("end")),
                    quoted_text=body,
                    lang=cm.group("lang"),
                    review_status=cm.group("status"),
                )
            )
        else:
            uncited.append(body)
    return cited, uncited
