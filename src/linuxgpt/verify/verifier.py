"""Byte-level citation verification (Layer 2 safety net): every cited
fenced code block is re-checked against the actual pinned source tree,
using the exact same line-reading helper (linuxgpt.util) used at ingest
time to build chunk.text, so the two can never drift out of sync. Any
fenced code block with no valid citation header is an automatic failure
-- see citation_parser for why that's the safe default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from linuxgpt.config import settings
from linuxgpt.generate.prompts import ABSTAIN_MARKER
from linuxgpt.util import read_lines, slice_lines
from linuxgpt.verify.citation_parser import CitationClaim, parse_fenced_blocks


@dataclass
class VerificationResult:
    cited_ok: list[CitationClaim]
    cited_failed: list[tuple[CitationClaim, str]]  # (claim, reason)
    uncited_blocks: list[str]
    # True when chunks were retrieved but the answer cites none of them at
    # all -- a live Phase-1 test showed the model will happily write
    # confident, fabricated prose claims (e.g. an invented struct field
    # name) with zero [[CHUNK:n]] references and no code fences, which
    # would otherwise show as "nothing to verify, so it passed". That is
    # not trustworthy, so it's treated as a failure like any other.
    zero_citation_attempt: bool = False

    @property
    def all_passed(self) -> bool:
        return not self.cited_failed and not self.uncited_blocks and not self.zero_citation_attempt


def _normalize(text: str) -> str:
    # Only trailing-whitespace/newline normalization is allowed -- kernel
    # code is tab-sensitive, so no whitespace collapsing beyond that.
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def verify_claim(claim: CitationClaim, kernel_src_dir: Path) -> tuple[bool, str]:
    root = kernel_src_dir.resolve()
    path = (root / claim.file_path).resolve()
    if root != path and root not in path.parents:
        return False, "file path escapes the indexed source tree"
    if not path.is_file():
        return False, f"{claim.file_path} not found in indexed source"
    lines = read_lines(path)
    if claim.start_line < 1 or claim.end_line > len(lines) or claim.end_line < claim.start_line:
        return False, f"line range {claim.start_line}-{claim.end_line} out of bounds"
    actual = slice_lines(lines, claim.start_line, claim.end_line)
    if _normalize(actual) != _normalize(claim.quoted_text):
        return False, "quoted text does not byte-match the source at that location"
    return True, ""


def verify_answer(rendered_answer: str, *, chunks_were_available: bool = False) -> VerificationResult:
    if rendered_answer.strip().startswith(ABSTAIN_MARKER):
        # An honest "the retrieved chunks don't answer this" is not a
        # citation failure -- retrieval always returns its top-k even for
        # an out-of-scope question, so "chunks were available" alone can't
        # be used to demand a citation here.
        return VerificationResult(cited_ok=[], cited_failed=[], uncited_blocks=[])
    cited, uncited = parse_fenced_blocks(rendered_answer)
    ok_claims: list[CitationClaim] = []
    failed_claims: list[tuple[CitationClaim, str]] = []
    for claim in cited:
        passed, reason = verify_claim(claim, settings.kernel_src_dir)
        if passed:
            ok_claims.append(claim)
        else:
            failed_claims.append((claim, reason))
    zero_citation_attempt = chunks_were_available and not (ok_claims or failed_claims or uncited)
    return VerificationResult(
        cited_ok=ok_claims,
        cited_failed=failed_claims,
        uncited_blocks=uncited,
        zero_citation_attempt=zero_citation_attempt,
    )
