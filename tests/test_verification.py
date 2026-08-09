"""Tests for the citation-integrity layer: fence parsing, byte-level
verification against a real file on disk, and the chunk-ref substitution
that guarantees Layer-1 grounding by construction."""

from pathlib import Path

from linuxgpt.generate.chunk_ref import render_citation, substitute_chunk_refs
from linuxgpt.ingest.chunk_merge import Chunk
from linuxgpt.verify.citation_parser import parse_fenced_blocks
from linuxgpt.verify.verifier import verify_answer, verify_claim


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="abc123",
        kernel_version="vtest",
        file_path="foo.c",
        kind="function",
        symbol_name="foo",
        start_line=1,
        end_line=3,
        text="int foo(void)\n{\n\treturn 1;\n}",
        low_confidence_parse=False,
        provenance="treesitter",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_parse_fenced_blocks_distinguishes_cited_from_uncited() -> None:
    answer = (
        "Here is grounded code:\n"
        "```c file=fs/f2fs/foo.c lines=10-12\n"
        "int foo(void) { return 1; }\n"
        "```\n"
        "And here is fabricated code with no citation:\n"
        "```c\n"
        "int made_up_helper(void) { return 42; }\n"
        "```\n"
    )
    cited, uncited = parse_fenced_blocks(answer)
    assert len(cited) == 1
    assert cited[0].file_path == "fs/f2fs/foo.c"
    assert cited[0].start_line == 10
    assert cited[0].end_line == 12
    assert len(uncited) == 1
    assert "made_up_helper" in uncited[0]


def test_verify_claim_passes_for_byte_exact_match(tmp_path: Path) -> None:
    src = tmp_path / "fs" / "f2fs"
    src.mkdir(parents=True)
    f = src / "foo.c"
    f.write_text("line1\nline2\nline3\nline4\n")

    from linuxgpt.verify.citation_parser import CitationClaim

    claim = CitationClaim(file_path="fs/f2fs/foo.c", start_line=2, end_line=3, quoted_text="line2\nline3")
    ok, reason = verify_claim(claim, tmp_path)
    assert ok, reason


def test_verify_claim_fails_for_mismatched_text(tmp_path: Path) -> None:
    src = tmp_path / "fs" / "f2fs"
    src.mkdir(parents=True)
    (src / "foo.c").write_text("line1\nline2\nline3\n")

    from linuxgpt.verify.citation_parser import CitationClaim

    claim = CitationClaim(file_path="fs/f2fs/foo.c", start_line=1, end_line=2, quoted_text="line1\nFABRICATED")
    ok, reason = verify_claim(claim, tmp_path)
    assert not ok
    assert "does not byte-match" in reason


def test_verify_claim_rejects_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "fs").mkdir()
    (tmp_path / "fs" / "f2fs.c").write_text("x\n")
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("top secret\n")

    from linuxgpt.verify.citation_parser import CitationClaim

    claim = CitationClaim(file_path="../secret.txt", start_line=1, end_line=1, quoted_text="top secret")
    ok, reason = verify_claim(claim, tmp_path)
    assert not ok
    assert "escapes" in reason


def test_verify_claim_rejects_out_of_bounds_lines(tmp_path: Path) -> None:
    src = tmp_path / "fs"
    src.mkdir()
    (src / "foo.c").write_text("line1\nline2\n")

    from linuxgpt.verify.citation_parser import CitationClaim

    claim = CitationClaim(file_path="fs/foo.c", start_line=1, end_line=100, quoted_text="whatever")
    ok, reason = verify_claim(claim, tmp_path)
    assert not ok
    assert "out of bounds" in reason


def test_verify_answer_all_passed_when_no_code_blocks_and_no_chunks_available() -> None:
    result = verify_answer("This question isn't covered by the indexed subsystems.")
    assert result.all_passed


def test_verify_answer_fails_zero_citation_prose_when_chunks_were_available() -> None:
    """Regression: a live Phase-1 test showed the model can write confident,
    uncited (and in that case fabricated) prose claims with no code fence
    at all -- that must fail, not silently pass because there was nothing
    to check."""
    result = verify_answer(
        "The cp_control struct has a sync_mode field.", chunks_were_available=True
    )
    assert not result.all_passed
    assert result.zero_citation_attempt


def test_verify_answer_abstain_marker_is_exempt_from_zero_citation_failure() -> None:
    """Retrieval always returns its top-k even for an out-of-scope question,
    so an honest "not indexed" abstention must not be penalized just
    because chunks happened to be retrieved."""
    result = verify_answer(
        "NOT INDEXED: the retrieved chunks don't cover this.", chunks_were_available=True
    )
    assert result.all_passed
    assert not result.zero_citation_attempt


def test_substitute_chunk_refs_is_byte_perfect_by_construction() -> None:
    chunk = _chunk()
    answer = "The function returns 1 [[CHUNK:1]]."
    rendered, unresolved = substitute_chunk_refs(answer, [chunk])
    assert unresolved == []
    assert render_citation(chunk) in rendered

    cited, uncited = parse_fenced_blocks(rendered)
    assert len(cited) == 1
    assert uncited == []
    result = verify_answer(rendered.replace("foo.c", "foo.c"))  # sanity: still parseable
    # verify against a real tmp file mirroring the chunk to confirm round-trip
    assert cited[0].quoted_text == chunk.text


def test_substitute_chunk_refs_flags_unresolved_indices() -> None:
    chunk = _chunk()
    answer = "See [[CHUNK:1]] and also [[CHUNK:99]]."
    rendered, unresolved = substitute_chunk_refs(answer, [chunk])
    assert unresolved == [99]
    assert "[[CHUNK:99]]" in rendered  # left visible, not silently dropped


def test_render_citation_survives_a_chunk_containing_an_embedded_fence() -> None:
    """Regression: a lesson chunk can itself contain an embedded
    ```-fenced citation (a lesson section quoting real doc content). Live
    testing showed that re-quoting such a chunk via [[CHUNK:n]] with a
    fixed triple-backtick outer fence silently truncated the citation at
    the inner fence's opening line. render_citation must pick a longer
    outer fence so the whole chunk round-trips intact through
    parse_fenced_blocks."""
    chunk = _chunk(
        kind="lesson",
        text=(
            "Some lesson prose.\n\n"
            "```rst file=Documentation/foo.rst lines=1-2\n"
            "quoted doc content\n"
            "```\n\n"
            "More lesson prose after the embedded citation."
        ),
    )
    rendered = render_citation(chunk)
    cited, uncited = parse_fenced_blocks(rendered)
    assert uncited == []
    assert len(cited) == 1
    assert cited[0].quoted_text == chunk.text  # not truncated at the inner fence


def test_substitute_chunk_refs_handles_malformed_multi_ref_cluster() -> None:
    """Regression: live testing showed the model sometimes writes
    "[[CHUNK:3], [CHUNK:5]]" instead of two separate [[CHUNK:3]] [[CHUNK:5]]
    tags -- a reasonable multi-citation attempt that the original strict
    regex didn't recognize at all (silently leaving both chunks uncited).
    Both chunks must resolve, byte-perfect, from this malformed-but-clear
    cluster."""
    chunk1 = _chunk(chunk_id="a", symbol_name="foo", start_line=1, end_line=3)
    chunk2 = _chunk(chunk_id="b", symbol_name="bar", start_line=10, end_line=12, text="int bar(void)\n{\n\treturn 2;\n}")
    chunks = [_chunk(chunk_id=str(i)) for i in range(1, 6)]
    chunks[2], chunks[4] = chunk1, chunk2  # 1-indexed CHUNK:3 and CHUNK:5

    answer = "Because of X and Y [[CHUNK:3], [CHUNK:5]]."
    rendered, unresolved = substitute_chunk_refs(answer, chunks)
    assert unresolved == []
    assert render_citation(chunk1) in rendered
    assert render_citation(chunk2) in rendered
