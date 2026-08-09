"""Authoring-time correctness check for hand-written lessons: every
citation embedded in a lesson file must be byte-exact against the real
pinned kernel source, exactly like a generated answer's citations would
be. Reuses verify_answer as-is -- a lesson's raw .rst text is just
another string with fenced ```<lang> file=...lines=... blocks in it, no
special-casing needed. Requires the kernel source to be fetched
(.kernel_src/), same as the live retrieval/generation tests.
"""

from pathlib import Path

import pytest

from linuxgpt.config import settings
from linuxgpt.verify.verifier import verify_answer

LESSONS_DIR = Path(__file__).resolve().parents[1] / "src" / "linuxgpt" / "lessons"

pytestmark = pytest.mark.skipif(
    not settings.kernel_src_dir.is_dir(), reason="kernel source not fetched (.kernel_src/ missing)"
)


def _lesson_files() -> list[Path]:
    return sorted(LESSONS_DIR.glob("*.rst"))


@pytest.mark.parametrize("lesson_path", _lesson_files(), ids=lambda p: p.name)
def test_lesson_citations_are_byte_exact(lesson_path: Path) -> None:
    text = lesson_path.read_text()
    result = verify_answer(text, chunks_were_available=False)
    assert result.cited_failed == [], f"{lesson_path.name}: {result.cited_failed}"
    assert result.uncited_blocks == [], (
        f"{lesson_path.name}: {len(result.uncited_blocks)} fenced block(s) with no valid "
        "file=...lines=... citation header"
    )
    assert result.cited_ok, f"{lesson_path.name}: no citations found at all"


def test_at_least_two_lessons_exist() -> None:
    assert len(_lesson_files()) >= 2
