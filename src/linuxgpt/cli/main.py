"""CLI entry point: `linuxgpt ingest` builds the index, `linuxgpt ask`
asks a citation-verified question against it. Output always shows a
VERIFIED/NOT FULLY VERIFIED status plus a footer manifest of exactly which
files/lines backed the answer, so failures are visible rather than
papered over -- for an audience testing this tool's integrity, that's the
point."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from linuxgpt.config import settings
from linuxgpt.generate.prompts import build_messages
from linuxgpt.ingest.pipeline import ingest_all
from linuxgpt.retrieve.retriever import retrieve
from linuxgpt.verify.retry import generate_with_verification

app = typer.Typer(add_completion=False, help="Citation-verified Q&A over indexed Linux kernel source.")
console = Console()


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", help="re-fetch and re-index even if already done"),
) -> None:
    """Fetch the pinned kernel source (per kernel_manifest.yaml), chunk it, embed it, and store it."""
    console.print(f"[bold]Ingesting kernel source per {settings.manifest_path}...[/bold]")
    chunks = ingest_all(force_fetch=force)
    console.print(f"[green]Indexed {len(chunks)} chunks.[/green]")


@app.command()
def ask(question: str) -> None:
    """Ask a citation-verified question against the indexed kernel source."""
    console.print("[dim]Retrieving relevant source...[/dim]")
    chunks = retrieve(question)
    messages = build_messages(question, chunks)

    console.print("[dim]Generating answer...[/dim]")
    rendered, result, attempts = generate_with_verification(messages, chunks)

    console.print()
    console.print(Panel(Markdown(rendered), title="Answer", border_style="cyan"))
    console.print()

    if result.all_passed:
        console.print(
            f"[bold green]VERIFIED[/bold green] -- all citations checked against the "
            f"pinned source tree ({attempts} attempt(s))."
        )
    else:
        console.print(f"[bold red]NOT FULLY VERIFIED[/bold red] after {attempts} attempt(s):")
        if result.zero_citation_attempt:
            console.print("  - The answer made claims without citing any [[CHUNK:n]] reference at all.")
        for block in result.uncited_blocks:
            stripped = block.strip()
            preview = stripped.splitlines()[0][:80] if stripped else ""
            # markup=False: this previews model-written (possibly
            # fabricated) code, which can easily contain "[something]"
            # that Rich would otherwise silently swallow as a markup tag
            # (confirmed: "[red]"/"[bold]"-shaped substrings vanish from
            # the displayed text instead of erroring, which would corrupt
            # exactly the fabricated content this is meant to surface).
            console.print(f"  - Uncited code block (not grounded in retrieved source): {preview!r}...", markup=False)
        for claim, reason in result.cited_failed:
            console.print(f"  - {claim.file_path} lines {claim.start_line}-{claim.end_line}: {reason}", markup=False)
        console.print("[yellow]Treat this answer as unverified -- it may contain fabricated details.[/yellow]")

    if result.cited_ok:
        console.print()
        console.print("[bold]Verified citations (footer manifest):[/bold]")
        any_unreviewed = False
        for claim in result.cited_ok:
            # parens, not square brackets -- rich.Console.print interprets
            # [xyz] as markup, so a literal "[doc]" tag would misrender.
            if claim.review_status:
                tag = "(lesson)"
            elif claim.lang == "rst":
                tag = "(doc)  "
            else:
                tag = "(code) "
            console.print(f"  - {tag} {claim.file_path}:{claim.start_line}-{claim.end_line}")
            if claim.review_status == "unreviewed":
                any_unreviewed = True
        if any_unreviewed:
            console.print()
            console.print(
                "[yellow]UNREVIEWED[/yellow] -- this lesson is original synthesis pending "
                "Jaeguek's review, not yet approved. Its citations are byte-verified against "
                "the real source, but the connective explanation between them hasn't been "
                "checked by a human F2FS expert yet."
            )


if __name__ == "__main__":
    app()
