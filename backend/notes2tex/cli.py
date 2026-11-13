"""Command line: notes2tex convert notes.pdf --subject math"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import config, dataset, evaluate, pipeline
from .subjects import list_subjects, load_subject
from .transcribe import ENGINES, engine_status, get_transcriber
from .transcribe.base import Transcriber, TranscriptionError

app = typer.Typer(add_completion=False, help="Convert handwritten tablet notes to LaTeX.")


ENGINE_HELP = f"Transcription engine: {', '.join(ENGINES)} (default: {config.ENGINE})"


def _transcriber(engine: str | None, offline: bool = False) -> Transcriber:
    try:
        return get_transcriber("offline" if offline else engine)
    except TranscriptionError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1)


def _print_build(result: pipeline.BuildResult) -> None:
    if result.fixes:
        typer.echo(f"Auto-fixed {result.fixes} compile error(s).")
    if result.ok:
        typer.secho(f"PDF: {result.pdf_path}", fg="green")
        if result.errors:
            typer.secho(f"{len(result.errors)} block(s) did not compile and are shown as red raw text:", fg="yellow")
            for e in result.errors:
                typer.echo(f"  blocks {e.block_ids}: {e.message}")
    else:
        typer.secho("LaTeX did not compile:", fg="red")
        for e in result.errors:
            where = f"line {e.line}" if e.line else "?"
            blocks = f" (blocks {e.block_ids})" if e.block_ids else ""
            typer.echo(f"  {where}: {e.message}{blocks}")
    typer.echo(f"TeX: {result.tex_path}")


@app.command()
def convert(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="PDF or image exported from your tablet app"),
    subject: str = typer.Option(..., "--subject", "-s", help="Subject key, e.g. math or cs"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Document title (default: file name)"),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help=ENGINE_HELP),
    offline: bool = typer.Option(False, help="Same as --engine offline: placeholders you type over"),
    final: bool = typer.Option(False, help="Final build: no highlighting of unsure readings"),
):
    """Import, transcribe and build a document in one go."""
    transcriber = _transcriber(engine, offline)
    doc_id = pipeline.import_document(file, subject, name)
    with dataset.connect() as conn:
        pages = dataset.list_pages(conn, doc_id)
    typer.echo(f"Imported document {doc_id}: {len(pages)} page(s), {sum(p['block_count'] for p in pages)} lines found.")

    total = 0.0

    def progress(page, result):
        nonlocal total
        cost = pipeline.estimate_cost(result.model, result.usage)
        total += cost or 0
        unsure = sum("\\unsure{" in b.latex for b in result.blocks)
        line = f"  page {page['page_index'] + 1}: {len(result.blocks)} blocks, {unsure} unsure"
        if cost is not None:
            line += f", ~${cost:.3f}"
        if result.usage.get("wall_seconds"):
            line += f", {result.usage['wall_seconds']:.0f}s"
        typer.echo(line)
        for w in result.warnings:
            typer.secho(f"    warning: {w}", fg="yellow")

    try:
        pipeline.transcribe_document(doc_id, transcriber, on_page=progress)
    except TranscriptionError as e:
        typer.secho(f"Transcription failed: {e}", fg="red")
        raise typer.Exit(1)
    if total:
        typer.echo(f"Estimated API cost: ~${total:.3f}")
    _print_build(pipeline.build_document(doc_id, draft=not final, fixer=None if transcriber.name == "offline" else transcriber))
    typer.echo(f"Review and correct it in the web app: notes2tex serve  (document {doc_id})")


@app.command("import")
def import_(
    file: Path = typer.Argument(..., exists=True, dir_okay=False),
    subject: str = typer.Option(..., "--subject", "-s"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
):
    """Import and segment without transcribing."""
    doc_id = pipeline.import_document(file, subject, name)
    typer.echo(f"Imported document {doc_id}")


@app.command()
def transcribe(
    doc_id: int,
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help=ENGINE_HELP),
    all_pages: bool = typer.Option(False, "--all", help="Redo reviewed pages too"),
):
    """Transcribe the pages of a document that are not done yet."""
    pipeline.transcribe_document(doc_id, _transcriber(engine), only_new=not all_pages)
    typer.echo("Done.")


@app.command()
def build(
    doc_id: int,
    final: bool = False,
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine used to auto-fix compile errors (offline = no auto-fix)"),
):
    """Assemble and compile a document to PDF."""
    fixer = _transcriber(engine)
    fixer = None if fixer.name == "offline" else fixer
    _print_build(pipeline.build_document(doc_id, draft=not final, fixer=fixer))


@app.command("list")
def list_docs():
    """List imported documents."""
    with dataset.connect() as conn:
        docs = dataset.list_documents(conn)
    if not docs:
        typer.echo("No documents yet.")
    for d in docs:
        typer.echo(
            f"{d['id']:>4}  {d['subject']:<6} {d['name']:<40} pages {d['page_count']}, reviewed {d['reviewed_count'] or 0}  [{d['split']}]"
        )


@app.command()
def engines():
    """Show which transcription engines are ready."""
    for e in engine_status():
        mark = typer.style("ready", fg="green") if e["available"] else typer.style("not ready", fg="yellow")
        default = " (default)" if e["name"] == config.ENGINE else ""
        typer.echo(f"{e['name']:<8} {mark:<20} {e['label']}{default}: {e['detail']}")


@app.command()
def subjects():
    """List subjects and how many example pages each has."""
    for key in list_subjects():
        s = load_subject(key)
        typer.echo(f"{key:<8} {s.name:<20} examples: {len(s.examples)}")


@app.command("eval")
def eval_(split: str = typer.Option("test", help="test | val | train | all"), as_json: bool = typer.Option(False, "--json")):
    """Token accuracy of model drafts against your reviewed finals."""
    with dataset.connect() as conn:
        rows = dataset.reviewed_blocks(conn, None if split == "all" else split)
    rep = evaluate.report(rows)
    if as_json:
        typer.echo(json.dumps(rep, indent=2))
        return
    o = rep["overall"]
    if not o["blocks"]:
        typer.echo(f"No reviewed blocks in the '{split}' split yet. Approve some pages in the web app first.")
        return
    typer.echo(f"Split: {split}   blocks: {o['blocks']}   tokens: {o['ref_tokens']}")
    typer.echo(f"Token accuracy: {o['token_accuracy']:.2%}   exact blocks: {o['exact_match']:.1%}   roles: {o['role_accuracy']:.1%}")
    for key in ("subject", "role", "model"):
        typer.echo(f"\nBy {key}:")
        for name, s in rep[key].items():
            acc = f"{s['token_accuracy']:.2%}" if s["token_accuracy"] is not None else "  n/a"
            typer.echo(f"  {name:<12} {acc}  ({s['blocks']} blocks)")


@app.command()
def serve(port: int = 8000, reload: bool = False):
    """Start the review web app."""
    import uvicorn

    typer.echo(f"notes2tex at http://localhost:{port}  (data: {config.DATA_DIR})")
    uvicorn.run("notes2tex.api:app", host="127.0.0.1", port=port, reload=reload)


if __name__ == "__main__":
    app()
