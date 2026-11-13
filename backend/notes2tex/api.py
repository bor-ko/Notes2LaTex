"""FastAPI backend for the review web app."""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, dataset, evaluate, pipeline
from .assemble import ROLES
from .ingest import IMAGE_SUFFIXES
from .subjects import list_subjects, load_subject
from .transcribe import ENGINES, engine_status, get_transcriber
from .transcribe.base import Transcriber, TranscriptionError

app = FastAPI(title="notes2tex")
# A local model handles one page at a time; API engines can overlap.
_local_jobs = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local")
_jobs = ThreadPoolExecutor(max_workers=2, thread_name_prefix="transcribe")


@app.on_event("startup")
def _recover() -> None:
    # Pages left mid-transcription by a previous server run start over.
    with dataset.connect() as conn:
        conn.execute("UPDATE pages SET status = 'new' WHERE status = 'transcribing'")


def _transcriber(engine: str | None) -> Transcriber:
    if engine is not None and engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine {engine}")
    try:
        return get_transcriber(engine)
    except TranscriptionError as e:
        raise HTTPException(400, str(e))


def _queue_pages(page_ids: list[int], transcriber: Transcriber) -> None:
    with dataset.connect() as conn:
        for pid in page_ids:
            dataset.set_page_status(conn, pid, "transcribing")

    def run(pid: int) -> None:
        try:
            pipeline.transcribe_page(pid, transcriber)
        except Exception:
            pass  # status 'error' + message are stored on the page

    pool = _local_jobs if transcriber.name == "ollama" else _jobs
    for pid in page_ids:
        pool.submit(run, pid)


def _need(value, what: str):
    if value is None:
        raise HTTPException(404, f"{what} not found")
    return value


# ---- meta ------------------------------------------------------------------


@app.get("/api/status")
def status():
    subjects = []
    for key in list_subjects():
        s = load_subject(key)
        subjects.append({"key": key, "name": s.name, "examples": len(s.examples), "default_code_language": s.default_code_language})
    return {"default_engine": config.ENGINE, "engines": engine_status(), "roles": ROLES, "subjects": subjects}


# ---- documents -------------------------------------------------------------


@app.get("/api/documents")
def documents():
    with dataset.connect() as conn:
        return dataset.list_documents(conn)


@app.post("/api/documents")
def upload(
    file: UploadFile = File(...),
    subject: str = Form(...),
    name: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    transcribe: bool = Form(True),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES | {".pdf"}:
        raise HTTPException(400, "Upload a PDF or an image export.")
    if subject not in list_subjects():
        raise HTTPException(400, f"Unknown subject {subject}")
    transcriber = _transcriber(engine) if transcribe else None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        doc_id = pipeline.import_document(path, subject, (name or "").strip() or Path(file.filename or "notes").stem)
    if transcriber:
        with dataset.connect() as conn:
            _queue_pages([p["id"] for p in dataset.list_pages(conn, doc_id)], transcriber)
    return {"id": doc_id}


def _pdf_info(doc: dict) -> dict:
    tex = pipeline.output_dir(doc) / f"{pipeline.slugify(doc['name'])}.tex"
    pdf = tex.with_suffix(".pdf")
    return {
        "tex_path": str(tex) if tex.exists() else None,
        "pdf_url": f"/files/pdf/{doc['id']}.pdf?t={int(pdf.stat().st_mtime)}" if pdf.exists() else None,
    }


@app.get("/api/documents/{doc_id}")
def document(doc_id: int):
    with dataset.connect() as conn:
        doc = _need(dataset.get_document(conn, doc_id), "Document")
        pages = dataset.list_pages(conn, doc_id)
    for p in pages:
        p["usage"] = json.loads(p.pop("usage_json") or "null")
        p["cost"] = pipeline.estimate_cost(p["model"] or "", p["usage"] or {})
    return {**doc, **_pdf_info(doc), "pages": pages}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: int):
    pipeline.delete_document(doc_id)
    return {"ok": True}


class TranscribeReq(BaseModel):
    engine: Optional[str] = None
    all: bool = False


@app.post("/api/documents/{doc_id}/transcribe")
def transcribe_document(doc_id: int, req: TranscribeReq):
    transcriber = _transcriber(req.engine)
    with dataset.connect() as conn:
        pages = dataset.list_pages(conn, doc_id)
    todo = [p["id"] for p in pages if req.all or p["status"] in ("new", "error")]
    _queue_pages(todo, transcriber)
    return {"queued": len(todo)}


class BuildReq(BaseModel):
    final: bool = False
    engine: Optional[str] = None  # used to auto-fix compile errors


@app.post("/api/documents/{doc_id}/build")
def build(doc_id: int, req: BuildReq):
    try:
        fixer: Transcriber | None = get_transcriber(req.engine)
    except TranscriptionError:
        fixer = None  # engine not available: build without auto-fix
    if fixer is not None and fixer.name == "offline":
        fixer = None
    try:
        result = pipeline.build_document(doc_id, draft=not req.final, fixer=fixer)
    except TranscriptionError:
        result = pipeline.build_document(doc_id, draft=not req.final, fixer=None)
    with dataset.connect() as conn:
        doc = dataset.get_document(conn, doc_id)
    return {
        "ok": result.ok,
        "fixes": result.fixes,
        "errors": [e.__dict__ for e in result.errors],
        "log_tail": "\n".join(result.log.splitlines()[-25:]) if not result.ok else "",
        **_pdf_info(doc),
    }


# ---- pages & blocks --------------------------------------------------------


def _block_view(b: dict) -> dict:
    return {**b, "crop_url": f"/files/crop/{b['id']}.png?v={b['x0']}-{b['y0']}-{b['x1']}-{b['y1']}", "unsure": "\\unsure{" in (b["latex"] or "")}


@app.get("/api/pages/{page_id}")
def page(page_id: int):
    with dataset.connect() as conn:
        p = _need(dataset.get_page(conn, page_id), "Page")
        doc = dataset.get_document(conn, p["document_id"])
        siblings = [s["id"] for s in dataset.list_pages(conn, p["document_id"])]
        blocks = [_block_view(b) for b in dataset.get_blocks(conn, page_id)]
    i = siblings.index(page_id)
    return {
        **{k: v for k, v in p.items() if k not in ("image_path", "mask_path", "usage_json")},
        "document": {"id": doc["id"], "name": doc["name"], "subject": doc["subject"], "split": doc["split"]},
        "image_url": f"/files/page/{page_id}.png",
        "prev_id": siblings[i - 1] if i > 0 else None,
        "next_id": siblings[i + 1] if i + 1 < len(siblings) else None,
        "page_count": len(siblings),
        "blocks": blocks,
    }


class BlockEdit(BaseModel):
    id: int
    role: str
    latex: str
    continues_previous: bool = False
    code_language: Optional[str] = None


def _save(page_id: int, edits: list[BlockEdit]) -> None:
    with dataset.connect() as conn:
        valid = {b["id"] for b in dataset.get_blocks(conn, page_id)}
        for e in edits:
            if e.id not in valid:
                raise HTTPException(400, f"Block {e.id} is not on page {page_id}")
            if e.role not in ROLES:
                raise HTTPException(400, f"Unknown role {e.role}")
            dataset.update_block(conn, e.id, **e.model_dump(exclude={"id"}))


@app.put("/api/pages/{page_id}/blocks")
def save_blocks(page_id: int, edits: list[BlockEdit]):
    _save(page_id, edits)
    return page(page_id)


@app.post("/api/pages/{page_id}/approve")
def approve(page_id: int, edits: list[BlockEdit]):
    _save(page_id, edits)
    with dataset.connect() as conn:
        p = _need(dataset.get_page(conn, page_id), "Page")
        if p["status"] not in ("transcribed", "reviewed"):
            raise HTTPException(400, "Transcribe this page before approving it.")
        dataset.approve_page(conn, page_id)
    return page(page_id)


@app.post("/api/pages/{page_id}/transcribe")
def retranscribe(page_id: int, req: TranscribeReq):
    with dataset.connect() as conn:
        _need(dataset.get_page(conn, page_id), "Page")
    _queue_pages([page_id], _transcriber(req.engine))
    return page(page_id)


@app.post("/api/blocks/{block_id}/merge-next")
def merge_next(block_id: int):
    with dataset.connect() as conn:
        b = _need(dataset.get_block(conn, block_id), "Block")
    try:
        pipeline.merge_with_next(block_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return page(b["page_id"])


class SplitReq(BaseModel):
    at: float = 0.5


@app.post("/api/blocks/{block_id}/split")
def split(block_id: int, req: SplitReq):
    with dataset.connect() as conn:
        b = _need(dataset.get_block(conn, block_id), "Block")
    pipeline.split_block(block_id, req.at)
    return page(b["page_id"])


# ---- evaluation ------------------------------------------------------------


@app.get("/api/eval")
def eval_report(split: str = "test"):
    with dataset.connect() as conn:
        rows = dataset.reviewed_blocks(conn, None if split == "all" else split)
        docs = dataset.list_documents(conn)
    counts = {s: sum(1 for d in docs if d["split"] == s) for s in ("train", "val", "test")}
    reviewed_pages = sum(d["reviewed_count"] or 0 for d in docs)
    return {"split": split, "documents_per_split": counts, "reviewed_pages": reviewed_pages, **evaluate.report(rows)}


# ---- files -----------------------------------------------------------------


@app.get("/files/page/{page_id}.png")
def page_image(page_id: int):
    with dataset.connect() as conn:
        p = _need(dataset.get_page(conn, page_id), "Page")
    return FileResponse(p["image_path"], media_type="image/png")


@app.get("/files/crop/{block_id}.png")
def crop_image(block_id: int):
    with dataset.connect() as conn:
        b = _need(dataset.get_block(conn, block_id), "Block")
    if not b["crop_path"] or not Path(b["crop_path"]).exists():
        raise HTTPException(404, "No crop")
    return FileResponse(b["crop_path"], media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/files/pdf/{doc_id}.pdf")
def pdf(doc_id: int):
    with dataset.connect() as conn:
        doc = _need(dataset.get_document(conn, doc_id), "Document")
    path = pipeline.output_dir(doc) / f"{pipeline.slugify(doc['name'])}.pdf"
    if not path.exists():
        raise HTTPException(404, "Not built yet")
    return FileResponse(path, media_type="application/pdf", headers={"Cache-Control": "no-cache"})


if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
