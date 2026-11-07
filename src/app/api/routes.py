"""HTTP API — thin layer over InspectionService.

Validates uploads (extension + size + decodability), returns clean errors, and
never exposes internal paths or exception traces to the client.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ..config import get_settings
from ..schemas import InspectionReport, RetrievalResult
from ..services import get_service

log = logging.getLogger(__name__)
router = APIRouter()

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.png$")


@router.get("/health")
def health() -> dict:
    s = get_service()
    return {
        "status": "ok",
        "device": s.device,
        "categories": s.available_categories(),
        "kb_chunks": s.retriever.n_chunks,
        "llm_provider": s.s.llm_provider,
    }


@router.get("/categories")
def categories() -> dict:
    return {"categories": get_service().available_categories()}


@router.post("/inspect", response_model=InspectionReport)
async def inspect(
    file: UploadFile = File(...),
    question: str = Form("Is there a defect?"),
    category: str | None = Form(None),
) -> InspectionReport:
    s = get_settings()
    svc = get_service()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in s.allowed_image_ext:
        raise HTTPException(415, f"unsupported file type {ext!r}; "
                              f"allowed: {sorted(s.allowed_image_ext)}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file exceeds {s.max_upload_mb} MB")

    path = svc.save_upload(file.filename or "upload", data)
    try:
        report = svc.inspect(path, question=question, category=category)
    except Exception as e:
        log.exception("inspect failed")
        raise HTTPException(500, "inspection failed") from e
    finally:
        # uploads are temporary working copies; the report is persisted to the DB
        pass
    return report


@router.get("/kb/query", response_model=RetrievalResult)
def kb_query(q: str, top_k: int | None = None) -> RetrievalResult:
    if not q.strip():
        raise HTTPException(400, "query must be non-empty")
    return get_service().query_kb(q, top_k=top_k)


@router.post("/kb/ingest")
async def kb_ingest(file: UploadFile = File(...)) -> dict:
    svc = get_service()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".pdf", ".txt", ".md", ".rst"}:
        raise HTTPException(415, f"unsupported doc type {ext!r}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    tmp = svc.save_upload(file.filename or "doc", data)
    n = svc.ingest_paths([tmp])
    svc.copy_into_kb(tmp)  # keep original in kb/ for the record
    return {"chunks_indexed": n, "total_chunks": svc.retriever.n_chunks}


@router.get("/heatmap/{name}")
def heatmap(name: str) -> FileResponse:
    if not _SAFE_NAME.match(name):
        raise HTTPException(400, "invalid file name")
    p = get_service().heatmap_dir / name
    if not p.exists():
        raise HTTPException(404, "heatmap not found")
    return FileResponse(p, media_type="image/png")


@router.get("/reports")
def reports(limit: int = 10) -> dict:
    return {"inspections": get_service().db.search(limit=limit)}


@router.get("/reports/{insp_id}")
def report_detail(insp_id: int) -> dict:
    r = get_service().db.get(insp_id)
    if r is None:
        raise HTTPException(404, "inspection not found")
    return r


@router.get("/eval/cv")
def eval_cv() -> dict:
    import json
    s = get_settings()
    p = s.runs_dir / "eval_cv.json"
    if not p.exists():
        raise HTTPException(404, "CV evaluation not found")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/eval/calibration")
def eval_calibration() -> dict:
    import json
    s = get_settings()
    p = s.runs_dir / "calibration.json"
    if not p.exists():
        raise HTTPException(404, "Calibration evaluation not found")
    return json.loads(p.read_text(encoding="utf-8"))
