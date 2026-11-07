"""FastAPI application entry point.

Run:
    uvicorn src.app.main:app --reload
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import get_settings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="Industrial Inspection & Maintenance Decision Support",
        version="0.1.0",
        description="Multimodal AI agent: PaDiM visual anomaly detection + "
                    "RAG technical retrieval + bounded tool-calling agent.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"])

    app.include_router(router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def index():
        idx = FRONTEND / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return {"service": "industrial-inspection", "docs": "/docs"}

    if FRONTEND.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.on_event("startup")
    def _warm() -> None:
        # load service + retriever + db on startup so first request isn't slow
        from .services import get_service
        get_service()

    return app


app = create_app()
