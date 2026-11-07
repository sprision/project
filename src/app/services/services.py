"""Application service: wires together vision models, retriever, DB and agent.

Loaded once (lazily) and shared by the API. Vision models are fitted lazily per
category from cached PaDiM weights in the index dir, so the server starts fast
and only builds what's needed.
"""
from __future__ import annotations

import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..agent import InspectionAgent
from ..agent.tools import ToolContext
from ..config import get_settings
from ..database import InspectionDB
from ..reporting import ReportGenerator
from ..retrieval import Retriever
from ..schemas import InspectionReport
from ..vision import (AnomalyInspector, CategoryMetadata, ModelRegistry,
                       get_process_memory_mb)


class InspectionService:
    def __init__(self, cache_capacity: int = 3) -> None:
        self.s = get_settings()
        self.device = self.s.resolved_device()
        self.retriever = Retriever()
        self.db = InspectionDB()
        self.reporter = ReportGenerator()
        self._upload_dir = self.s.data_dir / "uploads"
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._overlay_dir = self.s.data_dir / "heatmaps"

        # Scalable Model Registry with LRU Cache
        self.registry = ModelRegistry(
            vision_dir=self.s.index_dir / "vision",
            cache_capacity=cache_capacity,
            device=self.device,
            overlay_dir=self._overlay_dir,
        )

        ctx = ToolContext(
            retriever=self.retriever, db=self.db,
            inspectors=self.registry.cache,
            report_fn=self.reporter.generate,
            work_dir=self._upload_dir)
        self.agent = InspectionAgent(ctx)

    # -- vision model loading via LRU cache --
    def inspector_for(self, category: str | None) -> AnomalyInspector | None:
        return self.registry.get_inspector(category)

    def _default_category(self) -> str | None:
        cats = self.registry.discover_categories()
        return cats[0] if cats else None

    def available_categories(self) -> list[str]:
        return self.registry.discover_categories()

    def categories_metadata(self) -> list[dict[str, Any]]:
        return self.registry.list_all_metadata()

    def cache_stats(self) -> dict[str, Any]:
        return {
            "capacity": self.registry.cache.capacity,
            "cached_models": self.registry.cache.keys(),
            "stats": self.registry.cache.stats,
            "memory": get_process_memory_mb(),
        }

    @property
    def heatmap_dir(self) -> Path:
        return self._overlay_dir

    # -- workflows --
    def inspect(self, image_path: Path, question: str,
                category: str | None) -> InspectionReport:
        inspector = self.inspector_for(category)
        vision = None
        if inspector is not None:
            vision = inspector.inspect(image_path)
        return self.agent.run(image_path=image_path, question=question,
                              vision=vision, category=inspector.category
                              if inspector else category)

    def save_upload(self, filename: str, data: bytes) -> Path:
        ext = Path(filename).suffix.lower()
        safe = f"{uuid.uuid4().hex}{ext}"
        dest = self._upload_dir / safe
        dest.write_bytes(data)
        return dest

    def query_kb(self, question: str, top_k: int | None = None):
        return self.retriever.search(question, top_k=top_k)

    def ingest_paths(self, paths: list[Path]) -> int:
        from ..retrieval import ingest_documents
        chunks = ingest_documents(paths)
        return self.retriever.add_chunks(chunks)

    def copy_into_kb(self, src: Path) -> Path:
        kb = self.s.data_dir / "kb"
        kb.mkdir(parents=True, exist_ok=True)
        dest = kb / src.name
        shutil.copy2(src, dest)
        return dest


@lru_cache
def get_service() -> InspectionService:
    return InspectionService()
