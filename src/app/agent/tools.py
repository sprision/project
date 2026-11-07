"""Tool registry for the inspection agent.

Each tool is a small dataclass: validated inputs, a callable, and a structured
output. The agent dispatches by name; tools never touch the HTTP layer and
never fabricate results — failures raise ToolError which the agent records in
its trace instead of a fake success.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from ..config import get_settings
from ..database import InspectionDB
from ..retrieval import Retriever
from ..schemas import (EvidenceItem, InspectionReport, RetrievalResult,
                       VisionResult)
from ..vision import AnomalyInspector


class ToolError(Exception):
    """Raised when a tool cannot complete. Recorded in the agent trace."""


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0


@dataclass
class ToolContext:
    """Services handed to tools — keeps them decoupled from construction."""
    retriever: Retriever
    db: InspectionDB
    inspectors: dict[str, AnomalyInspector]  # category -> fitted model
    report_fn: Callable[..., InspectionReport]
    work_dir: Path


@dataclass
class Tool:
    name: str
    description: str
    input_model: type | None          # pydantic model for arg validation
    fn: Callable[..., Any]
    requires_image: bool = False


# --------------------------------------------------------------------------- #
#  Tool implementations                                                        #
# --------------------------------------------------------------------------- #

def _t_inspect_image(args: dict, ctx: ToolContext,
                     vision: VisionResult | None) -> VisionResult:
    """Run the vision model on the uploaded image (or reuse a prior result)."""
    if vision is not None and vision.image_id == args.get("image_id", vision.image_id):
        return vision  # reuse — avoid re-running the model
    category = args.get("category")
    inspector = ctx.inspectors.get(category) if category else \
        next(iter(ctx.inspectors.values()), None)
    if inspector is None:
        raise ToolError("no fitted vision model available for category "
                        f"{category!r}")
    image_path = Path(args["image_path"])
    if not image_path.exists():
        raise ToolError(f"image not found: {args['image_path']}")
    return inspector.inspect(image_path)


def _t_retrieve(args: dict, ctx: ToolContext,
                vision: VisionResult | None) -> RetrievalResult:
    """Semantic retrieval over the technical knowledge base."""
    query = args.get("query") or ""
    if not query.strip():
        raise ToolError("retrieve_technical_documents requires a non-empty query")
    top_k = int(args.get("top_k") or get_settings().retriever_top_k)
    return ctx.retriever.search(query, top_k=top_k)


def _t_history(args: dict, ctx: ToolContext,
               vision: VisionResult | None) -> list[dict]:
    """Search past inspection records (SQLite) for this category/verdict."""
    category = args.get("category") or (vision.category if vision else None)
    limit = int(args.get("limit") or 10)
    return ctx.db.search(category=category, limit=limit)


def _t_compare(args: dict, ctx: ToolContext,
               vision: VisionResult | None) -> dict:
    """Compare current result against a prior inspection of the same category."""
    if vision is None:
        raise ToolError("compare_inspection_results needs a current vision result")
    prior_id = args.get("prior_id")
    prior = ctx.db.get(prior_id) if prior_id else \
        ctx.db.latest(category=vision.category)
    if prior is None:
        raise ToolError("no prior inspection to compare against")
    return {
        "current": {"score": vision.image_anomaly_score,
                    "is_anomalous": vision.is_anomalous,
                    "n_regions": len(vision.regions)},
        "prior": {"id": prior["id"], "score": prior["image_anomaly_score"],
                  "is_anomalous": bool(prior["is_anomalous"]),
                  "created_at": prior["created_at"]},
        "delta_score": round(vision.image_anomaly_score -
                             prior["image_anomaly_score"], 4),
    }


def _t_summarize(args: dict, ctx: ToolContext,
                 vision: VisionResult | None) -> dict:
    """Condense accumulated evidence into a short structured summary the agent
    can hand to the report generator."""
    items: list[dict] = args.get("evidence") or []
    n_obs = sum(1 for i in items if i.get("kind") == "visual_observation")
    n_facts = sum(1 for i in items if i.get("kind") == "retrieved_fact")
    return {
        "n_evidence": len(items),
        "n_visual_observations": n_obs,
        "n_retrieved_facts": n_facts,
        "has_vision": vision is not None,
        "is_anomalous": bool(vision.is_anomalous) if vision else None,
    }


def _t_report(args: dict, ctx: ToolContext,
              vision: VisionResult | None) -> InspectionReport:
    """Assemble the final InspectionReport from accumulated evidence."""
    ret_arg = args.get("retrieval")
    ret_obj = None
    if isinstance(ret_arg, RetrievalResult):
        ret_obj = ret_arg
    elif isinstance(ret_arg, dict):
        ret_obj = RetrievalResult(**ret_arg)

    evidence_items = []
    for e in (args.get("evidence") or []):
        if isinstance(e, EvidenceItem):
            evidence_items.append(e)
        elif isinstance(e, dict):
            evidence_items.append(EvidenceItem(**e))

    return ctx.report_fn(
        image_id=args.get("image_id") or (vision.image_id if vision else "unknown"),
        question=args.get("question", ""),
        vision=vision,
        retrieval=ret_obj,
        evidence=evidence_items,
        trace=args.get("trace") or [],
    )


# --------------------------------------------------------------------------- #
#  Input schemas (lightweight pydantic for validation)                         #
# --------------------------------------------------------------------------- #
from pydantic import BaseModel  # noqa: E402


class _ImgArgs(BaseModel):
    image_path: str
    image_id: str | None = None
    category: str | None = None


class _RetrieveArgs(BaseModel):
    query: str
    top_k: int | None = None


class _HistoryArgs(BaseModel):
    category: str | None = None
    limit: int | None = None


class _CompareArgs(BaseModel):
    prior_id: int | None = None


class _SummarizeArgs(BaseModel):
    evidence: list[dict] = field(default_factory=list)


class _ReportArgs(BaseModel):
    image_id: str | None = None
    question: str = ""
    evidence: list[dict | Any] = field(default_factory=list)
    retrieval: dict | Any | None = None
    trace: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Registry                                                                    #
# --------------------------------------------------------------------------- #

def build_tools() -> dict[str, Tool]:
    return {
        "inspect_image": Tool(
            name="inspect_image",
            description="Run the anomaly-detection model on an image and return "
                        "structured observations (score + defect regions).",
            input_model=_ImgArgs, fn=_t_inspect_image, requires_image=True),
        "retrieve_technical_documents": Tool(
            name="retrieve_technical_documents",
            description="Search the technical knowledge base for passages "
                        "relevant to the observed finding or the user question.",
            input_model=_RetrieveArgs, fn=_t_retrieve),
        "search_inspection_history": Tool(
            name="search_inspection_history",
            description="Search prior inspections recorded in the database for "
                        "this category/verdict to establish history or trends.",
            input_model=_HistoryArgs, fn=_t_history),
        "compare_inspection_results": Tool(
            name="compare_inspection_results",
            description="Compare the current vision result against a prior "
                        "inspection of the same category.",
            input_model=_CompareArgs, fn=_t_compare),
        "summarize_evidence": Tool(
            name="summarize_evidence",
            description="Condense accumulated evidence into a structured summary.",
            input_model=_SummarizeArgs, fn=_t_summarize),
        "generate_inspection_report": Tool(
            name="generate_inspection_report",
            description="Assemble the final cited inspection report. Call last.",
            input_model=_ReportArgs, fn=_t_report),
    }


def run_tool(name: str, args: dict, ctx: ToolContext,
             vision: VisionResult | None) -> ToolResult:
    """Validate args, run the tool, and wrap the result. Never raises."""
    tools = build_tools()
    tool = tools.get(name)
    t0 = time.time()
    if tool is None:
        return ToolResult(name=name, ok=False,
                          error=f"unknown tool {name!r}")
    try:
        parsed = tool.input_model(**(args or {}))
        out = tool.fn(parsed.model_dump(exclude_none=True), ctx, vision)
        return ToolResult(name=name, ok=True, output=out,
                          elapsed_ms=round((time.time() - t0) * 1000, 1))
    except ValidationError as e:
        return ToolResult(name=name, ok=False,
                          error=f"invalid args: {e.errors()[0]['msg']}")
    except ToolError as e:
        return ToolResult(name=name, ok=False, error=str(e))
    except Exception as e:  # defensive: tool failure must not crash the agent
        return ToolResult(name=name, ok=False, error=f"{type(e).__name__}: {e}")
