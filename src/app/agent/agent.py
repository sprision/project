"""Single bounded tool-calling inspection agent.

This is the orchestrator. It is intentionally NOT a fixed pipeline: it inspects
the request and the observations already gathered, then *decides* which tool to
call next within a bounded number of iterations, records every call in a trace,
and finishes by generating a report. Tool selection is a transparent policy
(keyword/observation-driven) so the whole thing is deterministic and testable
without an LLM key; an LLM planner can replace `_decide` later.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..schemas import (EvidenceItem, EvidenceKind, InspectionReport,
                       RetrievalResult, VisionResult)
from .tools import ToolContext, build_tools, run_tool

_HIST_WORDS = re.compile(r"\b(history|previous|prior|before|trend|earlier|last "
                         r"time|past|again|recurr)\b", re.I)
_COMPARE_WORDS = re.compile(r"\b(compare|difference|changed|worse|better|same as|"
                            r"versus|vs\.?|than before)\b", re.I)
_TECH_WORDS = re.compile(r"\b(spec|standard|manual|procedure|tolerance|limit|"
                         r"acceptable|criteria|guideline|maintenance|repair|"
                         r"replace|inspect how|what should)\b", re.I)


class InspectionAgent:
    """Bounded tool-calling agent for the inspection workflow."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.s = get_settings()
        self.tools = build_tools()

    # ------------------------------------------------------------------ #
    #  Decision policy — returns (tool_name, args) or None to finish.      #
    # ------------------------------------------------------------------ #
    def _decide(self, question: str, vision: VisionResult | None,
                retrieval: RetrievalResult | None,
                evidence: list[EvidenceItem],
                calls: list[dict]) -> tuple[str, dict] | None:
        called = {c["tool"] for c in calls if c["ok"]}
        attempted = {c["tool"] for c in calls}

        # 1. We must have a vision result before anything else.
        if vision is None:
            return None  # caller supplies image_path; nothing to inspect yet

        # 2. Always ground technical answers in the KB when we have findings,
        #    unless the question is purely historical/comparative.
        want_history = bool(_HIST_WORDS.search(question))
        want_compare = bool(_COMPARE_WORDS.search(question))
        want_tech = bool(_TECH_WORDS.search(question))

        if "retrieve_technical_documents" not in attempted and not want_history \
                and not want_compare:
            q = question
            if vision.is_anomalous and vision.regions:
                # enrich the query with the observed defect extent
                top = vision.regions[0]
                q = (f"{question} {vision.category} surface defect "
                     f"{top.area_frac:.1%} of area")
            return "retrieve_technical_documents", {"query": q}

        # 3. History: if asked, or if records exist that could contextualise.
        if "search_inspection_history" not in attempted:
            if want_history or want_compare:
                return "search_inspection_history", {"category": vision.category}

        # 4. Compare: only when the user asks and a prior exists.
        if want_compare and "compare_inspection_results" not in attempted:
            return "compare_inspection_results", {}

        # 5. If we still haven't retrieved and the question is generic, do one
        #    retrieval pass anyway (cheap, grounds the answer).
        if "retrieve_technical_documents" not in attempted and not want_history:
            return "retrieve_technical_documents", {"query": question}

        # 6. Summarize before reporting (only if we gathered anything).
        if "summarize_evidence" not in attempted and evidence:
            return "summarize_evidence", {
                "evidence": [e.model_dump() for e in evidence]}

        # 7. Done -> report.
        if "generate_inspection_report" not in attempted:
            return "generate_inspection_report", {
                "image_id": vision.image_id,
                "question": question,
                "evidence": [e.model_dump() for e in evidence],
                "retrieval": retrieval.model_dump() if retrieval else None,
                "trace": calls,
            }
        return None

    # ------------------------------------------------------------------ #
    #  Evidence extraction from tool outputs                               #
    # ------------------------------------------------------------------ #
    def _harvest(self, name: str, out: Any,
                 evidence: list[EvidenceItem]) -> None:
        if isinstance(out, VisionResult):
            src = f"{out.model_name} ({out.backbone})"
            score_text = (f"raw anomaly score {out.raw_anomaly_score:.2f} vs "
                          f"calibrated threshold {out.threshold}") \
                if out.verdict_mode == "calibrated_raw" else \
                (f"image anomaly score {out.image_anomaly_score:.2f} vs "
                 f"threshold {out.threshold}")
            evidence.append(EvidenceItem(
                kind=EvidenceKind.VISUAL_OBSERVATION,
                statement=(f"{score_text} — {'anomalous' if out.is_anomalous else 'normal'}; "
                           f"{len(out.regions)} defect region(s)."),
                source=src, confidence=out.image_anomaly_score))
            for r in out.regions[:3]:
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.VISUAL_OBSERVATION,
                    statement=(f"Region {r.region_id}: bbox {r.bbox}, "
                               f"{r.area_frac:.1%} of image, "
                               f"mean score {r.score}."),
                    source=src, confidence=r.score))
        elif isinstance(out, RetrievalResult):
            if not out.chunks:
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.LIMITATION,
                    statement="Knowledge base returned no supporting passages "
                              f"for the query (mode={out.mode})."))
            for rc in out.chunks:
                c = rc.chunk
                loc = f"p.{c.page}" if c.page else c.source
                sec = f" §{c.section}" if c.section else ""
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.RETRIEVED_FACT,
                    statement=c.text[:300].replace("\n", " "),
                    source=f"{c.title}{sec}, {loc}", confidence=rc.score))
        elif isinstance(out, list):  # history rows
            if out:
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.INSPECTION_HISTORY,
                    statement=f"{len(out)} prior inspection(s) on record; "
                              f"latest score {out[0]['image_anomaly_score']:.2f} "
                              f"at {out[0]['created_at']}.",
                    source="inspection_history"))
            else:
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.INSPECTION_HISTORY,
                    statement="No prior inspections found for this category.",
                    source="inspection_history"))
        elif isinstance(out, dict) and "delta_score" in out:
            d = out["delta_score"]
            evidence.append(EvidenceItem(
                kind=EvidenceKind.HYPOTHESIS,
                statement=(f"Score changed by {d:+.2f} vs prior inspection "
                           f"#{out['prior']['id']} "
                           f"({'worse' if d > 0 else 'improved/same'})."),
                source="compare_inspection_results"))

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #
    def run(self, image_path: Path, question: str,
            vision: VisionResult | None = None,
            category: str | None = None) -> InspectionReport:
        max_iters = self.s.agent_max_iters
        evidence: list[EvidenceItem] = []
        retrieval: RetrievalResult | None = None
        trace: list[dict] = []
        report: InspectionReport | None = None

        # Seed: if caller already ran vision, use it; otherwise inspect first.
        if vision is None:
            res = run_tool("inspect_image",
                           {"image_path": str(image_path),
                            "category": category}, self.ctx, None)
            trace.append({"tool": "inspect_image", "ok": res.ok,
                          "error": res.error, "ms": res.elapsed_ms})
            if res.ok and isinstance(res.output, VisionResult):
                vision = res.output
                self._harvest("inspect_image", vision, evidence)
            else:
                # Vision failed — still produce a report stating the limitation.
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.LIMITATION,
                    statement=f"Vision analysis failed: {res.error}"))

        for _ in range(max_iters):
            nxt = self._decide(question, vision, retrieval, evidence, trace)
            if nxt is None:
                break
            name, args = nxt
            res = run_tool(name, args, self.ctx, vision)
            trace.append({"tool": name, "ok": res.ok, "error": res.error,
                          "ms": res.elapsed_ms})
            if res.ok:
                if isinstance(res.output, RetrievalResult):
                    retrieval = res.output
                if name == "generate_inspection_report" and \
                        isinstance(res.output, InspectionReport):
                    report = res.output
                else:
                    self._harvest(name, res.output, evidence)
            else:
                evidence.append(EvidenceItem(
                    kind=EvidenceKind.LIMITATION,
                    statement=f"Tool {name} failed: {res.error}",
                    source=name))

        if report is None:
            # Shouldn't happen (decide() always ends on report), but be safe.
            report = self.ctx.report_fn(
                image_id=vision.image_id if vision else "unknown",
                question=question, vision=vision, retrieval=retrieval,
                evidence=evidence, trace=trace)

        # Persist the inspection so history/compare have data next time.
        try:
            self.ctx.db.insert(report)
        except Exception as e:  # persistence failure shouldn't lose the report
            report.limitations.append(f"could not persist inspection: {e}")
        report.trace = trace
        return report
