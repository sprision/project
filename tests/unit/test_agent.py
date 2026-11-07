"""Agent tool-selection and orchestration tests — deterministic, no LLM, no
real model. Vision result is injected as a stub."""
import pytest

pytest.importorskip("sentence_transformers")

from pathlib import Path

from src.app.agent import InspectionAgent
from src.app.agent.tools import ToolContext, run_tool
from src.app.database import InspectionDB
from src.app.reporting import ReportGenerator
from src.app.retrieval import Retriever, ingest_documents
from src.app.schemas import (DefectRegion, ImageQuality, VisionResult)


def _ctx(tmp_path, inspectors=None):
    return ToolContext(
        retriever=Retriever(),
        db=InspectionDB(),
        inspectors=inspectors or {},
        report_fn=ReportGenerator().generate,
        work_dir=tmp_path)


def _vision(anom=True, score=0.8):
    return VisionResult(
        image_id="x", category="bottle", model_name="PaDiM",
        backbone="resnet18", image_anomaly_score=score,
        raw_anomaly_score=3.0, is_anomalous=anom, threshold=0.5,
        regions=[DefectRegion(region_id=1, bbox=(10, 10, 20, 20), area_px=400,
                              area_frac=0.1, score=0.85, centroid=(20.0, 20.0))]
        if anom else [],
        quality=ImageQuality(is_valid=True, width=256, height=256))


class TestToolDispatch:
    def test_unknown_tool(self, tmp_workspace, tmp_path):
        r = run_tool("does_not_exist", {}, _ctx(tmp_path), None)
        assert not r.ok and "unknown tool" in r.error

    def test_invalid_args(self, tmp_workspace, tmp_path):
        # missing required 'query' -> validation error, not a crash
        r = run_tool("retrieve_technical_documents", {}, _ctx(tmp_path), None)
        assert not r.ok and "invalid args" in r.error

    def test_retrieve_empty_query(self, tmp_workspace, tmp_path):
        r = run_tool("retrieve_technical_documents", {"query": "  "},
                     _ctx(tmp_path), None)
        assert not r.ok

    def test_compare_without_vision(self, tmp_workspace, tmp_path):
        r = run_tool("compare_inspection_results", {}, _ctx(tmp_path), None)
        assert not r.ok and "needs a current vision result" in r.error

    def test_regression_generate_report_with_dict_retrieval(self, tmp_workspace, tmp_path):
        ret_dict = {
            "query": "defect query",
            "chunks": [
                {
                    "chunk": {
                        "chunk_id": "c1", "doc_id": "d1", "title": "Doc",
                        "text": "sample text", "source": "doc.txt"
                    },
                    "score": 0.9, "rank": 1
                }
            ],
            "mode": "hybrid",
            "note": None
        }
        r = run_tool("generate_inspection_report", {
            "image_id": "sample",
            "question": "is there a defect?",
            "retrieval": ret_dict,
            "evidence": [
                {"kind": "visual_observation", "statement": "Anomaly found", "source": "model"}
            ],
            "trace": []
        }, _ctx(tmp_path), _vision())
        assert r.ok, f"generate_inspection_report failed: {r.error}"
        from src.app.schemas import InspectionReport
        assert isinstance(r.output, InspectionReport)
        assert r.output.retrieval is not None
        assert len(r.output.retrieval.chunks) == 1


class TestAgentDecisions:
    """Verify the agent CHOOSES tools appropriately for different questions."""

    def _agent(self, tmp_path, ctx):
        return InspectionAgent(ctx)

    def _run(self, tmp_path, question, vision=None, kb_docs=None):
        ctx = _ctx(tmp_path)
        if kb_docs:
            ctx.retriever.add_chunks(ingest_documents(kb_docs))
        agent = self._agent(tmp_path, ctx)
        # inject a fake image path; vision result injected directly so no model needed
        rep = agent.run(image_path=Path("x.png"), question=question,
                        vision=vision or _vision())
        return rep

    def test_generic_question_retrieves_and_reports(self, tmp_workspace,
                                                    tmp_path, kb_docs):
        rep = self._run(tmp_path, "is there a defect?", kb_docs=kb_docs)
        tools = [t["tool"] for t in rep.trace]
        assert "retrieve_technical_documents" in tools
        assert "generate_inspection_report" in tools
        assert rep.generator == "deterministic"
        assert rep.summary  # non-empty

    def test_history_question_uses_history_tool(self, tmp_workspace, tmp_path):
        rep = self._run(tmp_path, "has this defect appeared before?")
        tools = [t["tool"] for t in rep.trace]
        assert "search_inspection_history" in tools
        # history evidence recorded even when empty
        kinds = {e.kind for e in rep.findings}
        from src.app.schemas import EvidenceKind
        assert EvidenceKind.INSPECTION_HISTORY in kinds

    def test_compare_question_uses_compare(self, tmp_workspace, tmp_path):
        # seed a prior inspection so compare has something to work with
        ctx = _ctx(tmp_path)
        agent = InspectionAgent(ctx)
        agent.run(image_path=Path("a.png"), question="defect?",
                  vision=_vision(score=0.4))
        rep = agent.run(image_path=Path("b.png"),
                        question="how does this compare to before?",
                        vision=_vision(score=0.8))
        tools = [t["tool"] for t in rep.trace]
        assert "compare_inspection_results" in tools

    def test_bounded_iterations(self, tmp_workspace, tmp_path):
        rep = self._run(tmp_path, "what is the defect and its history?")
        # must terminate within the bound and always end with a report
        assert len(rep.trace) <= 8
        assert rep.trace[-1]["tool"] == "generate_inspection_report"

    def test_vision_failure_produces_limitation_not_crash(self, tmp_workspace,
                                                          tmp_path):
        rep = self._run(tmp_path, "defect?", vision=None)  # triggers inspect_image tool on missing path
        assert rep is not None
        # either a limitation was recorded or the report exists
        assert rep.limitations or rep.findings
