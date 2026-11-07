"""Report generator tests — deterministic path always works."""
from src.app.reporting import ReportGenerator
from src.app.schemas import (EvidenceItem, EvidenceKind, ImageQuality,
                             VisionResult)


def _vision(anom, score):
    return VisionResult(
        image_id="x", category="bottle", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=score, raw_anomaly_score=3.0,
        is_anomalous=anom, threshold=0.5, regions=[],
        quality=ImageQuality(is_valid=True, width=64, height=64))


def test_anomalous_summary(tmp_workspace):
    g = ReportGenerator()
    rep = g.generate("x", "defect?", _vision(True, 0.9), None, [], [])
    assert "Anomaly detected" in rep.summary
    assert rep.generator == "deterministic"
    assert rep.recommendations  # anomalous -> recommends human review
    assert "human" in rep.recommendations[0].statement.lower()


def test_normal_summary(tmp_workspace):
    rep = ReportGenerator().generate("x", "defect?", _vision(False, 0.1),
                                     None, [], [])
    assert "No significant anomaly" in rep.summary


def test_no_vision(tmp_workspace):
    rep = ReportGenerator().generate("x", "defect?", None, None, [], [])
    assert "could not be analysed" in rep.summary
    assert rep.vision is None


def test_limitation_recorded(tmp_workspace):
    ev = [EvidenceItem(kind=EvidenceKind.LIMITATION, statement="KB empty")]
    rep = ReportGenerator().generate("x", "q", _vision(False, 0.1), None, ev, [])
    assert "KB empty" in rep.limitations


def test_disclaimer_present(tmp_workspace):
    rep = ReportGenerator().generate("x", "q", None, None, [], [])
    assert "Decision support only" in rep.disclaimer


def test_regression_retrieval_as_dict(tmp_workspace):
    """Regression test: passing retrieval as a dictionary must not raise
    AttributeError: 'dict' object has no attribute 'chunks'."""
    retrieval_dict = {
        "query": "surface defect specification",
        "chunks": [
            {
                "chunk": {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "title": "MVTec AD",
                    "text": "Surface defect criteria",
                    "page": 1,
                    "section": "1.1",
                    "source": "manual.pdf",
                },
                "score": 0.95,
                "rank": 1,
            }
        ],
        "mode": "hybrid",
        "note": None,
    }
    g = ReportGenerator()
    rep = g.generate("x", "defect?", _vision(True, 0.9), retrieval_dict, [], [])
    assert rep is not None
    assert rep.retrieval is not None
    assert len(rep.retrieval.chunks) == 1
    assert rep.retrieval.chunks[0].chunk.title == "MVTec AD"


def test_regression_retrieval_as_dict_empty_chunks(tmp_workspace):
    """Regression test: dict retrieval with empty chunks correctly records limitation."""
    retrieval_dict = {
        "query": "unknown defect",
        "chunks": [],
        "mode": "hybrid",
        "note": "no documents indexed",
    }
    g = ReportGenerator()
    rep = g.generate("x", "defect?", _vision(True, 0.9), retrieval_dict, [], [])
    assert rep is not None
    assert any("No supporting passages were retrieved" in lim for lim in rep.limitations)
