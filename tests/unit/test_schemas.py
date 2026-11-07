"""Schema contract tests — the data exchanged between components."""
from src.app.schemas import (DefectRegion, DocumentChunk, EvidenceItem,
                             EvidenceKind, ImageQuality, InspectionReport,
                             RetrievedChunk, RetrievalResult, VisionResult)


def test_vision_result_valid():
    r = VisionResult(
        image_id="x", category="bottle", model_name="PaDiM",
        backbone="resnet18", image_anomaly_score=0.7, raw_anomaly_score=3.1,
        is_anomalous=True, threshold=0.5,
        regions=[DefectRegion(region_id=1, bbox=(1, 2, 3, 4), area_px=12,
                              area_frac=0.003, score=0.8,
                              centroid=(2.0, 3.0))],
        quality=ImageQuality(is_valid=True, width=10, height=10))
    assert r.is_anomalous and r.regions[0].bbox == (1, 2, 3, 4)


def test_report_serialises():
    rep = InspectionReport(
        image_id="x", question="defect?", summary="s",
        findings=[EvidenceItem(kind=EvidenceKind.VISUAL_OBSERVATION,
                               statement="score 0.7", source="PaDiM")],
        recommendations=[], limitations=["l"],
        vision=None, retrieval=None, generator="deterministic")
    js = rep.model_dump_json()
    assert '"generator":"deterministic"' in js
    assert rep.disclaimer  # safety disclaimer always present


def test_document_chunk_provenance():
    c = DocumentChunk(chunk_id="d:0", doc_id="d", title="T", text="x",
                      page=2, section="1. Overview", source="f.pdf")
    assert c.page == 2 and c.source == "f.pdf"


def test_retrieval_result():
    c = DocumentChunk(chunk_id="d:0", doc_id="d", title="T", text="x")
    rr = RetrievalResult(query="q", mode="hybrid",
                         chunks=[RetrievedChunk(chunk=c, score=0.5, rank=1)])
    assert rr.chunks[0].rank == 1
