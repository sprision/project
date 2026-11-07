"""Inspection history DB tests."""
from src.app.database import InspectionDB
from src.app.schemas import (EvidenceItem, EvidenceKind, InspectionReport,
                             ImageQuality, VisionResult)


def _report(cat="bottle", score=0.8, anom=True):
    v = VisionResult(
        image_id="img1", category=cat, model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=score, raw_anomaly_score=3.0,
        is_anomalous=anom, threshold=0.5, regions=[],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    return InspectionReport(
        image_id="img1", question="defect?", summary="s",
        findings=[EvidenceItem(kind=EvidenceKind.VISUAL_OBSERVATION,
                               statement="score", source="PaDiM")],
        recommendations=[], limitations=[], vision=v, retrieval=None,
        generator="deterministic")


def test_insert_and_search(tmp_workspace):
    db = InspectionDB()
    i1 = db.insert(_report(score=0.8))
    db.insert(_report(score=0.2, anom=False))
    assert i1 == 1
    rows = db.search(category="bottle")
    assert len(rows) == 2 and rows[0]["id"] == 2  # newest first


def test_latest_and_get(tmp_workspace):
    db = InspectionDB()
    db.insert(_report(score=0.3))
    latest = db.latest("bottle")
    assert latest["image_anomaly_score"] == 0.3
    assert db.get(1)["image_id"] == "img1"
    assert db.get(999) is None
    assert db.latest("nope") is None
