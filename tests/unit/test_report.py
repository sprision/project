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


def test_borderline_score_generates_human_review_recommendation(tmp_workspace):
    """Verify that scores close to the calibrated threshold trigger a borderline warning."""
    v = VisionResult(
        image_id="borderline_sample", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.55, raw_anomaly_score=31.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep = ReportGenerator().generate("borderline_sample", "defect?", v, None, [], [])
    assert any("Borderline score" in r.statement for r in rep.recommendations)
    assert any("Borderline score proximity" in lim for lim in rep.limitations)
    assert any("Cable category exhibits documented score overlap" in lim for lim in rep.limitations)


def test_categorized_findings_tripartite_structure(tmp_workspace):
    """Verify that findings are structured into confirmed defects, potential anomalies, and uncertain review."""
    # 1. Borderline sample -> uncertain_human_review
    v_border = VisionResult(
        image_id="borderline_cable", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.55, raw_anomaly_score=31.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep_border = ReportGenerator().generate("borderline_cable", "defect?", v_border, None, [], [])
    assert "categorized_findings" in rep_border.model_dump()
    assert len(rep_border.categorized_findings["uncertain_human_review"]) >= 1

    # 2. High-margin anomalous sample -> confirmed_visual_defects
    v_confirmed = VisionResult(
        image_id="severed_cable", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.95, raw_anomaly_score=45.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        notes=["Specialized detector 'conductor_continuity_detector' detected conductor continuity failure in left."],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep_conf = ReportGenerator().generate("severed_cable", "defect?", v_confirmed, None, [], [])
    assert len(rep_conf.categorized_findings["confirmed_visual_defects"]) >= 1
    assert any("conductor_continuity_detector" in s for s in rep_conf.categorized_findings["confirmed_visual_defects"])

    # 3. Moderate margin sample -> potential_anomalies
    v_pot = VisionResult(
        image_id="minor_cable", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.60, raw_anomaly_score=34.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep_pot = ReportGenerator().generate("minor_cable", "defect?", v_pot, None, [], [])
    assert len(rep_pot.categorized_findings["potential_anomalies"]) >= 1


def test_category_mutual_exclusivity_and_non_overlap(tmp_workspace):
    """Verify that findings are strictly mutually exclusive and non-overlapping."""
    v = VisionResult(
        image_id="cable_test_exclusivity", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.90, raw_anomaly_score=42.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        notes=["Specialized detector 'conductor_continuity_detector' detected conductor continuity failure in left."],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep = ReportGenerator().generate("cable_test_exclusivity", "defect?", v, None, [], [])

    # Verify structured findings exist
    assert len(rep.structured_findings) > 0
    # Collect IDs in each category
    by_cat = {}
    for sf in rep.structured_findings:
        by_cat.setdefault(sf.category, set()).add(sf.id)

    # Assert pairwise disjoint
    categories = list(by_cat.keys())
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            c1, c2 = categories[i], categories[j]
            assert by_cat[c1].isdisjoint(by_cat[c2]), f"Categories {c1} and {c2} overlap!"


def test_detector_disagreement_handling(tmp_workspace):
    """Verify that when detectors disagree (PaDiM normal vs Puncture detector), it routes to uncertain human review."""
    v_disagree = VisionResult(
        image_id="cable_disagree", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.20, raw_anomaly_score=26.5,
        is_anomalous=False, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        notes=["Specialized detector 'surface_puncture_detector' detected surface puncture (score 39.20 >= 38.00)."],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep = ReportGenerator().generate("cable_disagree", "defect?", v_disagree, None, [], [])

    # The puncture disagreement must be in uncertain_human_review, NOT a candidate defect
    unc_findings = [f for f in rep.structured_findings if f.category == "uncertain_human_review"]
    assert len(unc_findings) >= 1
    punc_finding = next(f for f in unc_findings if f.provenance.detector == "SurfacePunctureDetector")
    assert "Detector Disagreement" in punc_finding.title
    assert punc_finding.action_required is not None
    assert "Action Required" in punc_finding.action_required
    assert punc_finding.is_human_confirmed is False


def test_duplicate_findings_prevention(tmp_workspace):
    """Verify that duplicate notes or evidence items are never duplicated in structured findings."""
    v = VisionResult(
        image_id="cable_dup", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.90, raw_anomaly_score=42.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        notes=[
            "Specialized detector 'conductor_continuity_detector' detected conductor continuity failure in left.",
            "Specialized detector 'conductor_continuity_detector' detected conductor continuity failure in left.",
        ],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    evidence = [
        EvidenceItem(kind=EvidenceKind.VISUAL_OBSERVATION, statement="Duplicate observation statement"),
        EvidenceItem(kind=EvidenceKind.VISUAL_OBSERVATION, statement="Duplicate observation statement"),
    ]
    rep = ReportGenerator().generate("cable_dup", "defect?", v, None, evidence, [])

    # Verify no duplicate finding IDs
    ids = [sf.id for sf in rep.structured_findings]
    assert len(ids) == len(set(ids)), f"Duplicate finding IDs found: {ids}"


def test_evidence_provenance_and_human_unconfirmed(tmp_workspace):
    """Verify every finding retains detector provenance and is never labeled confirmed physical defect."""
    v = VisionResult(
        image_id="cable_prov", category="cable", model_name="PaDiM", backbone="resnet18",
        image_anomaly_score=0.90, raw_anomaly_score=42.0,
        is_anomalous=True, threshold=30.42, verdict_mode="calibrated_raw", regions=[],
        notes=["Specialized detector 'conductor_continuity_detector' detected conductor continuity failure in left."],
        quality=ImageQuality(is_valid=True, width=64, height=64))
    rep = ReportGenerator().generate("cable_prov", "defect?", v, None, [], [])

    for sf in rep.structured_findings:
        # Check provenance
        assert sf.provenance is not None
        assert sf.provenance.detector != ""
        # Check distinction: algorithmic prediction is NOT human-confirmed
        assert sf.is_human_confirmed is False
        assert "Confirmed Physical Defect" not in sf.title


