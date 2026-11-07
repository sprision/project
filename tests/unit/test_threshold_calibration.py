"""Unit tests for calibrated-threshold loading, fallback behaviour and verdicts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.app.vision.model import AnomalyInspector, load_calibrated_threshold


class _StubPaDiM:
    """Stands in for PaDiM: anomaly_map -> constant map at raw_max."""

    backbone_name = "resnet18"

    def __init__(self, raw_max: float, lo: float = 0.0, hi: float = 100.0) -> None:
        self.normalisation_ = {"min": lo, "max": hi}
        self._raw_max = raw_max

    def anomaly_map(self, img) -> np.ndarray:
        return np.full((256, 256), float(self._raw_max), dtype=np.float32)


def _png(tmp_path: Path, name: str = "img.png") -> Path:
    p = tmp_path / name
    Image.new("RGB", (64, 64), (120, 120, 120)).save(p)
    return p


def _write_threshold(tmp_path: Path, value) -> None:
    (tmp_path / "threshold.json").write_text(
        json.dumps({"threshold_raw": value}), encoding="utf-8")


def test_load_calibrated_threshold_missing(tmp_path):
    assert load_calibrated_threshold(tmp_path / "padim.pkl") is None


def test_load_calibrated_threshold_reads_value(tmp_path):
    _write_threshold(tmp_path, 12.5)
    assert load_calibrated_threshold(tmp_path / "padim.pkl") == pytest.approx(12.5)


def test_load_calibrated_threshold_invalid(tmp_path):
    (tmp_path / "padim.pkl").write_bytes(b"x")
    (tmp_path / "threshold.json").write_text(
        json.dumps({"not_threshold": 1}), encoding="utf-8")
    assert load_calibrated_threshold(tmp_path / "padim.pkl") is None


def test_calibrated_verdict_above_threshold(tmp_path):
    insp = AnomalyInspector(_StubPaDiM(40.0), "bottle", calibrated_threshold=20.0)
    res = insp.inspect(_png(tmp_path))
    assert res.verdict_mode == "calibrated_raw"
    assert res.is_anomalous is True
    assert res.threshold == pytest.approx(20.0)
    assert res.raw_anomaly_score == pytest.approx(40.0)
    # normalised score is display-only: (40 - 0) / (100 - 0) = 0.4
    assert res.image_anomaly_score == pytest.approx(0.4)


def test_calibrated_verdict_below_threshold(tmp_path):
    insp = AnomalyInspector(_StubPaDiM(40.0), "bottle", calibrated_threshold=100.0)
    res = insp.inspect(_png(tmp_path))
    assert res.verdict_mode == "calibrated_raw"
    assert res.is_anomalous is False
    assert res.threshold == pytest.approx(100.0)
    assert res.regions == []


def test_fallback_normalised_verdict(tmp_path):
    insp = AnomalyInspector(_StubPaDiM(40.0, lo=0.0, hi=100.0), "bottle",
                             threshold=0.5)
    res = insp.inspect(_png(tmp_path))
    assert res.verdict_mode == "normalised"
    assert res.is_anomalous is False
    assert res.threshold == pytest.approx(0.5)


def test_calibrated_threshold_files_exist_and_positive():
    """Verify that all production categories have calibrated raw thresholds."""
    vdir = Path("data/index/vision")
    for cat in ["bottle", "cable", "metal_nut", "screw"]:
        tpath = vdir / cat / "threshold.json"
        assert tpath.exists(), f"threshold.json missing for {cat}"
        data = json.loads(tpath.read_text(encoding="utf-8"))
        assert "threshold_raw" in data
        assert isinstance(data["threshold_raw"], (int, float))
        assert data["threshold_raw"] > 0, f"threshold must be positive for {cat}"
        assert data["category"] == cat


def test_metrics_calculation_known_values():
    """Validate sensitivity, specificity, balanced accuracy, and FPR arithmetic."""
    from scripts.calibrate_thresholds import metrics
    # y: 1=defect, 0=good. 3 TP, 1 FN, 4 TN, 2 FP
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    m = metrics(y, pred)
    assert m["tp"] == 3
    assert m["fn"] == 1
    assert m["tn"] == 4
    assert m["fp"] == 2
    assert m["sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(4 / 6)
    assert m["false_positive_rate"] == pytest.approx(2 / 6)
    assert m["balanced_accuracy"] == pytest.approx(0.5 * (3 / 4 + 4 / 6))


def test_calibration_split_stratified_disjoint_and_reproducible():
    """Verify that calibration and evaluation splits are disjoint and seeded reproducibility holds."""
    from scripts.calibrate_thresholds import split_stratified
    # Create 20 unique labeled rows
    rows = [(True, float(i)) for i in range(10)] + [(False, float(i + 100)) for i in range(10)]
    cal1, ev1 = split_stratified(rows, seed=42, frac=0.5)
    cal2, ev2 = split_stratified(rows, seed=42, frac=0.5)

    # Disjointness
    assert len(cal1) + len(ev1) == len(rows)
    assert set(cal1).isdisjoint(set(ev1)), "Calibration and evaluation splits must be disjoint!"

    # Stratification: equal proportion of good and defect
    assert sum(1 for g, _ in cal1 if g) == 5
    assert sum(1 for g, _ in cal1 if not g) == 5
    assert sum(1 for g, _ in ev1 if g) == 5
    assert sum(1 for g, _ in ev1 if not g) == 5

    # Reproducibility under same seed
    assert cal1 == cal2
    assert ev1 == ev2

    # Different seeds yield different permutations
    cal3, _ = split_stratified(rows, seed=999, frac=0.5)
    assert cal1 != cal3


def test_pick_threshold_maximizes_balanced_accuracy():
    """Verify that pick_threshold selects a cut-off maximizing balanced accuracy."""
    from scripts.calibrate_thresholds import pick_threshold, metrics
    # Scores for good (0): 10, 12, 14, 16. Scores for defect (1): 18, 20, 22, 24.
    s = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    t = pick_threshold(s, y)
    assert 16.0 < t < 18.0, f"Expected threshold separating good and defect, got {t}"
    m = metrics(y, (s >= t).astype(int))
    assert m["balanced_accuracy"] == 1.0


