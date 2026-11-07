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
