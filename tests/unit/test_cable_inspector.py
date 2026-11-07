"""Unit tests for the Specialized Cable Inspection Module."""
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from src.app.vision.cable_inspector import (
    ConductorContinuityDetector,
    SurfacePunctureDetector,
    SpecializedCableInspector,
    CableSpecializedEvidence,
)
from src.app.vision.model import AnomalyInspector, PaDiM
from src.app.schemas import VisionResult, DefectRegion, ImageQuality


def test_conductor_continuity_detector_initialization():
    det = ConductorContinuityDetector(threshold=0.50)
    assert det.threshold == 0.50
    assert len(det.wire_rois) == 3
    assert "top" in det.wire_rois
    assert "left" in det.wire_rois
    assert "right" in det.wire_rois


def test_conductor_continuity_detector_evaluation():
    det = ConductorContinuityDetector(threshold=0.50)
    # Test on a uniform dark image (simulating missing wires / empty harness)
    black_img = Image.new("RGB", (1024, 1024), color=(10, 10, 10))
    ev = det.inspect(black_img)
    assert isinstance(ev, CableSpecializedEvidence)
    assert ev.detector_name == "conductor_continuity_detector"
    assert ev.is_defect_detected is True  # Density is 0, so defect score is 1.0 >= 0.50
    assert ev.finding_type == "confirmed_visual_defect"
    assert ev.score == 1.0


def test_surface_puncture_detector_initialization():
    det = SurfacePunctureDetector(threshold=38.0)
    assert det.threshold == 38.0


def test_surface_puncture_detector_evaluation():
    det = SurfacePunctureDetector(threshold=38.0)
    # Test on a uniform image (no high frequency puncture)
    gray_img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
    ev = det.inspect(gray_img)
    assert isinstance(ev, CableSpecializedEvidence)
    assert ev.detector_name == "surface_puncture_detector"
    assert ev.is_defect_detected is False
    assert ev.score == 0.0


class DummyBaseInspector:
    """Mock base inspector for deterministic composite inspection testing."""
    def __init__(self, is_anomalous=False, score=25.0, threshold=30.0):
        self.category = "cable"
        self.calibrated_threshold = threshold
        self.is_anomalous = is_anomalous
        self.score = score

    def inspect(self, image_input):
        return VisionResult(
            image_id="test_cable",
            category="cable",
            model_name="PaDiM",
            backbone="resnet18",
            image_anomaly_score=self.score / 100.0,
            raw_anomaly_score=self.score,
            is_anomalous=self.is_anomalous,
            threshold=self.calibrated_threshold,
            verdict_mode="calibrated_raw",
            regions=[],
            heatmap_path=None,
            inference_ms=10.0,
            quality=ImageQuality(is_valid=True, width=1024, height=1024),
            notes=[],
        )


def test_specialized_cable_inspector_clean_composite():
    base = DummyBaseInspector(is_anomalous=False, score=24.0, threshold=30.0)
    inspector = SpecializedCableInspector(base, continuity_threshold=0.50, puncture_threshold=38.0)
    
    # Using a synthetic image with high brightness in the wire ROIs to simulate clean conductors
    img_arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Fill harness regions with texture/high values
    img_arr[240:420, 410:600] = 200
    img_arr[490:690, 240:450] = 200
    img_arr[490:710, 580:770] = 200
    clean_img = Image.fromarray(img_arr)
    
    vis_res, evidence = inspector.inspect(clean_img)
    assert isinstance(vis_res, VisionResult)
    assert len(evidence) == 2
    assert vis_res.is_anomalous is False
    assert all(e.finding_type in ("normal", "potential_anomaly", "uncertain_human_review") for e in evidence)


def test_specialized_cable_inspector_defect_augmentation():
    base = DummyBaseInspector(is_anomalous=False, score=24.0, threshold=30.0)
    inspector = SpecializedCableInspector(base, continuity_threshold=0.50, puncture_threshold=38.0)
    
    # Black image will trigger conductor continuity failure
    black_img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    vis_res, evidence = inspector.inspect(black_img)
    
    assert vis_res.is_anomalous is True
    assert len(vis_res.regions) >= 1
    # Check that conductor continuity was detected as confirmed visual defect
    cont_ev = next(e for e in evidence if e.detector_name == "conductor_continuity_detector")
    assert cont_ev.is_defect_detected is True
    assert cont_ev.finding_type == "confirmed_visual_defect"
