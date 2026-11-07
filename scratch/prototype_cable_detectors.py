"""Prototype and evaluate specialized cable inspection detectors.

Experiments on:
1. Conductor continuity / wire completeness detection (for missing_wire).
2. High-resolution patch/gradient inspection (for poke_insulation).
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths

raw_root = Path("data/raw/mvtec_ad/cable")
test_dir = raw_root / "test"
paths = collect_image_paths(test_dir)
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

print(f"Calibration split: {len(cal_paths)} images")
print(f"Held-out evaluation split: {len(ev_paths)} images")

# Inspect calibration defect breakdown
cal_defects = {}
for g, p in cal_paths:
    d = p.parent.name
    cal_defects[d] = cal_defects.get(d, 0) + 1
print("\nCalibration split defect counts:", cal_defects)

ev_defects = {}
for g, p in ev_paths:
    d = p.parent.name
    ev_defects[d] = ev_defects.get(d, 0) + 1
print("Held-out split defect counts:", ev_defects)

