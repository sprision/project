"""Analyze detector tradeoffs and calibration data distribution."""
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths
from src.app.vision.model import PaDiM, AnomalyInspector, load_calibrated_threshold
from src.app.vision.cable_inspector import ConductorContinuityDetector, SurfacePunctureDetector

raw_root = Path("data/raw/mvtec_ad/cable")
paths = collect_image_paths(raw_root / "test")
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

cable_model_path = Path("data/index/vision/cable/padim.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaDiM.load(cable_model_path, device=device)
threshold = load_calibrated_threshold(cable_model_path)
base_inspector = AnomalyInspector(model, "cable", calibrated_threshold=threshold)

cont_det = ConductorContinuityDetector(threshold=0.50)
punc_det = SurfacePunctureDetector(model, threshold=38.0)

print(f"Loaded model on {device}, threshold={threshold}", flush=True)
print(f"Calibration Split: {len(cal_paths)} images (Good: {sum(1 for g, _ in cal_paths if g)}, Defect: {sum(1 for g, _ in cal_paths if not g)})\n", flush=True)

# 1. Evaluate clean calibration cables
print("=== CALIBRATION SPLIT: CLEAN SAMPLES (29) ===", flush=True)
clean_cal = [p for is_good, p in cal_paths if is_good]
clean_scores = []
for p in clean_cal:
    img = Image.open(p).convert("RGB")
    base_res = base_inspector.inspect(p)
    cont_ev = cont_det.inspect(img)
    punc_ev = punc_det.inspect(img)
    rec = {
        "name": p.name,
        "padim_raw": base_res.raw_anomaly_score,
        "padim_anom": base_res.is_anomalous,
        "cont_score": cont_ev.score,
        "cont_anom": cont_ev.is_defect_detected,
        "punc_score": punc_ev.score,
        "punc_anom": punc_ev.is_defect_detected,
    }
    clean_scores.append(rec)
    status = []
    if base_res.is_anomalous: status.append("PaDiM")
    if cont_ev.is_defect_detected: status.append("Cont")
    if punc_ev.is_defect_detected: status.append("Punc")
    if status:
        print(f"  [FP on Calib Clean] {p.name}: Triggered by {', '.join(status)} | PaDiM={base_res.raw_anomaly_score:.2f} (T={threshold:.2f}) | Cont={cont_ev.score:.3f} | Punc={punc_ev.score:.2f}", flush=True)

# 2. Evaluate defective calibration cables
print("\n=== CALIBRATION SPLIT: DEFECTIVE SAMPLES (46) ===", flush=True)
defect_cal = [p for is_good, p in cal_paths if not is_good]
defect_scores = []
for p in defect_cal:
    img = Image.open(p).convert("RGB")
    base_res = base_inspector.inspect(p)
    cont_ev = cont_det.inspect(img)
    punc_ev = punc_det.inspect(img)
    defect_type = p.parent.name
    rec = {
        "name": p.name,
        "type": defect_type,
        "padim_raw": base_res.raw_anomaly_score,
        "padim_anom": base_res.is_anomalous,
        "cont_score": cont_ev.score,
        "cont_anom": cont_ev.is_defect_detected,
        "punc_score": punc_ev.score,
        "punc_anom": punc_ev.is_defect_detected,
    }
    defect_scores.append(rec)

# Defect breakdown on calibration
from collections import defaultdict
by_type = defaultdict(list)
for d in defect_scores:
    by_type[d["type"]].append(d)

print(f"{'Defect Type':24s} {'Count':6s} {'PaDiM TP':10s} {'Cont TP':10s} {'Punc TP':10s} {'Combined TP':12s}", flush=True)
print("-" * 75, flush=True)
for dt, items in sorted(by_type.items()):
    n = len(items)
    padim_tp = sum(1 for x in items if x["padim_anom"])
    cont_tp = sum(1 for x in items if x["cont_anom"])
    punc_tp = sum(1 for x in items if x["punc_anom"])
    comb_tp = sum(1 for x in items if x["padim_anom"] or x["cont_anom"] or x["punc_anom"])
    print(f"{dt:24s} {n:<6d} {padim_tp:<10d} {cont_tp:<10d} {punc_tp:<10d} {comb_tp:<12d}", flush=True)

print("\n--- Summary Distribution on Calibration Clean Cables ---", flush=True)
padim_clean = [x["padim_raw"] for x in clean_scores]
cont_clean = [x["cont_score"] for x in clean_scores]
punc_clean = [x["punc_score"] for x in clean_scores]
print(f"PaDiM clean: min={min(padim_clean):.2f}, max={max(padim_clean):.2f}, mean={np.mean(padim_clean):.2f}, P95={np.percentile(padim_clean, 95):.2f}", flush=True)
print(f"Cont clean:  min={min(cont_clean):.3f}, max={max(cont_clean):.3f}, mean={np.mean(cont_clean):.3f}, P95={np.percentile(cont_clean, 95):.3f}", flush=True)
print(f"Punc clean:  min={min(punc_clean):.2f}, max={max(punc_clean):.2f}, mean={np.mean(punc_clean):.2f}, P95={np.percentile(punc_clean, 95):.2f}", flush=True)

# Also check poke_insulation scores on calibration
poke_cal = [x for x in defect_scores if x["type"] == "poke_insulation"]
print(f"\nPoke insulation calibration scores: {[round(x['punc_score'], 2) for x in poke_cal]}", flush=True)
# Also check missing_wire scores on calibration
wire_cal = [x for x in defect_scores if x["type"] == "missing_wire"]
print(f"Missing wire calibration scores: {[round(x['cont_score'], 3) for x in wire_cal]}", flush=True)

