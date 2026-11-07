"""Comprehensive evaluation of Specialized Cable Inspector vs Production Baseline PaDiM.

Evaluates on identical calibration and held-out evaluation splits (seed 42, 50/50).
Reports defect-specific recall, confusion matrix, sensitivity, specificity, FPR,
balanced accuracy, inference latency, and memory footprint.
"""
import json
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.vision.model import PaDiM, AnomalyInspector, load_calibrated_threshold
from src.app.vision.cable_inspector import SpecializedCableInspector
from src.app.vision.registry import get_process_memory_mb
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths

raw_root = Path("data/raw/mvtec_ad/cable")
paths = collect_image_paths(raw_root / "test")
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

cable_model_path = Path("data/index/vision/cable/padim.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaDiM.load(cable_model_path, device=device)
threshold = load_calibrated_threshold(cable_model_path)

base_inspector = AnomalyInspector(model, "cable", calibrated_threshold=threshold)
specialized_inspector = SpecializedCableInspector(base_inspector, continuity_threshold=0.50, puncture_threshold=38.0)

def evaluate_detector(inspector, target_paths, is_specialized=False):
    t0 = time.time()
    records = []
    
    for is_good, p in target_paths:
        t_start = time.time()
        if is_specialized:
            vis_res, spec_ev = inspector.inspect(p)
            ev_list = [{"detector": e.detector_name, "detected": e.is_defect_detected, "type": e.finding_type} for e in spec_ev]
        else:
            vis_res = inspector.inspect(p)
            ev_list = []
            
        elapsed_img_ms = (time.time() - t_start) * 1000.0
        records.append({
            "path": str(p),
            "filename": p.name,
            "defect_type": p.parent.name,
            "is_good": is_good,
            "true_label": 0 if is_good else 1,
            "pred_anomalous": vis_res.is_anomalous,
            "pred_label": 1 if vis_res.is_anomalous else 0,
            "raw_score": vis_res.raw_anomaly_score,
            "n_regions": len(vis_res.regions),
            "inference_ms": elapsed_img_ms,
            "specialized_evidence": ev_list,
        })
        
    total_sec = time.time() - t0
    y_true = np.array([r["true_label"] for r in records])
    y_pred = np.array([r["pred_label"] for r in records])
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    bal = 0.5 * (sens + spec)
    
    # Defect-type breakdown
    defects = {}
    for r in records:
        d = r["defect_type"]
        if d not in defects:
            defects[d] = {"count": 0, "detected": 0}
        defects[d]["count"] += 1
        if r["pred_label"] == r["true_label"]:
            defects[d]["detected"] += 1
            
    breakdown = {}
    for d, info in defects.items():
        breakdown[d] = {
            "count": info["count"],
            "correct": info["detected"],
            "recall": round(info["detected"] / info["count"], 4),
        }
        
    return {
        "n_images": len(records),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "fpr": round(fpr, 4),
        "balanced_accuracy": round(bal, 4),
        "total_elapsed_sec": round(total_sec, 2),
        "avg_ms_per_image": round(np.mean([r["inference_ms"] for r in records]), 2),
        "defect_breakdown": breakdown,
    }

def main():
    print("================================================================================")
    print("SPECIALIZED CABLE INSPECTION MODULE BENCHMARK EVALUATION")
    print("================================================================================")
    mem = get_process_memory_mb()
    print(f"Device: {device} | Active RAM: {mem['ram_working_set_mb']:.1f} MB | CUDA VRAM: {mem['cuda_allocated_mb']:.1f} MB\n")

    # 1. Baseline PaDiM on Held-Out Split
    print("--- Evaluating Baseline PaDiM on Strict Held-Out Split (75 images) ---")
    base_res = evaluate_detector(base_inspector, ev_paths, is_specialized=False)

    # 2. Specialized Cable Inspector on Held-Out Split
    print("--- Evaluating Specialized Cable Inspector on Strict Held-Out Split (75 images) ---")
    spec_res = evaluate_detector(specialized_inspector, ev_paths, is_specialized=True)

    print("\n" + "=" * 80)
    print("STRICT HELD-OUT PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metric':30s} {'Baseline PaDiM':20s} {'Specialized Inspector':20s} {'Delta':10s}")
    print("-" * 80)
    print(f"{'Total Evaluated Images':30s} {base_res['n_images']:<20d} {spec_res['n_images']:<20d}")
    b_cm = f"{base_res['tp']}/{base_res['fp']}/{base_res['tn']}/{base_res['fn']}"
    s_cm = f"{spec_res['tp']}/{spec_res['fp']}/{spec_res['tn']}/{spec_res['fn']}"
    print(f"{'Confusion Matrix (TP/FP/TN/FN)':30s} {b_cm:<20s} {s_cm:<20s}")
    print(f"{'Overall Sensitivity (Recall)':30s} {base_res['sensitivity']:<20.4f} {spec_res['sensitivity']:<20.4f} {spec_res['sensitivity'] - base_res['sensitivity']:+.4f}")
    print(f"{'Overall Specificity (Clean Recall)':30s} {base_res['specificity']:<20.4f} {spec_res['specificity']:<20.4f} {spec_res['specificity'] - base_res['specificity']:+.4f}")
    print(f"{'False Positive Rate (FPR)':30s} {base_res['fpr']:<20.4f} {spec_res['fpr']:<20.4f} {spec_res['fpr'] - base_res['fpr']:+.4f}")
    print(f"{'Balanced Accuracy':30s} {base_res['balanced_accuracy']:<20.4f} {spec_res['balanced_accuracy']:<20.4f} {spec_res['balanced_accuracy'] - base_res['balanced_accuracy']:+.4f}")
    b_lat = f"{base_res['avg_ms_per_image']} ms"
    s_lat = f"{spec_res['avg_ms_per_image']} ms"
    print(f"{'Avg Latency per Image':30s} {b_lat:<20s} {s_lat:<20s}")

    print("\n" + "-" * 80)
    print("DEFECT-SPECIFIC RECALL COMPARISON (HELD-OUT EVALUATION)")
    print("-" * 80)
    print(f"{'Defect Type':24s} {'Count':6s} {'Baseline Recall':20s} {'Specialized Recall':20s} {'Gain':10s}")
    print("-" * 80)
    for d, b_info in sorted(base_res["defect_breakdown"].items()):
        s_info = spec_res["defect_breakdown"][d]
        gain = s_info["recall"] - b_info["recall"]
        b_str = f"{b_info['correct']}/{b_info['count']} ({b_info['recall']*100:.1f}%)"
        s_str = f"{s_info['correct']}/{s_info['count']} ({s_info['recall']*100:.1f}%)"
        print(f"{d:24s} {b_info['count']:<6d} {b_str:<20s} {s_str:<20s} {gain*100:+.1f}%")

    out_file = Path("runs/cable_specialized_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "evaluation_protocol": "strict_held_out_evaluation",
        "calibration_seed": 42,
        "baseline_padim": base_res,
        "specialized_inspector": spec_res,
        "memory_mb": get_process_memory_mb(),
    }
    out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark records to {out_file}")

if __name__ == "__main__":
    main()
