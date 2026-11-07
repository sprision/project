"""Comprehensive evaluation of Consensus & Boundary-Masked Cable Inspector vs Previous Approaches.

Evaluates strictly on the untouched held-out evaluation split (75 images, seed 42, 50/50).
Compares:
1. Baseline Production PaDiM
2. Previous Specialized Inspector (Naive OR)
3. Consensus & Boundary-Masked Specialized Inspector

Saves output to runs/cable_consensus_evaluation.json.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths
from src.app.vision.model import PaDiM, AnomalyInspector, load_calibrated_threshold
from src.app.vision.cable_inspector import SpecializedCableInspector
from src.app.vision.registry import get_process_memory_mb

raw_root = Path("data/raw/mvtec_ad/cable")
paths = collect_image_paths(raw_root / "test")
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

cable_model_path = Path("data/index/vision/cable/padim.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaDiM.load(cable_model_path, device=device)
threshold = load_calibrated_threshold(cable_model_path)

base_inspector = AnomalyInspector(model, "cable", calibrated_threshold=threshold)
previous_inspector = SpecializedCableInspector(base_inspector, continuity_threshold=0.50, puncture_threshold=38.0, consensus_mode=False)
consensus_inspector = SpecializedCableInspector(base_inspector, continuity_threshold=0.50, puncture_threshold=38.0, puncture_high_threshold=42.0, consensus_mode=True)


def evaluate_detector(inspector, target_paths, name="Detector"):
    t0 = time.time()
    records = []
    
    for is_good, p in target_paths:
        t_start = time.time()
        vis_res, spec_ev = inspector.inspect(p) if hasattr(inspector, "puncture_detector") else (inspector.inspect(p), [])
        elapsed_img_ms = (time.time() - t_start) * 1000.0
        
        ev_list = [{"detector": e.detector_name, "detected": e.is_defect_detected, "type": e.finding_type} for e in spec_ev]
        records.append({
            "path": str(p),
            "filename": p.name,
            "defect_type": p.parent.name,
            "is_good": is_good,
            "true_label": 0 if is_good else 1,
            "pred_anomalous": vis_res.is_anomalous,
            "pred_label": 1 if vis_res.is_anomalous else 0,
            "raw_score": float(vis_res.raw_anomaly_score),
            "n_regions": len(vis_res.regions),
            "inference_ms": elapsed_img_ms,
            "specialized_evidence": ev_list,
        })

    total_sec = time.time() - t0
    tp = sum(1 for r in records if not r["is_good"] and r["pred_anomalous"])
    fn = sum(1 for r in records if not r["is_good"] and not r["pred_anomalous"])
    tn = sum(1 for r in records if r["is_good"] and not r["pred_anomalous"])
    fp = sum(1 for r in records if r["is_good"] and r["pred_anomalous"])

    n_pos = tp + fn
    n_neg = tn + fp
    sens = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    bal_acc = (sens + spec) / 2.0

    breakdown = {}
    by_type = defaultdict(list)
    for r in records:
        by_type[r["defect_type"]].append(r)

    for dt, items in by_type.items():
        n = len(items)
        if dt == "good":
            correct = sum(1 for r in items if not r["pred_anomalous"])
        else:
            correct = sum(1 for r in items if r["pred_anomalous"])
        breakdown[dt] = {
            "count": n,
            "correct": correct,
            "recall": round(correct / n, 4) if n > 0 else 0.0,
        }

    return {
        "name": name,
        "n_images": len(records),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "fpr": round(fpr, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "total_elapsed_sec": round(total_sec, 2),
        "avg_ms_per_image": round(float(np.mean([r["inference_ms"] for r in records])), 2),
        "defect_breakdown": breakdown,
        "records": records,
    }


def main():
    print("=" * 90, flush=True)
    print("STRICT HELD-OUT EVALUATION: CABLE INSPECTION COMPARATIVE VALIDATION", flush=True)
    print("=" * 90, flush=True)
    mem = get_process_memory_mb()
    print(f"Device: {device} | Active RAM: {mem['ram_working_set_mb']:.1f} MB | CUDA VRAM: {mem['cuda_allocated_mb']:.1f} MB\n", flush=True)

    print("--- 1. Evaluating Production Baseline PaDiM on Held-Out Split (75 images) ---", flush=True)
    base_res = evaluate_detector(base_inspector, ev_paths, name="Baseline PaDiM")

    print("--- 2. Evaluating Previous Specialized Inspector (Naive OR) on Held-Out Split ---", flush=True)
    prev_res = evaluate_detector(previous_inspector, ev_paths, name="Previous (Naive OR)")

    print("--- 3. Evaluating Consensus & Boundary-Masked Inspector on Held-Out Split ---", flush=True)
    cons_res = evaluate_detector(consensus_inspector, ev_paths, name="Consensus & Masked")

    print("\n" + "=" * 90, flush=True)
    print(f"{'Metric':32s} {'Baseline PaDiM':18s} {'Previous (Naive OR)':20s} {'Consensus & Masked':20s}", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Total Evaluated Images':32s} {base_res['n_images']:<18d} {prev_res['n_images']:<20d} {cons_res['n_images']:<20d}", flush=True)
    b_cm = f"{base_res['tp']}/{base_res['fp']}/{base_res['tn']}/{base_res['fn']}"
    p_cm = f"{prev_res['tp']}/{prev_res['fp']}/{prev_res['tn']}/{prev_res['fn']}"
    c_cm = f"{cons_res['tp']}/{cons_res['fp']}/{cons_res['tn']}/{cons_res['fn']}"
    print(f"{'Confusion Matrix (TP/FP/TN/FN)':32s} {b_cm:<18s} {p_cm:<20s} {c_cm:<20s}", flush=True)
    print(f"{'Overall Sensitivity (Recall)':32s} {base_res['sensitivity']:<18.4f} {prev_res['sensitivity']:<20.4f} {cons_res['sensitivity']:<20.4f}", flush=True)
    print(f"{'Overall Specificity (Clean Recall)':32s} {base_res['specificity']:<18.4f} {prev_res['specificity']:<20.4f} {cons_res['specificity']:<20.4f}", flush=True)
    print(f"{'False Positive Rate (FPR)':32s} {base_res['fpr']:<18.4f} {prev_res['fpr']:<20.4f} {cons_res['fpr']:<20.4f}", flush=True)
    print(f"{'Balanced Accuracy':32s} {base_res['balanced_accuracy']:<18.4f} {prev_res['balanced_accuracy']:<20.4f} {cons_res['balanced_accuracy']:<20.4f}", flush=True)
    b_lat = f"{base_res['avg_ms_per_image']} ms"
    p_lat = f"{prev_res['avg_ms_per_image']} ms"
    c_lat = f"{cons_res['avg_ms_per_image']} ms"
    print(f"{'Avg Latency per Image':32s} {b_lat:<18s} {p_lat:<20s} {c_lat:<20s}", flush=True)

    print("\n" + "-" * 90, flush=True)
    print("DEFECT-SPECIFIC RECALL COMPARISON (75 HELD-OUT IMAGES)", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Defect Type':24s} {'Count':6s} {'Baseline PaDiM':18s} {'Previous (Naive OR)':20s} {'Consensus & Masked':20s}", flush=True)
    print("-" * 90, flush=True)
    for dt in sorted(base_res["defect_breakdown"].keys()):
        b_info = base_res["defect_breakdown"][dt]
        p_info = prev_res["defect_breakdown"][dt]
        c_info = cons_res["defect_breakdown"][dt]
        b_str = f"{b_info['correct']}/{b_info['count']} ({b_info['recall']*100:.1f}%)"
        p_str = f"{p_info['correct']}/{p_info['count']} ({p_info['recall']*100:.1f}%)"
        c_str = f"{c_info['correct']}/{c_info['count']} ({c_info['recall']*100:.1f}%)"
        print(f"{dt:24s} {b_info['count']:<6d} {b_str:<18s} {p_str:<20s} {c_str:<20s}", flush=True)

    # Save benchmark file separately
    out_path = Path("runs/cable_consensus_evaluation.json")
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_protocol": "strict_held_out_evaluation",
        "calibration_seed": 42,
        "n_evaluated_samples": 75,
        "baseline_padim": {k: v for k, v in base_res.items() if k != "records"},
        "previous_specialized": {k: v for k, v in prev_res.items() if k != "records"},
        "consensus_specialized": {k: v for k, v in cons_res.items() if k != "records"},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved independent evaluation benchmark to {out_path}", flush=True)


if __name__ == "__main__":
    main()
