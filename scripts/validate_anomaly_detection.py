"""Anomaly Detection Validation Script.

Validates PaDiM anomaly detection reliability on the MVTec AD test sets
(bottle, cable, metal_nut) across all known-good and known-defective samples.
Uses the existing fitted models and calibrated thresholds.
Does NOT modify or overwrite existing model files or benchmark files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.app.config import get_settings
from src.app.vision.model import AnomalyInspector, PaDiM, load_calibrated_threshold


def run_validation(raw_root: Path, categories: list[str]) -> dict:
    settings = get_settings()
    device = settings.resolved_device()
    vision_dir = settings.index_dir / "vision"

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "categories": {},
    }

    for cat in categories:
        cat_start = time.time()
        model_path = vision_dir / cat / "padim.pkl"
        if not model_path.exists():
            print(f"[-] Model not found for {cat}: {model_path}")
            continue

        model = PaDiM.load(model_path, device=device)
        threshold = load_calibrated_threshold(model_path)
        inspector = AnomalyInspector(model, cat, calibrated_threshold=threshold)

        test_dir = raw_root / cat / "test"
        sample_records = []

        # Iterate over all test folders (good and defect types)
        for folder in sorted(test_dir.iterdir()):
            if not folder.is_dir():
                continue
            is_good = (folder.name == "good")
            true_label = 0 if is_good else 1
            defect_type = folder.name

            for img_path in sorted(folder.glob("*.png")):
                # Perform inspection using the AnomalyInspector
                vision_res = inspector.inspect(img_path)

                pred_label = 1 if vision_res.is_anomalous else 0
                sample_records.append({
                    "path": str(img_path.relative_to(raw_root)),
                    "filename": img_path.name,
                    "defect_type": defect_type,
                    "is_good": is_good,
                    "true_label": true_label,
                    "raw_score": vision_res.raw_anomaly_score,
                    "norm_score": vision_res.image_anomaly_score,
                    "threshold": vision_res.threshold,
                    "verdict_mode": vision_res.verdict_mode,
                    "pred_anomalous": vision_res.is_anomalous,
                    "pred_label": pred_label,
                    "n_regions": len(vision_res.regions),
                    "inference_ms": vision_res.inference_ms,
                })

        y_true = np.array([r["true_label"] for r in sample_records])
        y_pred = np.array([r["pred_label"] for r in sample_records])
        y_scores = np.array([r["raw_score"] for r in sample_records])

        # Confusion matrix elements
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        balanced_accuracy = 0.5 * (sensitivity + specificity)
        auroc = float(roc_auc_score(y_true, y_scores))

        # Categorize false positives and false negatives
        false_positives = [
            r for r in sample_records if r["true_label"] == 0 and r["pred_label"] == 1
        ]
        false_negatives = [
            r for r in sample_records if r["true_label"] == 1 and r["pred_label"] == 0
        ]

        # Score distributions by defect type and good
        defect_type_stats = {}
        for folder in sorted(test_dir.iterdir()):
            if not folder.is_dir():
                continue
            dname = folder.name
            d_records = [r for r in sample_records if r["defect_type"] == dname]
            d_scores = [r["raw_score"] for r in d_records]
            defect_type_stats[dname] = {
                "count": len(d_records),
                "min": round(float(np.min(d_scores)), 4),
                "mean": round(float(np.mean(d_scores)), 4),
                "max": round(float(np.max(d_scores)), 4),
                "std": round(float(np.std(d_scores)), 4),
                "misclassified": sum(1 for r in d_records if r["pred_label"] != r["true_label"]),
            }

        cat_elapsed = round(time.time() - cat_start, 2)

        results["categories"][cat] = {
            "threshold": threshold,
            "n_total": len(sample_records),
            "n_good": int((y_true == 0).sum()),
            "n_defective": int((y_true == 1).sum()),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
            "metrics": {
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "balanced_accuracy": round(balanced_accuracy, 4),
                "auroc": round(auroc, 4),
            },
            "defect_type_stats": defect_type_stats,
            "false_positives": [
                {
                    "path": fp_item["path"],
                    "raw_score": fp_item["raw_score"],
                    "threshold": fp_item["threshold"],
                    "delta": round(fp_item["raw_score"] - fp_item["threshold"], 4),
                }
                for fp_item in false_positives
            ],
            "false_negatives": [
                {
                    "path": fn_item["path"],
                    "defect_type": fn_item["defect_type"],
                    "raw_score": fn_item["raw_score"],
                    "threshold": fn_item["threshold"],
                    "delta": round(fn_item["threshold"] - fn_item["raw_score"], 4),
                }
                for fn_item in false_negatives
            ],
            "elapsed_seconds": cat_elapsed,
        }

    return results


def print_summary(results: dict):
    print("\n" + "=" * 80)
    print("ANOMALY DETECTION RELIABILITY VALIDATION REPORT")
    print("=" * 80)
    print(f"Device: {results['device']} | Timestamp: {results['timestamp']}\n")

    for cat, data in results["categories"].items():
        cm = data["confusion_matrix"]
        m = data["metrics"]
        print(f"Category: {cat.upper()} (Threshold: {data['threshold']:.4f})")
        print(f"  Total Images: {data['n_total']} (Good: {data['n_good']}, Defective: {data['n_defective']})")
        print(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
        print(f"  Sensitivity (Recall): {m['sensitivity']:.4f}")
        print(f"  Specificity:          {m['specificity']:.4f}")
        print(f"  False Positive Rate:  {m['false_positive_rate']:.4f}")
        print(f"  False Negative Rate:  {m['false_negative_rate']:.4f}")
        print(f"  Balanced Accuracy:    {m['balanced_accuracy']:.4f}")
        print(f"  AUROC (Image-level):  {m['auroc']:.4f}")
        print(f"  Elapsed Time:         {data['elapsed_seconds']}s")

        if data["false_positives"]:
            print(f"  [!] False Positives ({len(data['false_positives'])}):")
            for fp_item in data["false_positives"]:
                print(f"      - {fp_item['path']}: score={fp_item['raw_score']:.4f} (exceeds threshold by {fp_item['delta']:+.4f})")
        else:
            print("  [+] False Positives: 0 (No clean samples flagged as anomalous)")

        if data["false_negatives"]:
            print(f"  [!] False Negatives ({len(data['false_negatives'])}):")
            for fn_item in data["false_negatives"]:
                print(f"      - {fn_item['path']} ({fn_item['defect_type']}): score={fn_item['raw_score']:.4f} (below threshold by {fn_item['delta']:+.4f})")
        else:
            print("  [+] False Negatives: 0 (All defective samples detected)")

        print("  Breakdown by Subdirectory:")
        for dt, s in data["defect_type_stats"].items():
            print(f"      {dt:22s} count={s['count']:2d} | range=[{s['min']:6.2f}, {s['max']:6.2f}] mean={s['mean']:6.2f} | errors={s['misclassified']}")
        print("-" * 80)


if __name__ == "__main__":
    raw_root = Path("data/raw/mvtec_ad")
    categories = ["bottle", "cable", "metal_nut"]
    res = run_validation(raw_root, categories)
    print_summary(res)

    out_file = Path("runs/validation_results.json")
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved full validation records to {out_file}")

