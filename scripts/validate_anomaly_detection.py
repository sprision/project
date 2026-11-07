"""Anomaly Detection Validation Script.

Validates PaDiM anomaly detection reliability on MVTec AD test sets
(bottle, cable, metal_nut, screw) across known-good and known-defective samples.
Uses the existing fitted models and calibrated thresholds.

Supports three evaluation protocols:
  1. held_out (default): Strictly evaluates untouched evaluation samples from the
     seeded split (seed 42, 50/50). Zero threshold-selection leakage.
     Outputs to runs/held_out_evaluation.json.
  2. full: Evaluates all test samples. Explicitly labeled as apparent /
     threshold-selection-contaminated because 50% of samples were in the calibration pool.
     Outputs to runs/validation_results.json.
  3. calibration: Evaluates only the calibration subset.

Does NOT modify or overwrite existing model files or calibration benchmark files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.config import get_settings
from src.app.vision.model import AnomalyInspector, PaDiM, load_calibrated_threshold

CALIBRATION_SEED = 42
SPLIT_FRACTION = 0.5


def collect_image_paths(cat_dir: Path) -> list[tuple[bool, Path]]:
    """Deterministically collect (is_good, image_path) sorted by directory and filename."""
    paths = []
    for dd in sorted(cat_dir.iterdir()):
        if not dd.is_dir():
            continue
        is_good = dd.name == "good"
        for p in sorted(dd.glob("*.png")):
            paths.append((is_good, p))
    return paths


def split_stratified_paths(paths: list[tuple[bool, Path]], seed: int = CALIBRATION_SEED, frac: float = SPLIT_FRACTION):
    """Partition paths into disjoint calibration and held-out evaluation subsets."""
    rng = np.random.default_rng(seed)
    cal, ev = [], []
    for is_good in (True, False):
        idx = [i for i, (g, _) in enumerate(paths) if g == is_good]
        rng.shuffle(idx)
        k = int(round(len(idx) * (1.0 - frac)))
        cal += [paths[i] for i in idx[:k]]
        ev += [paths[i] for i in idx[k:]]
    return cal, ev


def run_validation(raw_root: Path, categories: list[str], split_mode: str = "held_out") -> dict:
    settings = get_settings()
    device = settings.resolved_device()
    vision_dir = settings.index_dir / "vision"

    is_contaminated = (split_mode == "full")
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "evaluation_protocol": (
            "strict_held_out_evaluation" if split_mode == "held_out"
            else "threshold_selection_contaminated_apparent" if split_mode == "full"
            else "calibration_subset_evaluation"
        ),
        "split_mode": split_mode,
        "calibration_seed": CALIBRATION_SEED,
        "is_contaminated": is_contaminated,
        "contamination_warning": (
            "CAUTION: Full test evaluation includes samples used during threshold selection. "
            "Reported metrics are apparent and optimistic compared to held-out generalization."
            if is_contaminated else None
        ),
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
        all_paths = collect_image_paths(test_dir)
        cal_paths, ev_paths = split_stratified_paths(all_paths, seed=CALIBRATION_SEED, frac=SPLIT_FRACTION)

        if split_mode == "held_out":
            target_items = ev_paths
        elif split_mode == "calibration":
            target_items = cal_paths
        else:
            target_items = all_paths

        sample_records = []
        for is_good, img_path in target_items:
            true_label = 0 if is_good else 1
            defect_type = img_path.parent.name

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
        auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else float("nan")

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
            if not d_records:
                continue
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
            "split_mode": split_mode,
            "n_total": len(sample_records),
            "n_good": int((y_true == 0).sum()),
            "n_defective": int((y_true == 1).sum()),
            "split_composition": {
                "calibration_pool_total": len(cal_paths),
                "calibration_pool_good": sum(1 for g, _ in cal_paths if g),
                "calibration_pool_defect": sum(1 for g, _ in cal_paths if not g),
                "held_out_pool_total": len(ev_paths),
                "held_out_pool_good": sum(1 for g, _ in ev_paths if g),
                "held_out_pool_defect": sum(1 for g, _ in ev_paths if not g),
            },
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
    print(f"Protocol: {results['evaluation_protocol'].upper()}")
    print(f"Split Mode: {results['split_mode'].upper()} | Seed: {results['calibration_seed']}")
    if results.get("is_contaminated"):
        print(">>> WARNING: THRESHOLD-SELECTION-CONTAMINATED APPARENT METRICS <<<")
    else:
        print(">>> CERTIFIED STRICT HELD-OUT EVALUATION (LEAK-FREE) <<<")
    print("=" * 80)
    print(f"Device: {results['device']} | Timestamp: {results['timestamp']}\n")

    for cat, data in results["categories"].items():
        cm = data["confusion_matrix"]
        m = data["metrics"]
        sc = data["split_composition"]
        print(f"Category: {cat.upper()} (Threshold: {data['threshold']:.4f})")
        print(f"  Split Pool: Cal = {sc['calibration_pool_good']}G / {sc['calibration_pool_defect']}D | Held-out = {sc['held_out_pool_good']}G / {sc['held_out_pool_defect']}D")
        print(f"  Evaluated Images: {data['n_total']} (Good: {data['n_good']}, Defective: {data['n_defective']})")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", choices=["held_out", "full", "calibration"], default="held_out",
                    help="Evaluation split: 'held_out' (strict leak-free ev split), 'full' (all test samples, apparent/contaminated), 'calibration' (cal split)")
    ap.add_argument("--out-file", default=None, help="Custom output JSON path")
    args = ap.parse_args()

    s = get_settings()
    raw_root = Path("data/raw/mvtec_ad")
    vdir = s.index_dir / "vision"

    if args.category:
        categories = [args.category]
    else:
        categories = sorted(p.name for p in vdir.iterdir() if (p / "padim.pkl").exists())

    res = run_validation(raw_root, categories, split_mode=args.split)
    print_summary(res)

    if args.out_file:
        out_file = Path(args.out_file)
    elif args.split == "held_out":
        out_file = Path("runs/held_out_evaluation.json")
    else:
        out_file = Path("runs/validation_results.json")

    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            existing["timestamp"] = res["timestamp"]
            existing["device"] = res["device"]
            existing["evaluation_protocol"] = res["evaluation_protocol"]
            existing["split_mode"] = res["split_mode"]
            existing["calibration_seed"] = res["calibration_seed"]
            existing["is_contaminated"] = res["is_contaminated"]
            existing["contamination_warning"] = res.get("contamination_warning")
            if "categories" not in existing:
                existing["categories"] = {}
            for c, cdata in res["categories"].items():
                existing["categories"][c] = cdata
            res = existing
        except Exception:
            pass

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved validation records to {out_file}")



