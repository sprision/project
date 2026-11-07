"""Calibrate per-category anomaly thresholds via a seeded calibration split.

Implements the approved methodology:
  1. Per category, split the labelled MVTec test set into disjoint
     calibration / evaluation halves via a fixed seeded stratified split.
  2. Pick the raw-score decision threshold maximising balanced accuracy on the
     calibration half only.
  3. Report held-out evaluation metrics on the untouched evaluation half:
     confusion matrix, sensitivity, specificity, balanced accuracy, FPR.
  4. Repeat over multiple seeds and report threshold / metric variability.

Writes data/index/vision/<cat>/threshold.json and runs/calibration.json.
Does not touch the PaDiM pickles or the full-test AUROC results in runs/eval_cv.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.config import get_settings                 # noqa: E402
from src.app.vision.model import PaDiM                # noqa: E402

CALIBRATION_SEED = 42
N_SEEDS = 50
SPLIT_FRACTION = 0.5


def raw_max_of(model, p: Path) -> float:
    with Image.open(p) as im:
        return float(model.anomaly_map(im.convert("RGB")).max())


def collect_rows(model, cat_dir: Path) -> list[tuple[bool, float]]:
    rows = []
    for dd in sorted(cat_dir.iterdir()):
        if not dd.is_dir():
            continue
        is_good = dd.name == "good"
        for p in sorted(dd.glob("*.png")):
            rows.append((is_good, raw_max_of(model, p)))
    return rows


def split_stratified(rows, seed: int, frac: float):
    rng = np.random.default_rng(seed)
    cal, ev = [], []
    for is_good in (True, False):
        idx = [i for i, (g, _) in enumerate(rows) if g == is_good]
        rng.shuffle(idx)
        k = int(round(len(idx) * (1.0 - frac)))
        cal += [rows[i] for i in idx[:k]]
        ev += [rows[i] for i in idx[k:]]
    return cal, ev


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens,
            "specificity": spec, "balanced_accuracy": 0.5 * (sens + spec),
            "false_positive_rate": fp / max(1, fp + tn)}


def pick_threshold(s: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(s)
    sc, yc = s[order], y[order]
    mids = (sc[:-1] + sc[1:]) / 2.0
    candidates = np.concatenate([[float(sc[0]) - 1.0], mids, [float(sc[-1]) + 1.0]])
    best_t, best_bal = None, -1.0
    for t in candidates:
        m = metrics(yc, (sc >= t).astype(int))
        if m["balanced_accuracy"] > best_bal:
            best_bal, best_t = m["balanced_accuracy"], float(t)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw/mvtec_ad")
    ap.add_argument("--index-dir", default=None)
    args = ap.parse_args()

    s = get_settings()
    raw_root = Path(args.raw_root)
    index_dir = Path(args.index_dir) if args.index_dir else s.index_dir
    vdir = index_dir / "vision"

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "seeded_calibration_split_then_held_out_evaluation",
        "calibration_seed": CALIBRATION_SEED,
        "n_seeds": N_SEEDS,
        "limitations": [
            "Evaluation subsets are small (bottle: 10 good / 32 defect held-out; "
            "metal_nut: 11 good / 47 defect; cable: 29 good / 46 defect), so per-split "
            "metric variances are high; the 50-seed sweep quantifies this rather than "
            "reporting single-split numbers as exact.",
            "Cable retains substantial good/defect score overlap (image AUROC ~0.84), so "
            "no threshold can fully resolve it; calibration reduces false-positive "
            "flooding but does not solve that overlap.",
        ],
        "categories": {},
    }

    for cat in sorted(p.name for p in vdir.iterdir() if (p / "padim.pkl").exists()):
        t0 = time.time()
        model = PaDiM.load(vdir / cat / "padim.pkl", device=s.resolved_device())
        rows = collect_rows(model, raw_root / cat / "test")
        cal, ev = split_stratified(rows, CALIBRATION_SEED, SPLIT_FRACTION)

        cal_y = np.array([0 if g else 1 for g, _ in cal])
        cal_s = np.array([r for _, r in cal])
        ev_y = np.array([0 if g else 1 for g, _ in ev])
        ev_s = np.array([r for _, r in ev])

        t_cal = pick_threshold(cal_s, cal_y)
        cal_m = metrics(cal_y, (cal_s >= t_cal).astype(int))
        ev_m = metrics(ev_y, (ev_s >= t_cal).astype(int))

        bal_accs, fprs, thrs = [], [], []
        for seed in range(N_SEEDS):
            c, e = split_stratified(rows, seed, SPLIT_FRACTION)
            cy = np.array([0 if g else 1 for g, _ in c])
            cs = np.array([r for _, r in c])
            t_c = pick_threshold(cs, cy)
            ye = np.array([0 if g else 1 for g, _ in e])
            se = np.array([r for _, r in e])
            m_e = metrics(ye, (se >= t_c).astype(int))
            bal_accs.append(m_e["balanced_accuracy"])
            fprs.append(m_e["false_positive_rate"])
            thrs.append(t_c)

        bal_arr = np.array(bal_accs); fpr_arr = np.array(fprs); thr_arr = np.array(thrs)
        sel = {
            "category": cat,
            "threshold_raw": float(t_cal),
            "decision_rule": "raw_anomaly_score >= threshold_raw",
            "method": "maximising_balanced_accuracy_on_seeded_calibration_subset",
            "seed": CALIBRATION_SEED,
            "split": {
                "calibration": {"good": int((cal_y == 0).sum()),
                                "defect": int((cal_y == 1).sum())},
                "evaluation": {"good": int((ev_y == 0).sum()),
                               "defect": int((ev_y == 1).sum())},
            },
            "calibration_metrics": cal_m,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        threshold_path = vdir / cat / "threshold.json"
        threshold_path.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")

        report["categories"][cat] = {
            **sel,
            "evaluation_metrics": ev_m,
            "stability_50seeds": {
                "n_splits": N_SEEDS,
                "eval_balanced_acc_mean": round(float(bal_arr.mean()), 4),
                "eval_balanced_acc_std": round(float(bal_arr.std()), 4),
                "eval_balanced_acc_min": round(float(bal_arr.min()), 4),
                "eval_balanced_acc_max": round(float(bal_arr.max()), 4),
                "eval_fpr_mean": round(float(fpr_arr.mean()), 4),
                "eval_fpr_max": round(float(fpr_arr.max()), 4),
                "selected_threshold_median": round(float(np.median(thr_arr)), 4),
                "selected_threshold_q25": round(float(np.percentile(thr_arr, 25)), 4),
                "selected_threshold_q75": round(float(np.percentile(thr_arr, 75)), 4),
            },
        }
        print(f"\n== {cat} == took {time.time() - t0:.1f}s")
        print(f"  split: cal {int((cal_y == 0).sum())} good / {int((cal_y == 1).sum())} defect | "
              f"eval {int((ev_y == 0).sum())} good / {int((ev_y == 1).sum())} defect")
        print(f"  selected threshold_raw (from cal): {t_cal:.4f} | cal bal-acc: {cal_m['balanced_accuracy']:.3f}")
        print(f"  held-out eval (at selected threshold): bal-acc {ev_m['balanced_accuracy']:.3f} | "
              f"sens {ev_m['sensitivity']:.3f} spec {ev_m['specificity']:.3f} | "
              f"FPR {ev_m['false_positive_rate']:.3f}")

    out = s.runs_dir / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[wrote] {threshold_path}")
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
