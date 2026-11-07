"""Empirical investigation into cable's held-out sensitivity limitations.

Analyzes defect mask sizes, patch-level Mahalanobis score distributions,
and ground-truth visual characteristics across all defect types in MVTec AD cable.
"""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.vision.model import PaDiM, load_calibrated_threshold

def main():
    cable_model_path = Path("data/index/vision/cable/padim.pkl")
    model = PaDiM.load(cable_model_path, device="cuda")
    threshold = load_calibrated_threshold(cable_model_path)
    print(f"Cable Calibrated Raw Threshold: {threshold:.4f}")

    raw_root = Path("data/raw/mvtec_ad/cable/test")
    gt_root = Path("data/raw/mvtec_ad/cable/ground_truth")

    print(f"\n{'Defect Type':24s} {'Count':5s} {'Min':6s} {'Mean':6s} {'Max':6s} {'Defect Pixels':14s} {'Missed (< thr)':14s}")
    print("-" * 80)

    for folder in sorted(raw_root.iterdir()):
        if not folder.is_dir(): continue
        scores = []
        gt_pixel_counts = []
        is_good = (folder.name == "good")

        for p in sorted(folder.glob("*.png")):
            with Image.open(p) as im:
                am = model.anomaly_map(im.convert("RGB"))
                s = float(am.max())
                scores.append(s)

            if not is_good:
                gt_p = gt_root / folder.name / (p.stem + "_mask.png")
                if gt_p.exists():
                    gt_arr = np.array(Image.open(gt_p).convert("L"))
                    gt_pixel_counts.append(int(np.sum(gt_arr > 0)))

        scores = np.array(scores)
        n_missed = int((scores < threshold).sum()) if not is_good else int((scores >= threshold).sum())
        gt_px_str = f"{np.mean(gt_pixel_counts):.0f}" if gt_pixel_counts else "N/A"
        print(f"{folder.name:24s} {len(scores):5d} {scores.min():6.2f} {scores.mean():6.2f} {scores.max():6.2f} {gt_px_str:14s} {n_missed:5d} ({n_missed/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    main()
