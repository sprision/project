"""Evaluate PaDiM on MVTec AD test sets.

Computes image-level AUROC and pixel-level AUROC against ground-truth masks.
Results are written to runs/eval_cv.json — real measured numbers only.

Usage:
    python -m scripts.evaluate_cv --category bottle
    python -m scripts.evaluate_cv --all
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
from src.app.config import get_settings           # noqa: E402
from src.app.vision.model import PaDiM            # noqa: E402


def _load_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as m:
        m = m.convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 0).astype(np.uint8)


def eval_category(raw_root: Path, category: str, device: str,
                  limit: int | None) -> dict:
    s = get_settings()
    model_path = s.index_dir / "vision" / category / "padim.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"no fitted model for {category}; "
                                "run scripts/fit_padim.py first")
    model = PaDiM.load(model_path, device=device)
    size = s.vision_input_size

    test_dir = raw_root / category / "test"
    gt_dir = raw_root / category / "ground_truth"
    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    n_img = 0
    t0 = time.time()
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_anom = defect_dir.name != "good"
        for img_path in sorted(defect_dir.glob("*.png")):
            with Image.open(img_path) as im:
                amap = model.anomaly_map(im.convert("RGB"))
            img_scores.append(float(amap.max()))
            img_labels.append(1 if is_anom else 0)
            if is_anom:
                mask_path = gt_dir / defect_dir.name / \
                    (img_path.stem + "_mask.png")
                if mask_path.exists():
                    mask = _load_mask(mask_path, size)
                    pix_scores.append(amap.ravel())
                    pix_labels.append(mask.ravel())
            n_img += 1
            if limit and n_img >= limit:
                break
        if limit and n_img >= limit:
            break

    img_auroc = roc_auc_score(img_labels, img_scores)
    res = {
        "category": category, "n_images": n_img,
        "n_defective": int(sum(img_labels)),
        "image_auroc": round(float(img_auroc), 4),
        "elapsed_s": round(time.time() - t0, 1),
        "device": device,
    }
    if pix_scores:
        p = np.concatenate(pix_scores); l = np.concatenate(pix_labels)
        res["pixel_auroc"] = round(float(roc_auc_score(l, p)), 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw-root", default="data/raw/mvtec_ad")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per category (quick eval)")
    args = ap.parse_args()

    s = get_settings()
    device = s.resolved_device()
    raw_root = Path(args.raw_root)
    cats = sorted([p.name for p in raw_root.iterdir()
                   if p.is_dir() and (p / "test").is_dir()]) \
        if args.all else [args.category]
    if not cats or cats == [None]:
        ap.error("provide --category or --all")

    new_results = [eval_category(raw_root, c, device, args.limit) for c in cats]
    out = s.runs_dir / "eval_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing evaluation results if single category evaluated
    merged_results = []
    if out.exists() and not args.all:
        try:
            existing_data = json.loads(out.read_text(encoding="utf-8"))
            existing_cats = {r["category"]: r for r in existing_data.get("results", [])}
            for nr in new_results:
                existing_cats[nr["category"]] = nr
            merged_results = list(existing_cats.values())
        except Exception:
            merged_results = new_results
    else:
        merged_results = new_results

    out.write_text(json.dumps({"results": merged_results}, indent=2) + "\n")
    for r in new_results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
