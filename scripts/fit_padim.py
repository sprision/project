"""Fit a PaDiM model per category on MVTec AD *defect-free* training images.

Saves weights + a normalisation (min/max of train-good scores) under
data/index/vision/<category>/padim.pkl so the API can load it at runtime.

Usage:
    python -m scripts.fit_padim --category bottle
    python -m scripts.fit_padim --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.config import get_settings          # noqa: E402
from src.app.vision.model import PaDiM            # noqa: E402


def good_images(category_dir: Path) -> list[Path]:
    return sorted((category_dir / "train" / "good").glob("*.png"))


def fit_category(raw_root: Path, category: str, device: str) -> dict:
    cat_dir = raw_root / category
    imgs = good_images(cat_dir)
    if not imgs:
        raise FileNotFoundError(f"no train/good images under {cat_dir} — "
                              "run scripts/download_mvtec.py first")

    s = get_settings()
    model = PaDiM(backbone=s.vision_backbone, n_features=s.padim_n_features,
                  device=device, input_size=s.vision_input_size)
    t0 = time.time()
    stats = model.fit(imgs)
    print(f"[fit] {category}: {stats} in {time.time()-t0:.1f}s")

    # normalisation: min/max of train-good anomaly-map maxima
    scores = []
    for p in imgs[: min(50, len(imgs))]:
        with Image.open(p) as im:
            m = model.anomaly_map(im.convert("RGB"))
        scores.append(float(m.max()))
    lo, hi = float(np.min(scores)), float(np.max(scores))
    model.set_normalisation(lo, hi)

    out = s.index_dir / "vision" / category / "padim.pkl"
    model.save(out)
    print(f"[fit] {category}: normalisation [{lo:.2f},{hi:.2f}] -> {out}")
    return {"category": category, "n_train": stats["n_train"],
            "norm": [lo, hi], "model_path": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--raw-root", default="data/raw/mvtec_ad")
    args = ap.parse_args()

    s = get_settings()
    device = s.resolved_device()
    raw_root = Path(args.raw_root)
    cats = sorted([p.name for p in raw_root.iterdir()
                   if p.is_dir() and (p / "train" / "good").is_dir()]) \
        if args.all else [args.category]
    if not cats or cats == [None]:
        ap.error("provide --category or --all")

    results = [fit_category(raw_root, c, device) for c in cats]
    print(f"[fit] done: {len(results)} categor(ies) on {device}")


if __name__ == "__main__":
    main()
