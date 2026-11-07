"""Download a subset of MVTec AD from the HuggingFace mirror `foersben/mvtec-ad`.

The official mydrive download links are dead; this mirror contains the full
dataset (train/good, test/<defect>, ground_truth/<defect> masks) under
CC-BY-NC-SA-4.0. We download selectively per category to keep the footprint small.

Usage:
    python -m scripts.download_mvtec --categories bottle cable metal_nut
    python -m scripts.download_mvtec --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "foersben/mvtec-ad"
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]


def download(categories: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cat in categories:
        print(f"[download] {cat} ...")
        local = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{cat}/**"],
            local_dir=str(out_dir),
        )
        p = Path(local) / cat
        n = sum(1 for _ in p.rglob("*.png"))
        print(f"[download]   -> {p}  ({n} png files)")
        written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "cable", "metal_nut"],
                    choices=ALL_CATEGORIES)
    ap.add_argument("--all", action="store_true", help="download all 15 categories")
    ap.add_argument("--out", default="data/raw/mvtec_ad")
    args = ap.parse_args()

    cats = ALL_CATEGORIES if args.all else args.categories
    download(cats, Path(args.out))
    print("[download] done.")


if __name__ == "__main__":
    main()
