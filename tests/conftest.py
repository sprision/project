"""Shared fixtures. Sets APP_* env to a temp workspace BEFORE app modules load,
so tests never touch the real data dir and never need a fitted model."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_workspace(tmp_path, monkeypatch):
    """Point data/index/runs at a temp dir for an isolated run."""
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_INDEX_DIR", str(tmp_path / "data" / "index"))
    monkeypatch.setenv("APP_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("APP_LLM_PROVIDER", "none")
    from src.app.config import get_settings
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture()
def sample_image(tmp_path) -> Path:
    """A small valid PNG for upload/validation tests."""
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(p)
    return p


@pytest.fixture()
def textured_images(tmp_path) -> list[Path]:
    """Several normal-looking + one anomalous synthetic images for PaDiM tests."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        arr = rng.integers(100, 140, (64, 64, 3), dtype=np.uint8)
        p = tmp_path / f"good_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    # anomalous: strong localised bright square
    arr = rng.integers(100, 140, (64, 64, 3), dtype=np.uint8)
    arr[20:44, 20:44] = 255
    p = tmp_path / "anom.png"
    Image.fromarray(arr).save(p)
    paths.append(p)
    return paths


@pytest.fixture()
def kb_docs(tmp_path) -> list[Path]:
    """Two small text docs with real, attributable content for retrieval tests."""
    d1 = tmp_path / "doc_a.txt"
    d1.write_text(
        "MVTec AD Dataset\n"
        "1. Overview\n"
        "MVTec AD contains images of industrial objects and textures for "
        "unsupervised anomaly detection. Training uses only defect-free images. "
        "Pixel-precise ground truth masks are provided for defective images. "
        "The task is anomaly localisation, not supervised classification.\n\n"
        "2. Evaluation\n"
        "Reported metrics are image-level and pixel-level AUROC.",
        encoding="utf-8")
    d2 = tmp_path / "doc_b.txt"
    d2.write_text(
        "Anomaly Scoring\n"
        "A high Mahalanobis distance means a patch deviates from learned "
        "normality. It is a hypothesis of a defect, not a confirmed diagnosis, "
        "and the score is not a calibrated probability.",
        encoding="utf-8")
    return [d1, d2]
