"""Unit tests for the experimental PatchCore anomaly detection model."""
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from src.app.vision.patchcore import PatchCore


def _create_dummy_image(path: Path, color=(128, 128, 128)) -> Path:
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)
    return path


def test_patchcore_initialization():
    pc = PatchCore(device="cpu", coreset_size=500, input_size=224)
    assert pc.backbone_name == "resnet18"
    assert pc.coreset_size == 500
    assert pc.input_size == 224
    assert pc.memory_bank_ is None


def test_patchcore_feature_extraction():
    pc = PatchCore(device="cpu")
    dummy = torch.randn(2, 3, 224, 224)
    feats = pc._extract_patches(dummy)
    assert feats.shape == (2, 384, 28, 28)


def test_patchcore_fit_and_anomaly_map(tmp_path):
    pc = PatchCore(device="cpu", coreset_size=200)
    train_paths = [_create_dummy_image(tmp_path / f"train_{i}.png", (i * 20, 100, 150)) for i in range(4)]

    res = pc.fit(train_paths, batch_size=2)
    assert res["n_train_images"] == 4
    assert res["memory_bank_size"] == min(200, 4 * 28 * 28)
    assert pc.memory_bank_ is not None

    test_img = Image.new("RGB", (224, 224), (200, 20, 20))
    am = pc.anomaly_map(test_img)
    assert isinstance(am, np.ndarray)
    assert am.shape == (224, 224)
    assert am.max() >= 0.0


def test_patchcore_save_and_load(tmp_path):
    pc = PatchCore(device="cpu", coreset_size=100)
    train_paths = [_create_dummy_image(tmp_path / "img1.png"), _create_dummy_image(tmp_path / "img2.png")]
    pc.fit(train_paths)
    pc.set_normalisation(5.0, 25.0)

    save_path = tmp_path / "patchcore.pt"
    pc.save(save_path)
    assert save_path.exists()

    loaded = PatchCore.load(save_path, device="cpu")
    assert loaded.memory_bank_ is not None
    assert loaded.memory_bank_.shape == pc.memory_bank_.shape
    assert loaded.normalisation_["min"] == pytest.approx(5.0)
    assert loaded.normalisation_["max"] == pytest.approx(25.0)
