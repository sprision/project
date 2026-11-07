"""Unit tests for ModelRegistry and LRUModelCache."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.app.vision.model import AnomalyInspector
from src.app.vision.registry import (
    CategoryMetadata,
    LRUModelCache,
    ModelRegistry,
    get_process_memory_mb,
)


def _fake_inspector(category: str) -> AnomalyInspector:
    mock_model = MagicMock()
    mock_model.backbone_name = "resnet18"
    mock_model.anomaly_map.return_value = None
    return AnomalyInspector(mock_model, category=category, calibrated_threshold=25.0)


class TestLRUModelCache:
    def test_cache_hit_and_miss_accounting(self):
        cache = LRUModelCache(capacity=3)
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

        # Miss
        res = cache.get("bottle")
        assert res is None
        assert cache.stats["misses"] == 1

        # Direct insert for testing
        insp = _fake_inspector("bottle")
        cache._cache["bottle"] = insp

        # Hit
        hit_res = cache.get("bottle")
        assert hit_res is insp
        assert cache.stats["hits"] == 1
        assert "bottle" in cache
        assert len(cache) == 1

    def test_lru_eviction_at_capacity(self, tmp_path):
        cache = LRUModelCache(capacity=3)

        # Pre-populate with 3 items: a, b, c
        cache._cache["a"] = _fake_inspector("a")
        cache._cache["b"] = _fake_inspector("b")
        cache._cache["c"] = _fake_inspector("c")
        assert len(cache) == 3

        # Access "a" so "b" becomes the least recently used
        cache.get("a")

        # Now load a 4th model "d" with mocked PaDiM.load
        mock_p = tmp_path / "d.pkl"
        mock_p.write_bytes(b"dummy")

        with patch("src.app.vision.registry.PaDiM.load", return_value=MagicMock()):
            cache.load_or_get("d", mock_p, calibrated_threshold=20.0)

        # "b" should have been evicted because "a" was accessed, leaving "c" and "a"
        assert len(cache) == 3
        assert "b" not in cache
        assert "a" in cache
        assert "c" in cache
        assert "d" in cache
        assert cache.stats["evictions"] == 1

    def test_cache_clear(self):
        cache = LRUModelCache(capacity=3)
        cache._cache["a"] = _fake_inspector("a")
        cache._cache["b"] = _fake_inspector("b")
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0


class TestModelRegistry:
    def test_dynamic_discovery(self, tmp_path):
        vdir = tmp_path / "vision"
        vdir.mkdir()

        # Category 1: fitted (has padim.pkl)
        cat1 = vdir / "alpha"
        cat1.mkdir()
        (cat1 / "padim.pkl").write_bytes(b"model")
        (cat1 / "threshold.json").write_text(json.dumps({"threshold_raw": 22.5}))

        # Category 2: unfitted (no padim.pkl)
        cat2 = vdir / "beta"
        cat2.mkdir()

        # Category 3: fitted
        cat3 = vdir / "gamma"
        cat3.mkdir()
        (cat3 / "padim.pkl").write_bytes(b"model")

        reg = ModelRegistry(vision_dir=vdir)
        discovered = reg.discover_categories()
        assert discovered == ["alpha", "gamma"]

    def test_category_metadata_loading(self, tmp_path):
        vdir = tmp_path / "vision"
        vdir.mkdir()

        cat = vdir / "screw"
        cat.mkdir()
        (cat / "padim.pkl").write_bytes(b"model")
        (cat / "threshold.json").write_text(json.dumps({"threshold_raw": 19.85}))

        reg = ModelRegistry(vision_dir=vdir)
        meta = reg.get_category_metadata("screw")
        assert meta is not None
        assert meta.name == "screw"
        assert meta.display_name == "Industrial Screws"
        assert meta.domain == "Mechanical & Tooling"
        assert meta.threshold == pytest.approx(19.85)
        assert meta.is_fitted is True
        assert meta.is_cached is False

    def test_memory_stats_reporting(self):
        mem = get_process_memory_mb()
        assert "ram_working_set_mb" in mem
        assert "cuda_allocated_mb" in mem
        assert isinstance(mem["ram_working_set_mb"], (int, float))

    def test_production_categories_discovery(self):
        """Verify that all 4 production categories (including screw) are discovered."""
        reg = ModelRegistry()
        cats = reg.discover_categories()
        for expected in ["bottle", "cable", "metal_nut", "screw"]:
            assert expected in cats, f"Expected {expected} in discovered categories: {cats}"

        # Verify screw metadata
        meta = reg.get_category_metadata("screw")
        assert meta is not None
        assert meta.is_fitted is True
        assert meta.threshold is not None
        assert meta.threshold == pytest.approx(24.1733, abs=0.01)

    def test_lru_cache_eviction_integration(self):
        """Verify LRU cache capacity constraint and eviction order with real models."""
        reg = ModelRegistry(cache_capacity=3)
        assert len(reg.cache) == 0

        # Load bottle, cable, metal_nut
        insp_bottle = reg.get_inspector("bottle")
        insp_cable = reg.get_inspector("cable")
        insp_nut = reg.get_inspector("metal_nut")
        assert len(reg.cache) == 3
        assert reg.cache.keys() == ["bottle", "cable", "metal_nut"]

        # Loading 4th model ('screw') must evict 'bottle'
        insp_screw = reg.get_inspector("screw")
        assert len(reg.cache) == 3
        assert "bottle" not in reg.cache
        assert "screw" in reg.cache
        assert reg.cache.stats["evictions"] == 1


