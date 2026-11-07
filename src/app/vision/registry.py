"""Scalable Model Registry and In-Memory LRU Cache for PaDiM Anomaly Detectors.

Provides:
  - Dynamic discovery of fitted models from data/index/vision/
  - Lazy, on-demand loading of PaDiM models into AnomalyInspectors
  - Thread-safe LRU model cache bounded by configurable capacity (default=3)
  - Safe GPU/CPU memory eviction with gc and CUDA cache clearing
  - RAM & VRAM tracking
  - Category metadata and threshold validation
"""
from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from ..config import get_settings
from .model import AnomalyInspector, PaDiM, load_calibrated_threshold

log = logging.getLogger(__name__)


@dataclass
class CategoryMetadata:
    name: str
    display_name: str
    domain: str
    threshold: float | None
    reliability: str
    model_path: str
    threshold_path: str
    is_fitted: bool
    is_cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_process_memory_mb() -> dict[str, float]:
    """Measure actual process Working Set RAM (Windows/POSIX) and CUDA VRAM in MB."""
    ram_mb = 0.0
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            h = kernel32.GetCurrentProcess()
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                ram_mb = round(pmc.WorkingSetSize / (1024 * 1024), 2)
        except Exception as e:
            log.debug(f"Failed to query Windows RAM working set: {e}")
    else:
        try:
            import resource
            ram_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        except Exception:
            pass

    cuda_alloc_mb = 0.0
    cuda_res_mb = 0.0
    if torch.cuda.is_available():
        cuda_alloc_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        cuda_res_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)

    return {
        "ram_working_set_mb": ram_mb,
        "cuda_allocated_mb": cuda_alloc_mb,
        "cuda_reserved_mb": cuda_res_mb,
    }


class LRUModelCache:
    """Thread-safe bounded in-memory LRU cache for AnomalyInspectors."""

    def __init__(self, capacity: int = 3, device: str = "cpu",
                 overlay_dir: Path | None = None):
        self.capacity = max(1, capacity)
        self.device = device
        self.overlay_dir = overlay_dir
        self._cache: OrderedDict[str, AnomalyInspector] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "loads": 0,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, category: str) -> bool:
        with self._lock:
            return category in self._cache

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> list[AnomalyInspector]:
        with self._lock:
            return list(self._cache.values())

    def items(self) -> list[tuple[str, AnomalyInspector]]:
        with self._lock:
            return list(self._cache.items())

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._cache.keys()))

    def get(self, category: str, default: Any = None) -> AnomalyInspector | None:
        """Fetch inspector from cache or return default without loading."""
        with self._lock:
            if category in self._cache:
                self._cache.move_to_end(category)
                self.stats["hits"] += 1
                return self._cache[category]
            self.stats["misses"] += 1
            return default

    def load_or_get(self, category: str, model_path: Path,
                    calibrated_threshold: float | None = None) -> AnomalyInspector:
        """Fetch inspector from cache; if missing, lazily load and evict LRU if full."""
        with self._lock:
            if category in self._cache:
                self._cache.move_to_end(category)
                self.stats["hits"] += 1
                return self._cache[category]

            self.stats["misses"] += 1
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Evict least recently used if at capacity
            if len(self._cache) >= self.capacity:
                self._evict_lru()

            # Load model
            log.info(f"[LRU Cache] Loading model for '{category}' on {self.device}...")
            t_model = PaDiM.load(model_path, device=self.device)
            inspector = AnomalyInspector(
                t_model,
                category=category,
                calibrated_threshold=calibrated_threshold,
                overlay_dir=self.overlay_dir,
            )
            self._cache[category] = inspector
            self.stats["loads"] += 1
            return inspector

    def _evict_lru(self) -> str | None:
        """Evict the oldest accessed inspector and clean memory."""
        with self._lock:
            if not self._cache:
                return None
            evicted_cat, inspector = self._cache.popitem(last=False)
            self.stats["evictions"] += 1
            log.info(f"[LRU Cache] Evicted '{evicted_cat}' (capacity={self.capacity})")

            # Clean up PyTorch references
            try:
                if hasattr(inspector, "model") and inspector.model:
                    inspector.model.extractor = None
                    inspector.model.means_ = None
                    inspector.model.inv_cov_ = None
                    inspector.model._inv_t = None
                    inspector.model = None
            except Exception as e:
                log.debug(f"Error releasing model references during eviction: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return evicted_cat

    def clear(self) -> None:
        """Evict all cached models and free memory."""
        with self._lock:
            cats = list(self._cache.keys())
            for c in cats:
                inspector = self._cache.pop(c)
                try:
                    if hasattr(inspector, "model") and inspector.model:
                        inspector.model = None
                except Exception:
                    pass
            self._cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ModelRegistry:
    """Discovers available categories, manages thresholds, and coordinates LRU caching."""

    KNOWN_DOMAINS = {
        "bottle": ("Glass & Containers", "Packaging", "high"),
        "cable": ("Multi-wire Cables", "Electronics & Wiring", "moderate"),
        "metal_nut": ("Fastener Hardware", "Mechanical & Tooling", "high"),
        "capsule": ("Gelatin Capsules", "Pharmaceuticals", "high"),
        "screw": ("Industrial Screws", "Mechanical & Tooling", "high"),
        "pill": ("Compressed Tablets", "Pharmaceuticals", "high"),
        "transistor": ("Semiconductor Components", "Electronics & PCB", "high"),
        "tile": ("Ceramic Surface", "Surface & Structural", "high"),
        "grid": ("Wire Sieve & Mesh", "Filtration", "high"),
        "zipper": ("Apparel & Closures", "Textiles & Closures", "high"),
    }

    def __init__(self, vision_dir: Path | None = None,
                 cache_capacity: int = 3,
                 device: str | None = None,
                 overlay_dir: Path | None = None):
        s = get_settings()
        self.vision_dir = vision_dir or (s.index_dir / "vision")
        self.vision_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or s.resolved_device()
        self.overlay_dir = overlay_dir or (s.data_dir / "heatmaps")
        self.cache = LRUModelCache(capacity=cache_capacity, device=self.device,
                                   overlay_dir=self.overlay_dir)

    def discover_categories(self) -> list[str]:
        """Dynamically list all categories that have a fitted padim.pkl."""
        if not self.vision_dir.exists():
            return []
        cats = [
            p.name for p in self.vision_dir.iterdir()
            if p.is_dir() and (p / "padim.pkl").exists()
        ]
        return sorted(cats)

    def get_category_metadata(self, category: str) -> CategoryMetadata | None:
        """Fetch metadata, threshold, and status for a single category."""
        cat_dir = self.vision_dir / category
        model_path = cat_dir / "padim.pkl"
        threshold_path = cat_dir / "threshold.json"
        is_fitted = model_path.exists()

        if not is_fitted and not cat_dir.exists():
            return None

        # Determine display name and domain
        if category in self.KNOWN_DOMAINS:
            disp, dom, rel = self.KNOWN_DOMAINS[category]
        else:
            disp = category.replace("_", " ").title()
            dom = "General Industrial"
            rel = "standard"

        thresh = load_calibrated_threshold(model_path) if is_fitted else None

        return CategoryMetadata(
            name=category,
            display_name=disp,
            domain=dom,
            threshold=thresh,
            reliability=rel,
            model_path=str(model_path),
            threshold_path=str(threshold_path),
            is_fitted=is_fitted,
            is_cached=(category in self.cache),
        )

    def list_all_metadata(self) -> list[dict[str, Any]]:
        """List metadata for all dynamically discovered categories."""
        res = []
        for cat in self.discover_categories():
            meta = self.get_category_metadata(cat)
            if meta:
                res.append(meta.to_dict())
        return res

    def get_inspector(self, category: str | None) -> AnomalyInspector | None:
        """Obtain an inspector for category, lazily loading it through LRU cache."""
        available = self.discover_categories()
        if not available:
            return None

        cat = category or available[0]
        if cat not in available:
            return None

        model_path = self.vision_dir / cat / "padim.pkl"
        threshold = load_calibrated_threshold(model_path)

        return self.cache.load_or_get(
            category=cat,
            model_path=model_path,
            calibrated_threshold=threshold,
        )

