"""Measure actual RAM and VRAM during sequential model loading and LRU eviction."""
import gc
import json
import torch
from pathlib import Path
from src.app.vision.registry import ModelRegistry, get_process_memory_mb

def main():
    reg = ModelRegistry(cache_capacity=3)
    print("=== BASELINE MEMORY ===")
    base_mem = get_process_memory_mb()
    print(json.dumps(base_mem, indent=2))

    models = ["bottle", "cable", "metal_nut", "screw"]
    measurements = []

    for i, cat in enumerate(models, 1):
        insp = reg.get_inspector(cat)
        mem = get_process_memory_mb()
        cached = reg.cache.keys()
        record = {
            "step": i,
            "loaded_category": cat,
            "cached_count": len(cached),
            "cached_categories": cached,
            "ram_working_set_mb": mem["ram_working_set_mb"],
            "cuda_allocated_mb": mem["cuda_allocated_mb"],
            "cuda_reserved_mb": mem["cuda_reserved_mb"],
            "cache_stats": dict(reg.cache.stats),
        }
        measurements.append(record)
        print(f"\n--- STEP {i}: Loaded '{cat}' ---")
        print(f"  Active cache: {cached}")
        print(f"  RAM Working Set: {mem['ram_working_set_mb']} MB")
        print(f"  CUDA Allocated:  {mem['cuda_allocated_mb']} MB | Reserved: {mem['cuda_reserved_mb']} MB")
        print(f"  Evictions so far: {reg.cache.stats['evictions']}")

    # Assertions
    assert len(reg.cache) == 3, f"Expected cache size 3, got {len(reg.cache)}"
    assert "bottle" not in reg.cache, "Expected 'bottle' to be evicted as LRU"
    assert "screw" in reg.cache, "Expected 'screw' to be present"
    assert reg.cache.stats["evictions"] == 1, "Expected exactly 1 eviction"

    print("\n=== EVICTION VERIFICATION PASSED ===")
    print(f"Successfully evicted oldest model ('bottle') when loading 4th model ('screw') into capacity-3 cache.")

if __name__ == "__main__":
    main()

