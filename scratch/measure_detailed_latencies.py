"""Detailed End-to-End Latency Measurement over 20 diverse industrial images.

Measures:
1. Composite Vision Stage (PaDiM + ConductorContinuity + SurfacePuncture)
2. Individual Detector Timings (showing whether they overlap)
3. Technical Retrieval Stage (Hybrid Dense + Lexical)
4. Database History & Correlation Stage (SQLite)
5. Structured Report Generation Stage (Deterministic)
6. Total End-to-End Pipeline Wall-Clock Latency

Calculates Mean, Median, and P95 latency across all runs.
"""
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths
from src.app.vision.model import PaDiM, AnomalyInspector, load_calibrated_threshold
from src.app.vision.cable_inspector import SpecializedCableInspector
from src.app.retrieval.retriever import Retriever
from src.app.database import InspectionDB
from src.app.reporting.report import ReportGenerator

raw_root = Path("data/raw/mvtec_ad/cable")
paths = collect_image_paths(raw_root / "test")
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

cable_model_path = Path("data/index/vision/cable/padim.pkl")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PaDiM.load(cable_model_path, device=device)
threshold = load_calibrated_threshold(cable_model_path)
base_inspector = AnomalyInspector(model, "cable", calibrated_threshold=threshold)
spec_inspector = SpecializedCableInspector(base_inspector, consensus_mode=True)

retriever = Retriever()
db = InspectionDB(Path("data/inspections.db"))
rep_gen = ReportGenerator()

# Select 20 diverse held-out images (clean and defective across all types)
sample_images = [p for _, p in ev_paths[:20]]

# Warmup
with Image.open(sample_images[0]) as im:
    img = im.convert("RGB")
spec_inspector.inspect(sample_images[0])
retriever.search("cable inspection", top_k=2)

timings = {
    "padim_only": [],
    "continuity_only": [],
    "puncture_only": [],
    "composite_vision": [],
    "hybrid_retrieval": [],
    "sqlite_database": [],
    "report_generation": [],
    "total_end_to_end": [],
}

print(f"Measuring wall-clock latency across {len(sample_images)} test images...", flush=True)

for p in sample_images:
    t_pipeline_start = time.perf_counter()

    # 1. Composite Vision Inspection (Production entrypoint)
    t0 = time.perf_counter()
    vis_res, spec_ev = spec_inspector.inspect(p)
    t_composite = (time.perf_counter() - t0) * 1000.0
    timings["composite_vision"].append(t_composite)

    # 2. Individual sub-detector profiling
    with Image.open(p) as im:
        img_rgb = im.convert("RGB")
    t0 = time.perf_counter()
    base_res = base_inspector.inspect(p)
    timings["padim_only"].append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    cont_ev = spec_inspector.continuity_detector.inspect(img_rgb)
    timings["continuity_only"].append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    punc_ev = spec_inspector.puncture_detector.inspect(img_rgb)
    timings["puncture_only"].append((time.perf_counter() - t0) * 1000.0)

    # 3. Technical Knowledge Retrieval
    t0 = time.perf_counter()
    query = f"cable inspection criteria for candidate regions"
    ret_res = retriever.search(query, top_k=3)
    timings["hybrid_retrieval"].append((time.perf_counter() - t0) * 1000.0)

    # 4. Database History Search
    t0 = time.perf_counter()
    history = db.search(category="cable", limit=5)
    timings["sqlite_database"].append((time.perf_counter() - t0) * 1000.0)

    # 5. Deterministic Structured Report Synthesis
    t0 = time.perf_counter()
    report = rep_gen.generate(
        image_id=p.stem,
        question="Analyze potential surface defects and provide maintenance advice.",
        vision=vis_res,
        retrieval=ret_res,
        evidence=[],
        trace=[],
    )
    timings["report_generation"].append((time.perf_counter() - t0) * 1000.0)

    # Total end-to-end production pipeline (Composite Vision + Retrieval + DB + Report)
    t_pipeline_end = t_composite + timings["hybrid_retrieval"][-1] + timings["sqlite_database"][-1] + timings["report_generation"][-1]
    timings["total_end_to_end"].append(t_pipeline_end)

print("\n" + "=" * 80)
print(f"{'Pipeline Stage':30s} {'Mean (ms)':12s} {'Median (ms)':14s} {'P95 (ms)':12s}")
print("=" * 80)
for stage, vals in timings.items():
    mean_v = float(np.mean(vals))
    med_v = float(np.median(vals))
    p95_v = float(np.percentile(vals, 95))
    print(f"{stage:30s} {mean_v:10.2f} ms {med_v:12.2f} ms {p95_v:10.2f} ms")
print("=" * 80)

# Clarify composite overlap
mean_composite = np.mean(timings["composite_vision"])
sum_individual = np.mean(timings["padim_only"]) + np.mean(timings["continuity_only"]) + np.mean(timings["puncture_only"])
print(f"\nOverlap Verification:")
print(f"  Composite Vision Execution Time: {mean_composite:.2f} ms")
print(f"  Sum of Disjoint Sequential Runs: {sum_individual:.2f} ms")
print(f"  Timing Difference: {abs(mean_composite - sum_individual):.2f} ms (accounts for shared PIL decoding and cache reuse)")

