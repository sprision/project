"""Prototype and calibrate Conductor Continuity Detector for missing_wire."""
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_anomaly_detection import collect_image_paths, split_stratified_paths

raw_root = Path("data/raw/mvtec_ad/cable")
paths = collect_image_paths(raw_root / "test")
cal_paths, ev_paths = split_stratified_paths(paths, seed=42, frac=0.5)

def analyze_conductor_continuity(img_path: Path):
    """Analyze conductor continuity and wire completeness across the harness channels."""
    arr = np.array(Image.open(img_path))
    # Cable background is dark cyan/teal ~[53, 71, 87]
    bg = np.array([53, 71, 87])
    diff_bg = np.linalg.norm(arr.astype(float) - bg, axis=2)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Wire foreground mask: pixels differing from background and above noise
    fg_mask = (diff_bg > 35) & (gray > 30)
    
    # The 3 harness wire channels:
    # Top channel: y in [240, 420], x in [410, 600]
    # Left channel: y in [490, 690], x in [240, 450]
    # Right channel: y in [490, 710], x in [580, 770]
    channels = {
        "top": fg_mask[240:420, 410:600],
        "left": fg_mask[490:690, 240:450],
        "right": fg_mask[490:710, 580:770],
    }
    
    densities = {k: float(np.mean(m)) for k, m in channels.items()}
    # Min density across the 3 critical channels
    min_ch = min(densities, key=densities.get)
    min_density = densities[min_ch]
    
    # Also compute edge density in the minimum channel to verify absence of wire edges
    gray_ch = {
        "top": gray[240:420, 410:600],
        "left": gray[490:690, 240:450],
        "right": gray[490:710, 580:770],
    }[min_ch]
    edges = cv2.Canny(gray_ch, 40, 120)
    edge_density = float(np.mean(edges > 0))
    
    # Continuity score: low density indicates missing conductor channel
    # We invert so higher score = higher probability of missing conductor
    defect_score = float(1.0 - min_density)
    return {
        "min_channel": min_ch,
        "min_density": min_density,
        "edge_density": edge_density,
        "continuity_defect_score": defect_score,
        "densities": densities
    }

print("=== CALIBRATION SPLIT EVALUATION (Conductor Continuity) ===")
cal_good_scores = []
cal_mw_scores = []

for is_good, p in cal_paths:
    dname = p.parent.name
    res = analyze_conductor_continuity(p)
    if is_good:
        cal_good_scores.append(res["continuity_defect_score"])
    elif dname == "missing_wire":
        cal_mw_scores.append(res["continuity_defect_score"])
        print(f"  Cal missing_wire {p.name}: score={res['continuity_defect_score']:.3f}, min_ch={res['min_channel']} (dens={res['min_density']:.3f}, edge={res['edge_density']:.3f})")

print(f"\nCal Good cables (29 images): min={min(cal_good_scores):.3f}, max={max(cal_good_scores):.3f}, mean={np.mean(cal_good_scores):.3f}")
print(f"Cal Missing Wire (2 images): min={min(cal_mw_scores):.3f}, max={max(cal_mw_scores):.3f}, mean={np.mean(cal_mw_scores):.3f}")

