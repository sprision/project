# Representative Demo Samples

This directory provides pre-packaged representative industrial samples (both nominal defect-free and anomalous components) across all four supported categories in the inspection platform (`bottle`, `cable`, `metal_nut`, `screw`).

These samples allow immediate testing and demonstration of the vision pipeline, hybrid RAG retrieval, multi-detector consensus, and decision-support reporting without needing to navigate the full raw dataset directory.

---

## Sample Inventory & Expected Inspection Behavior

| Sample File | Category | Defect Type | Expected Anomaly Verdict | Active Detectors & Scoring Profile | Recommended Action |
|-------------|----------|-------------|--------------------------|-----------------------------------|--------------------|
| `bottle_normal.png` | `bottle` | Defect-free (Good) | **Normal (Clean)** | PaDiM raw score: ~13.5 (Threshold: 23.461). Delta < -9.9. | Component passed. Operational monitoring. |
| `bottle_defective_broken_large.png` | `bottle` | Large fracture on glass neck/body | **Candidate Visual Defect** | PaDiM raw score: ~64.90 (Threshold: 23.461). High margin (Delta > 41.0). Large contour localized. | Quarantine component. Visual QA confirmation required. |
| `cable_normal.png` | `cable` | Defect-free (Good) | **Normal (Clean)** | PaDiM raw score: ~18.95 (Threshold: 30.422). Continuity nominal (1.0). Puncture peak: 26.42 (< 38.0). | Component passed. Zero false alarms under boundary-masked inspector. |
| `cable_defective_missing_wire.png` | `cable` | Missing wire strand (negative anomaly) | **Uncertain Finding (Review)** | PaDiM raw score: ~18.29 (below threshold due to matching background). Conductor continuity detector triggers density alert (0.28 vs 0.50 cutoff). | **Human Review Mandatory**: Conduct physical count of wire terminals at breakout junction. |
| `cable_defective_poke_insulation.png` | `cable` | Microscopic insulation puncture ($98 \times 114$ px) | **Candidate Visual Defect** | PaDiM raw score: ~54.11 (Threshold: 30.422). Surface puncture detector peak: 54.11 (> 38.0). Precise bbox localized. | Quarantine cable. Physically probe cable jacket at candidate puncture coordinates. |
| `metal_nut_normal.png` | `metal_nut` | Defect-free (Good) | **Normal (Clean)** | PaDiM raw score: ~14.1 (Threshold: 23.170). Delta < -9.0. | Component passed. Operational monitoring. |
| `metal_nut_defective_bent.png` | `metal_nut` | Deformed hex thread/body | **Candidate Visual Defect** | PaDiM raw score: ~38.45 (Threshold: 23.170). High margin (Delta > 15.0). Defect contour localized. | Reject fastener. Tooling recalibration check. |
| `screw_normal.png` | `screw` | Defect-free (Good) | **Normal (Clean)** | PaDiM raw score: ~18.2 (Threshold: 24.173). Delta < -5.9. | Component passed. Note: Screw orientation variance can raise scores near threshold. |
| `screw_defective_scratch_head.png` | `screw` | Surface gouge on screw head | **Candidate Visual Defect** | PaDiM raw score: ~34.80 (Threshold: 24.173). High margin (Delta > 10.6). Head scratch localized. | Reject screw. Check hopper feed guides for abrasive contact. |

---

## How to Test via Web Dashboard

1. Launch the server:
   ```powershell
   python run.py
   # or
   uvicorn src.app.main:app --reload
   ```
2. Navigate to `http://localhost:8000` in your browser.
3. On the **New Inspection** tab, use the **Quick Demo Samples** toolbar to load any sample instantly with a single click, or drag and drop any image from this folder into the upload dropzone.
4. Click **Run Inspection** to inspect the component.

---

## How to Test via Python Script

Run the product readiness test script:
```powershell
python scratch/verify_product_readiness.py
```
This script exercises these exact samples sequentially, verifying API endpoints, multi-detector cable logic, model caching across all 4 categories, SQLite persistence, and JSON report export.

