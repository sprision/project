# TASK_TRACKER.md

## Completed Tasks

- [x] **Project Discovery & Memory Establishment**
  - Read `AGENTS.md`, `MASTER_CONTEXT.md`, `PROJECT_STATUS.md`, `TASK_TRACKER.md`, `BUGS_AND_ISSUES.md`.
  - Inspected repository git commit history (commit `3c5c85a` confirmed).
  - Confirmed Python 3.12 virtual environment, PyTorch with CUDA acceleration, torchvision, timm, and sentence-transformers.

- [x] **PaDiM Model Implementation & Optimization**
  - Verified 3 pre-trained PaDiM models (`bottle`, `cable`, `metal_nut`) in `data/index/vision/`.
  - Verified batched matrix multiplication (`torch.bmm`) acceleration for Mahalanobis distances.
  - Verified calibrated raw threshold loading and anomaly verdict logic.
  - Verified JET heatmap overlay generation.

- [x] **Hybrid RAG Pipeline Optimization**
  - Resolved Hugging Face Hub offline retry latency in `src/app/retrieval/retriever.py` by configuring `local_files_only=True` with graceful online fallback.
  - Verified dense embeddings (MiniLM-L6-v2) + lexical BM25Plus retrieval in 17.7s cold / <1s warm.

- [x] **Final Deployment, Demonstration & Delivery Milestone**
  - **Reproducible Windows Setup & Launcher:**
    - Authored single-command cross-platform entrypoint `run.py` (`python run.py --host 127.0.0.1 --port 8000`).
    - Added one-click Windows launchers `launch.bat` and `launch.ps1`.
    - Documented clean environment installation (Python 3.12, PyTorch CUDA 12.6 or CPU wheel) and safe environment configuration (`.env.example` with zero mandatory API keys).
    - Hardened Docker support (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) without disrupting Windows workflows.
  - **Guided Demo Experience & Representative Samples:**
    - Created `demo_samples/` and `frontend/demo_samples/` with 9 pre-packaged industrial test images across all four categories:
      - Bottle: `bottle_normal.png` (Clean, score ~13.5) & `bottle_defective_broken_large.png` (Fracture, score ~64.90).
      - Cable: `cable_normal.png` (Clean, score ~18.95), `cable_defective_missing_wire.png` (Strand break, continuity alert), & `cable_defective_poke_insulation.png` (Puncture, score ~54.11).
      - Metal Nut: `metal_nut_normal.png` (Clean, score ~14.1) & `metal_nut_defective_bent.png` (Deformed hex, score ~38.45).
      - Screw: `screw_normal.png` (Clean, score ~18.2) & `screw_defective_scratch_head.png` (Head scratch, score ~34.80).
    - Authored `demo_samples/README.md` documenting expected scores, thresholds, and inspection actions.
    - Integrated **⚡ One-Click Demo Samples** toolbar directly into the Image Upload card on the New Inspection page.
    - Added **🚀 Guided Demo Tour** modal (`demo-tour-modal`) in `frontend/index.html` with interactive category launchers, 3-level precedence explanations, and authority boundary reminders.
  - **Comprehensive Production Documentation:**
    - Completely overhauled root `README.md` into an exhaustive, publication-grade manual covering architecture, features, 4-category inventory, empirical benchmarks (held-out vs calibration), detailed latency percentiles, real-world factory vs benchmark gap analysis, setup, launcher scripts, REST API reference, and verification.
  - **Final Regression & Verification:**
    - Verified full unit test suite: **74 / 74 passed in 40.02s**.
    - Verified product readiness verification script (`scratch/verify_product_readiness.py`): all 5 steps passed.
    - Verified application startup path and dynamic model registry initialization across all 4 categories (`bottle`, `cable`, `metal_nut`, `screw`) on `cuda`.
    - Preserved all historical benchmark records and model artifacts strictly intact.

- [x] **Final Decision Logic & Product Readiness Pass**
  - **Explicit 3-Level Precedence Enforcement in `src/app/reporting/report.py`:**
    - Level 1 (`uncertain_human_review`) evaluated first: triggers on borderline proximity ($|\Delta| < 3.0$) OR detector disagreement (e.g. specialized detector trigger while PaDiM is nominal).
    - Level 2 (`candidate_defects`) evaluated second: triggers on high margin ($\Delta \ge 5.0$) OR multi-detector corroborated consensus ($\Delta \ge 3.0$), provided Level 1 did not trigger.
    - Level 3 (`potential_anomalies`) evaluated third: captures remaining single-detector moderate activations ($3.0 \le \Delta < 5.0$).
    - Pairwise disjoint sets: strictly enforced so no finding can belong to multiple categories.
    - Hardcoded `is_human_confirmed=False` across all algorithmic outputs; attached specific action instructions (`🔍 Action Required: ...` or `⚡ Action Required: ...`).
  - **Comprehensive Boundary Condition Unit Testing (`tests/unit/test_report.py`):**
    - Added `test_finding_categorization_boundary_conditions` testing score margins $-3.0$, $2.99$ (just below 3.0), $3.0$, $4.99$ (just below 5.0), and $5.0$.
    - Tested detector disagreement cases and corroborated findings near uncertainty boundaries.
    - Verified strict mutual exclusivity ($\text{Uncertain} \cap \text{Candidate} = \emptyset$, $\text{Uncertain} \cap \text{Potential} = \emptyset$, $\text{Candidate} \cap \text{Potential} = \emptyset$).
  - **Evaluation Integrity & Dashboard Reporting (`frontend/index.html`):**
    - Preserved all historical benchmark artifacts and model files untouched.
    - Added dedicated Cable Benchmark Comparison Table in UI contrasting:
      - Baseline PaDiM: 60.87% Sens / 82.76% Spec / 71.81% BalAcc
      - ⭐ Naive OR: **89.13% Sens / 68.97% Spec / 79.05% BalAcc** (highlighted as best binary balanced accuracy on this split)
      - Consensus & Masked: **80.43% Sens / 72.41% Spec / 76.42% BalAcc** (explained operational trade-off: suppresses edge glints and routes ambiguous spikes to human review).
    - Reordered inspection findings display order to match 3-level precedence: Uncertain (1st), Candidate Defects (2nd), Potential Anomalies (3rd).
  - **End-to-End Product Readiness Verification (`scratch/verify_product_readiness.py`):**
    - Verified dynamic model registry with LRU caching (capacity = 3) and eviction on 4 categories (`bottle`, `cable`, `metal_nut`, `screw`).
    - Verified real inspections on clean cable, missing wire, puncture, bottle, metal nut, screw.
    - Verified SQLite persistence, report detail retrieval, and JSON export serialization (14,561 bytes) with disclaimer.
  - **Full Unit Test Suite:** **74 / 74 tests passed in 48.38s**.

- [x] **Specialized Multi-Detector Cable Inspection Module & Finding Categorization**
  - Investigated root causes of low baseline PaDiM cable sensitivity:
    - Missing wire (mean score 21.74 vs clean 22.77) matches background Gaussian distribution (negative anomaly problem).
    - Poke insulation ($98 \times 114$ px) diluted by default $\sigma=4.0$ smoothing (max score 21.27–27.99 vs threshold 30.42).
  - Implemented `ConductorContinuityDetector` in `src/app/vision/cable_inspector.py`: evaluates transmission density across top, left, and right harness divergence channels. Threshold calibrated on seed 42 calibration split ($T=0.50$).
  - Implemented `SurfacePunctureDetector` in `src/app/vision/cable_inspector.py`: evaluates high-frequency $\sigma=1.0$ patch residual maps (poke peaks reach 33.58–54.11 vs clean 24.42–28.41). Threshold calibrated on seed 42 calibration split ($T=38.0$).
  - Implemented `SpecializedCableInspector` composite inspection engine.
  - Benchmarked baseline PaDiM vs Specialized Inspector head-to-head across all 75 held-out evaluation samples (seed 42, zero leakage):
    - `poke_insulation` recall: **100.0% (6/6)** vs baseline 0.0% (0/6) — **+100.0% gain**.
    - `missing_wire` recall: **62.5% (5/8)** vs baseline 12.5% (1/8) — **+50.0% gain**.
    - Overall sensitivity: **89.13% (41/46)** vs baseline 60.87% (28/46) — **+28.26% recall gain**.
    - False negatives reduced from 18 to 5.
    - Balanced accuracy: **77.32%** vs baseline 71.81% (+5.51% gain).
    - Latency: 358.39 ms vs baseline 156.11 ms.
  - Implemented tripartite finding categorization across schemas (`InspectionReport.categorized_findings`), `ReportGenerator`, and UI (`frontend/index.html`):
    - 🔴 *Confirmed visual defects*
    - 🟡 *Potential anomalies*
    - 🔵 *Uncertain findings requiring human review*
  - Added unit test suite `tests/unit/test_cable_inspector.py` (6 tests) and extended `test_report.py` (9 tests).
  - Total unit test suite: **69 / 69 passing in 79.09s**.
  - Recorded benchmark results in `runs/cable_specialized_evaluation.json`.

- [x] **Multi-Detector Decision & Reporting System Validation & Operational Verification**
  - Inspected decision rules and resolved overlapping conditions between Potential Anomalies and Human Review:
    - Made all three categories strictly mutually exclusive: borderline zone ($|\Delta| < 3.0$) routes exclusively to `uncertain_human_review`, high-margin ($\Delta \ge 5.0$) or corroborated consensus routes to `candidate_defects`, and moderate single-detector activations ($3.0 \le \Delta < 5.0$) route to `potential_anomalies`.
    - Prohibited labeling algorithmic predictions as confirmed physical defects: added `is_human_confirmed: bool = False` and titled findings as Candidate Defect / Potential Anomaly / Uncertain Finding.
  - Implemented multi-detector provenance tracking with `FindingProvenance` (`detector`, `score`, `threshold`, `margin`, `location`, `bbox`) and `CategorizedFinding`.
  - Implemented duplicate finding prevention using unique deterministic IDs.
  - Resolved detector disagreements (e.g. PaDiM nominal vs localized puncture trigger) by routing uncorroborated single-detector boundary spikes to `uncertain_human_review` with actionable review instructions.
  - Independent Held-Out Evaluation on untouched evaluation split (75 images, seed 42, `runs/cable_consensus_evaluation.json`):
    - Baseline PaDiM: 28 TP / 5 FP / 24 TN / 18 FN | Sens: 60.87% | Spec: 82.76% | FPR: 17.24% | BalAcc: 71.81%
    - Naive OR Multi-Detector: 41 TP / 9 FP / 20 TN / 5 FN | Sens: 89.13% | Spec: 68.97% | FPR: 31.03% | BalAcc: 79.05%
    - Consensus & Masked Inspector: 37 TP / 8 FP / 21 TN / 9 FN | Sens: 80.43% | Spec: 72.41% | FPR: 27.59% | BalAcc: 76.42%
  - Measured detailed wall-clock latency across 20 test images:
    - PaDiM alone: Mean 148.23 ms | Median 147.47 ms | P95 155.88 ms
    - Conductor Continuity: Mean 60.39 ms | Median 59.57 ms | P95 64.37 ms
    - Surface Puncture: Mean 93.58 ms | Median 93.43 ms | P95 95.73 ms
    - Composite Vision: Mean 355.84 ms | Median 352.08 ms | P95 374.28 ms
    - Hybrid Retrieval: Mean 14.20 ms | Median 14.17 ms | P95 14.90 ms
    - SQLite DB History: Mean 0.62 ms | Median 0.60 ms | P95 0.83 ms
    - Report Synthesis: Mean 0.11 ms | Median 0.09 ms | P95 0.17 ms
    - Total End-to-End: Mean 370.77 ms | Median 367.40 ms | P95 389.10 ms
  - Verified operational inspection via FastAPI backend and service layer on real images:
    - Missing wire (defective conductor continuity alert)
    - Clean cable (normal verdict, no false alarms)
    - Puncture insulation (puncture detector activation, provenance, coordinates)
    - Verified `/api/reports` and `/api/reports/{id}` detail retrieval
    - Verified full JSON download serialization (12,275 bytes) with disclaimer and non-probabilistic distances.
  - Enhanced dashboard (`frontend/index.html`) with mutually exclusive categories, provenance pills, and actionable human review alerts.
  - Total unit test suite: **73 / 73 passed in 53.45s** across 11 test suites.

- [x] **Unit Test Suite Verification (73 / 73 Passed)**
  - `tests/unit/test_schemas.py`: 4 passed (0.31s)
  - `tests/unit/test_db.py`: 2 passed (0.72s)
  - `tests/unit/test_ingest.py`: 5 passed (1.23s)
  - `tests/unit/test_model_registry.py`: 8 passed (LRU cache accounting, eviction order, production discovery, memory reporting)
  - `tests/unit/test_patchcore.py`: 4 passed (PatchCore memory bank, feature extraction, anomaly maps, persistence)
  - `tests/unit/test_cable_inspector.py`: 6 passed (conductor continuity detector, surface puncture detector, composite inspection, finding categorization)
  - `tests/unit/test_report.py`: 13 passed (regression tests, category exclusivity, detector disagreement, duplicate prevention, evidence provenance)
  - `tests/unit/test_vision_validation.py`: 5 passed (14.45s)
  - `tests/unit/test_threshold_calibration.py`: 12 passed (15.58s) (disjoint split verification, certified sample counts, calibration threshold presence, metric arithmetic, seeded reproducibility)
  - `tests/unit/test_retriever.py`: 4 passed (38.71s)
  - `tests/unit/test_agent.py`: 10 passed (32.76s)
  - Total: **73 passing tests** across 11 unit test suites.

- [x] **Category-Specific Inspection Strategy & Empirical Model Comparison**
  - Investigated 2D in-plane rotation normalization for screw images (PCA-based major axis alignment and vertical rotation with head-at-top detection).
  - Proved empirically that 2D rotation normalization collapses PaDiM's held-out AUROC to 0.4374 and sensitivity to 1.69% due to border interpolation padding, 180° head/tip flip discontinuities on symmetrical machine screws, and failure to align 3D helical roll angles.
  - Implemented `PatchCore` in `src/app/vision/patchcore.py` as an experimental alternative model architecture (neighborhood-aware ResNet-18 layer2+3 embeddings + 10k memory bank).
  - Evaluated PatchCore on identical calibration and held-out splits:
    - Held-out specificity increased from 28.57% (PaDiM) to **47.62%** (PatchCore).
    - Held-out FPR decreased from 71.43% to **52.38%**.
    - Held-out Pixel AUROC remained state-of-the-art at **0.9709**.
    - Held-out Balanced Accuracy increased from 50.73% to **53.47%**.
  - Investigated cable held-out sensitivity (60.87%): proved that coarse defects achieve 85–100% recall, while `missing_wire` (90% missed) and `poke_insulation` (100% missed) are the sole failure modes due to negative anomalies (matching background) and microscopic puncture smoothing ($\sigma=4$).
  - Recorded complete comparative results in `runs/experimental_benchmarks.json`.
  - Preserved all production models and historical benchmark records intact. Zero tuning against held-out labels.

- [x] **Calibration Integrity Audit & Evaluation Protocol Separation**
  - Audited `scripts/calibrate_thresholds.py` and `scripts/validate_anomaly_detection.py`.
  - Identified calibration overlap in historical validation: 50% of images evaluated in `runs/validation_results.json` were used to pick decision thresholds.
  - Formally classified full-test confusion matrices as **threshold-selection-contaminated (apparent metrics)**.
  - Implemented strict evaluation protocol in `scripts/validate_anomaly_detection.py`:
    - `--split held_out`: Evaluates only untouched `ev` split. Zero leakage. Writes `runs/held_out_evaluation.json`.
    - `--split full`: Evaluates all test samples with explicit contamination notice. Writes `runs/validation_results.json`.
    - `--split calibration`: Evaluates calibration split.
  - Verified sample counts: bottle (10G/32D cal, 10G/31D ev), cable (29G/46D cal, 29G/46D ev), metal_nut (11G/46D cal, 11G/47D ev), screw (20G/60D cal, 21G/59D ev).
  - Recalculated strict held-out metrics across all 4 categories without touching benchmark records.
  - Added unit tests in `tests/unit/test_threshold_calibration.py` proving calibration and held-out paths are strictly disjoint.

- [x] **Screw Model Evaluation Disparity & Root Cause Investigation**
  - Investigated disparity between Pixel AUROC = `0.9726` and Image AUROC = `0.6280` / `0.5843` (held-out specificity = `28.57%`).
  - Proved clean screws exhibit raw max scores averaging `25.12`, exceeding the calibrated threshold `24.1733`.
  - Proved pooling sweeps (max, top-10, top-50, 99.5%, 99.0%) all yield AUROC ~0.62; pooling alone does not resolve the overlap.
  - Identified root cause: PaDiM models normal visual distribution per patch position $\mathcal{N}(\mu_{ij}, \Sigma_{ij})$, requiring spatial registration. 3D helical screw rotation shifts threads across the spatial grid, generating patch Mahalanobis spikes (25–35) on normal thread perimeters, overlapping subtle defects.
  - Explained why Pixel AUROC is 0.9726: over 8 million test pixels, 99.7% are normal with score ~8, so defect pixels (~30) rank higher. But Image AUROC relies on the worst single patch per image, triggering false alarms on rotating clean screws.

- [x] **Phase 1: Scalable Model Registry & LRU Cache Implementation**
  - Implemented `src/app/vision/registry.py` with `ModelRegistry`, `LRUModelCache` (capacity = 3), and `CategoryMetadata`.
  - Added thread-safe model eviction and automatic GPU/RAM cleanup (`gc.collect()`, `torch.cuda.empty_cache()`).
  - Added RAM working set and CUDA VRAM tracking (`get_process_memory_mb()`).
  - Updated `InspectionService` and `routes.py` (`/categories`, `/categories/metadata`, `/system/cache`).
  - Verified dynamic frontend dropdown population without duplicate appending.

- [x] **Phase 2: MVTec AD `screw` Category Integration**
  - Downloaded `screw` dataset (320 train normal, 160 test: 41 good, 119 defective) from Hugging Face mirror.
  - Fitted PaDiM model on 320 train images in 103.5s on CUDA (normalization: [11.30, 15.49], saved to `data/index/vision/screw/padim.pkl`, 662 MB).
  - Calibrated raw decision threshold in 19.7s on CUDA (`threshold_raw` = 24.1733, balanced acc = 0.7333 calib / 0.5076 eval).
  - Evaluated computer vision benchmark in 17.3s on CUDA: Image AUROC = 0.6280, Pixel AUROC = 0.9726.
  - Generated full validation report across all 160 test images (TP=86, FP=20, TN=21, FN=33, balanced acc = 0.6174).
  - Measured empirical memory and verified LRU eviction: 4th model load (`screw`) evicted oldest model (`bottle`), bounding active RAM at ~2.68 GB and CUDA VRAM at ~131 MB.
  - Added category limitation and inspection guidance in `report.py` and `frontend/index.html`.
  - Verified live end-to-end inspection on sample `screw/test/manipulated_front/000.png`.

- [x] **Calibration Integrity & Authority Boundary Review**
  - Confirmed calibration split methodology (`split_stratified` 50/50 disjoint split with seed 42) ensures 0% test leakage.
  - Verified 50-seed Monte Carlo stability cross-validation reproducibility.
  - Inspected all 7 false positives (6 cable twist/bend variants, 1 metal nut crop-boundary reflection) and 41 false negatives (cable low-activation defects: `poke_insulation` 10/10, `missing_wire` 9/10; metal nut margin defects).
  - Enhanced dashboard and report generator:
    - Non-probabilistic score framing (explicit Mahalanobis distance & threshold margin).
    - Category-specific benchmark reliability tiers.
    - Automated borderline score alert ($|raw - threshold| < 3.0$) with human-review requirement.
    - Persistent authority boundary banner ("Decision-support system, not autonomous quality authority").

- [x] **Anomaly Detection Reliability Validation (Complete 348 MVTec AD Samples)**
  - Evaluated all known-good and known-defective images for `bottle` (83), `cable` (150), and `metal_nut` (115) with calibrated thresholds.
  - Calculated confusion matrices, sensitivity, specificity, FPR, FNR, balanced accuracy, and AUROC for each category.
  - Confirmed AUROCs match benchmark values: Bottle 1.0000, Cable 0.8413, Metal Nut 0.9741.
  - Investigated clean false-positive samples:
    - Bottle: 0 false positives (100% specificity).
    - Metal Nut: 1 false positive (`006.png`, score 30.0865, edge artifact at `y=0`; 95.45% specificity).
    - Cable: 6 false positives (scores 31.42–35.80, twisted normal cables; 89.66% specificity).
  - Saved full validation records to `runs/validation_results.json` without modifying existing benchmark files.

- [x] **Report Generation Data-Contract Bug Fix**
  - Identified data-contract mismatch where `agent.py` serialized `RetrievalResult` into a dictionary when invoking `generate_inspection_report`, causing `_t_report` to pass a dict to `ReportGenerator._deterministic` which accessed `retrieval.chunks` directly (`AttributeError: 'dict' object has no attribute 'chunks'`).
  - Implemented defense-in-depth deserialization and coercion in both `src/app/agent/tools.py` (`_t_report`) and `src/app/reporting/report.py` (`generate()`, `_deterministic()`).
  - Added 3 regression tests covering dict retrieval inputs, empty chunks handling, and agent tool execution.
  - Verified full inspection pipeline with report generation and SQLite persistence.

- [x] **Live End-to-End Inference & Pipeline Verification**
  - Executed end-to-end inspection on sample defective MVTec AD bottle (`data/raw/mvtec_ad/bottle/test/broken_large/000.png`):
    - Raw score: `64.897` vs calibrated threshold `23.4613`.
    - Verdict: `is_anomalous = True` (Correct).
    - Defect regions: 1 contour localized.
    - Heatmap overlay: generated and written to `000.png`.
    - Summary & citations: Extractive report generated with 5 harvested findings, 1 recommendation, and 3-step agent trace in 4.26s.
  - Verified SQLite persistence: inspection recorded with ID 1 and retrievable via `reports()`.

- [x] **API Route Enhancements**
  - Added `GET /api/eval/cv` returning real benchmark evaluation data from `runs/eval_cv.json`.
  - Added `GET /api/eval/calibration` returning real calibration stability sweeps from `runs/calibration.json`.

- [x] **Premium Dark Industrial Dashboard Frontend**
  - Verified and enhanced `frontend/index.html` with:
    - Live health check chip and status indicator.
    - Real-time Overview with dynamic model performance charts and calibration metrics pulled from `/api/eval/`.
    - Drag-and-drop Image Upload with live canvas preview and category auto-detection.
    - Anomaly Heatmap viewer with defect region breakdown.
    - Retrieved technical evidence citation viewer.
    - Agent execution trace step visualizer.
    - SQLite inspection history table with filtering and instant modal preview.
    - Knowledge Base interactive search and document ingestion.
    - Report viewer with print and structured JSON export.

- [x] **Deployment Configuration**
  - Created production `Dockerfile` with multi-stage system libraries (`libgl1`, `libglib2.0-0`), CPU PyTorch wheel for portable execution, health checks, and Uvicorn entrypoint.
  - Created `.dockerignore` excluding caches, logs, and raw datasets.
  - Created `docker-compose.yml` with persistent volume mappings.

---

## Current / Ongoing Tasks

- [ ] **10-Category Expansion Pipeline Execution (Planned)**
  - Stage 1: Download raw datasets for 7 target categories (`capsule`, `pill`, `screw`, `transistor`, `tile`, `grid`, `zipper`) via `scripts/download_mvtec.py`.
  - Stage 2: Fit PaDiM weights via `scripts/fit_padim.py` (~15s per category on CUDA).
  - Stage 3: Calibrate decision thresholds via `scripts/calibrate_thresholds.py` (50-seed Monte Carlo sweeps).
  - Stage 4: Run CV benchmark evaluation and record in separate evaluation files.
  - Stage 5: Ingest matching technical specifications into Knowledge Base (`data/kb/`).
  - Stage 6: Verify API dynamic category discovery and dashboard dropdown population.

---

## Pending Tasks

- [ ] Interactive live browser test verification (launching uvicorn outside sandbox or on user command).
- [ ] Add Git LFS tracking configuration for remote repositories (for 6.62 GB model pickle footprint).
