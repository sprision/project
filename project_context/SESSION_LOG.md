# SESSION_LOG.md

## Chronological Development Handoffs

### Session 0: Initial Foundation & Previous Engineering Work
- **Context:** Initial baseline construction performed via Claude Code.
- **Implemented:**
  - PaDiM anomaly detection using ResNet-18 on MVTec AD (`bottle`, `cable`, `metal_nut`).
  - Three models fitted and saved in `data/index/vision/`.
  - Hybrid RAG retrieval index seeded with 4 technical documents (17 chunks).
  - 6-tool deterministic orchestrator agent and SQLite history database.
  - Initial evaluation runs recorded:
    - `runs/eval_cv.json`: Bottle AUROC 1.0, Cable AUROC 0.8413, Metal Nut AUROC 0.9741.
    - `runs/eval_retrieval.json`: Recall@5 = 1.000, MRR = 0.844 across 8 queries.
  - Threshold calibration performed and saved in `data/index/vision/*/threshold.json` and `runs/calibration.json`.
  - Inference optimization in `src/app/vision/model.py` using cached batched matrix multiplication (`torch.bmm`).
  - Unit and integration tests authored across 10 test files.
- **Previous Issues:** Upstream LLM provider API failures encountered when invoking external models; confirmed as provider failure rather than application code defect.

---

### Session 1: State Recovery & Persistent Project Memory Establishment
- **Date & Local Time:** 2026-11-07T12:00:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Actions Taken:**
  1. Verified virtual environment, PyTorch with CUDA acceleration, timm, and sentence-transformers.
  2. Established persistent project memory (`AGENTS.md` + 7 context files in `project_context/`).
  3. Confirmed code integrity and preserved all existing artifacts and fitted models.

---

### Session 2: Implementation Completion & Full Verification
- **Date & Local Time:** 2026-11-07T12:24:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Changes Made:**
  1. **Hybrid RAG Offline Optimization:**
     - In `src/app/retrieval/retriever.py`, added `local_files_only=True` to `SentenceTransformer` loading with graceful fallback. This eliminated 5 successive DNS retry loops (60s delay) when operating in offline/sandboxed environments.
  2. **Evaluation API Routes:**
     - In `src/app/api/routes.py`, added `GET /api/eval/cv` (serving `runs/eval_cv.json`) and `GET /api/eval/calibration` (serving `runs/calibration.json`) so the dashboard displays genuine benchmark evaluation metrics directly from the backend.
  3. **Frontend Dashboard Connection:**
     - In `frontend/index.html`, updated `loadEvalResults()` and `loadCalibration()` to query the backend endpoints `/api/eval/cv` and `/api/eval/calibration` with clean table rendering and progress bars.
  4. **Deployment Packaging:**
     - Created production `Dockerfile` with system libraries (`libgl1`, `libglib2.0-0`), CPU-compatible PyTorch for portable deployment, healthcheck probe, and Uvicorn entrypoint.
     - Created `.dockerignore` to exclude development caches and raw datasets.
     - Created `docker-compose.yml` with persistent volume mounts for uploads, heatmaps, and knowledge indices.
- **Test Results Verified:**
  - `tests/unit/test_schemas.py`: 4 passed (0.31s)
  - `tests/unit/test_db.py`: 2 passed (0.72s)
  - `tests/unit/test_ingest.py`: 5 passed (1.23s)
  - `tests/unit/test_report.py`: 5 passed (0.55s)
  - `tests/unit/test_vision_validation.py`: 5 passed (14.45s)
  - `tests/unit/test_threshold_calibration.py`: 6 passed (14.81s)
  - `tests/unit/test_retriever.py`: 4 passed (38.71s)
  - `tests/unit/test_agent.py`: 9 passed (41.19s)
  - Total: **40 passing unit tests**.
- **End-to-End Pipeline Verification:**
  - Tested live PaDiM inspection on real MVTec AD bottle defective image (`data/raw/mvtec_ad/bottle/test/broken_large/000.png`):
    - Raw anomaly score: 64.897 vs calibrated threshold 23.4613.
    - Anomaly verdict: True (detected).
    - Defect regions: 1 localized region.
    - Heatmap overlay: generated and written to `000.png`.
    - Agent trace: 3 deliberated tool steps.
    - Evidence harvested: 5 items.
    - Execution latency: 4.26 seconds.
    - SQLite persistence: Verified and retrievable via `reports()`.
- **Next Steps:**
  - To launch the web application for live user interaction:
    Run `.\.venv\Scripts\uvicorn.exe src.app.main:app --host 0.0.0.0 --port 8000` (outside sandbox for external browser access).

---

### Session 3: Report Generation Data-Contract Bug Fix & Regression Testing
- **Date & Local Time:** 2026-11-07T13:00:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:** Fix `AttributeError: 'dict' object has no attribute 'chunks'` occurring during report generation in end-to-end inspection pipeline.
- **Root Cause Analysis:**
  - In `src/app/agent/agent.py`, when invoking the `generate_inspection_report` tool, the agent serialized the retrieval result as a dict: `retrieval.model_dump() if retrieval else None`.
  - In `src/app/agent/tools.py`, `_t_report` forwarded `args.get("retrieval")` (dict) directly into `ctx.report_fn`.
  - In `src/app/reporting/report.py`, `_deterministic()` checked `if retrieval and not retrieval.chunks:`, causing python to throw an `AttributeError` on the raw dict.
- **Code Changes:**
  1. `src/app/agent/tools.py`:
     - Updated `_t_report` to deserialize dictionary `retrieval` to `RetrievalResult(**ret_arg)`.
     - Ensured `evidence` items in list are converted to `EvidenceItem` if dicts.
     - Updated `_ReportArgs` schema to allow `dict | Any`.
  2. `src/app/reporting/report.py`:
     - Updated `generate()` and `_deterministic()` with defense-in-depth coercion: `if isinstance(retrieval, dict): retrieval = RetrievalResult(**retrieval)`.
     - Used safe chunk access `chunks = getattr(retrieval, "chunks", None)`.
  3. `tests/unit/test_report.py`:
     - Added `test_regression_retrieval_as_dict`: verifies report generation succeeds when `retrieval` is passed as a dict.
     - Added `test_regression_retrieval_as_dict_empty_chunks`: verifies dict with empty chunks properly appends limitation.
  4. `tests/unit/test_agent.py`:
     - Added `test_regression_generate_report_with_dict_retrieval`: verifies `run_tool("generate_inspection_report", ...)` handles dict retrieval and returns `InspectionReport`.
- **Verification Results:**
  - Full unit test suite: **43 / 43 passed** in 33.41s across all 8 suites (`test_schemas.py`, `test_db.py`, `test_ingest.py`, `test_report.py`, `test_vision_validation.py`, `test_threshold_calibration.py`, `test_retriever.py`, `test_agent.py`).
  - End-to-end inspection pipeline verified on defective MVTec AD bottle (`data/raw/mvtec_ad/bottle/test/broken_large/000.png`): All tool calls succeeded (`inspect_image`, `retrieve_technical_documents`, `summarize_evidence`, `generate_inspection_report`), SQLite database recorded the report, and report JSON export is verified.
- **Preservation Check:** All existing models (`data/index/vision/`) and evaluation benchmarks (`runs/`) remain intact.

---

### Session 4: Comprehensive Anomaly Detection Reliability Validation
- **Date & Local Time:** 2026-11-07T13:40:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:** Validate PaDiM anomaly detection reliability across all known-good and known-defective samples in MVTec AD (`bottle`, `cable`, `metal_nut`) using existing calibrated thresholds.
- **Commands Executed:**
  - `.\.venv\Scripts\python.exe -m scripts.validate_anomaly_detection`
  - `.\.venv\Scripts\pytest.exe tests\unit\test_threshold_calibration.py -v`
  - `.\.venv\Scripts\pytest.exe tests\unit\ -v`
- **Measured Results Across All 348 MVTec AD Test Images:**
  1. **Bottle (83 images: 20 good, 63 defective; Threshold 23.4613):**
     - Confusion Matrix: TP=63, FP=0, TN=20, FN=0
     - Sensitivity: 1.0000 | Specificity: 1.0000 | Balanced Accuracy: 1.0000 | FPR: 0.0000
     - Image AUROC: 1.0000 (Matches `runs/eval_cv.json`)
     - False positives: 0. False negatives: 0.
  2. **Cable (150 images: 58 good, 92 defective; Threshold 30.4218):**
     - Confusion Matrix: TP=60, FP=6, TN=52, FN=32
     - Sensitivity: 0.6522 | Specificity: 0.8966 | Balanced Accuracy: 0.7744 | FPR: 0.1034
     - Image AUROC: 0.8413 (Matches `runs/eval_cv.json`)
     - False Positives (6 clean samples): `cable/test/good/007.png` (34.37), `008.png` (34.46), `016.png` (34.09), `019.png` (35.72), `036.png` (31.42), `041.png` (35.80) due to natural cable twists/sheath folds.
     - False Negatives (32 defective samples): primarily `poke_insulation` (10/10 missed, scores 21.2–27.9) and `missing_wire` (9/10 missed, scores 15.8–29.0) where subtle pinprick or missing conductor features do not create high patch activation.
  3. **Metal Nut (115 images: 22 good, 93 defective; Threshold 23.1698):**
     - Confusion Matrix: TP=84, FP=1, TN=21, FN=9
     - Sensitivity: 0.9032 | Specificity: 0.9545 | Balanced Accuracy: 0.9289 | FPR: 0.0455
     - Image AUROC: 0.9741 (Matches `runs/eval_cv.json`)
     - False Positives (1 clean sample): `metal_nut/test/good/006.png` (30.0865) due to an edge feature at image crop boundary (`y=0`).
     - False Negatives (9 defective samples): subtle scratches (4) and minor bends (5) scoring between 17.8 and 23.1.
- **Unit Test Additions:**
  - Added `test_calibrated_threshold_files_exist_and_positive` and `test_metrics_calculation_known_values` in `tests/unit/test_threshold_calibration.py`.
  - All 45 unit tests pass in 33.5s.
- **Artifacts Created:**
  - `scripts/validate_anomaly_detection.py` (Validation script).
  - `runs/validation_results.json` (Full 348-image record without touching `runs/eval_cv.json` or `runs/calibration.json`).
- **Preservation Check:** All existing PaDiM pickle weights in `data/index/vision/` and benchmark runs in `runs/` preserved unmodified.

---

### Session 5: Calibration Integrity Review & Decision-Support Dashboard Hardening
- **Date & Local Time:** 2026-11-07T13:52:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:** Verify calibration integrity, dataset disjointness, error distributions, and enhance the dashboard and report generator with decision-support authority boundaries, non-probabilistic score framing, and borderline human-review recommendations.
- **Verification Findings:**
  1. **Calibration Separation Integrity:** In `scripts/calibrate_thresholds.py`, `split_stratified` partitions the test set into disjoint halves (seed 42). The threshold `threshold_raw` was selected solely on the calibration subset (`cal`) maximizing balanced accuracy. The held-out evaluation subset was untouched during threshold picking.
  2. **50-Seed Monte Carlo Stability:** Confirmed that across 50 random splits, metric variability is quantified (bottle mean bal-acc: 0.9751, metal nut: 0.9109, cable: 0.7373). Seeded reproducibility was verified with unit tests.
  3. **Defect Analysis of Errors:**
     - 7 False Positives: 1 on `metal_nut/good/006.png` caused by an edge reflection at `y=0`; 6 on `cable/good/` caused by normal cable curvature/twist shadows scoring 31.42–35.80 vs threshold 30.42.
     - 41 False Negatives: Cable defects with low patch activation (`poke_insulation` 10/10 missed, `missing_wire` 9/10 missed) blend into normal background feature distributions; Metal nut defects (4 scratches, 5 bends) lying near the margin of threshold 23.17.
- **Code & UI Enhancements:**
  1. **Dashboard UI (`frontend/index.html`):**
     - Added persistent decision-support notice banner at the top of content.
     - Framed anomaly scores explicitly as unnormalized spatial Mahalanobis distances and threshold margins (not probabilities).
     - Added category benchmark reliability indicators (`bottle` high, `metal_nut` high, `cable` moderate).
     - Added automated borderline score alert card ($|raw - threshold| < 3.0$) mandating secondary human visual review.
     - Added category limitation cards for cable geometry overlap and metal nut boundary edge reflections.
     - Added authority boundary notice in the full report viewer.
  2. **Reporting Layer (`src/app/reporting/report.py`):**
     - Automatically generates borderline score recommendations and limitations when raw score is within 3.0 points of threshold.
     - Automatically includes cable category score overlap caveats and metal nut boundary reflection notices.
  3. **Unit Tests:**
     - Added `test_calibration_split_stratified_disjoint_and_reproducible` and `test_pick_threshold_maximizes_balanced_accuracy` in `tests/unit/test_threshold_calibration.py`.
     - Added `test_borderline_score_generates_human_review_recommendation` in `tests/unit/test_report.py`.
     - All **48 unit tests passing** across 8 test suites.
- **Preservation Check:** All existing models (`data/index/vision/`) and evaluation benchmarks (`runs/`) remain intact.

---

### Session 6: 10-Category Expansion Feasibility Study & Architecture Design
- **Date & Local Time:** 2026-11-07T13:58:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:** Evaluate feasibility and design the architectural roadmap to expand the inspection system from 3 to 10 MVTec AD categories, reusing existing pipelines while strictly preserving current models and benchmarks.
- **Key Findings & Quantitative Estimates:**
  1. **Category Selection (7 new + 3 existing = 10 total):**
     - Fasteners: `metal_nut` (existing), `screw` (new).
     - Electronics: `cable` (existing), `transistor` (new).
     - Pharmaceuticals & Packaging: `bottle` (existing), `capsule` (new), `pill` (new).
     - Surface Textures & Closures: `tile` (new), `grid` (new), `zipper` (new).
  2. **Computational Effort:**
     - Training (patch Gaussian covariance fit): ~15–20s per category on CUDA GPU (~2 mins total for 7 categories).
     - Threshold Calibration (50-seed Monte Carlo): ~25–35s per category (~3.5 mins total).
     - CV AUROC Evaluation: ~20s per category (~2.3 mins total).
     - Combined automated execution time: **~7 to 10 minutes on CUDA GPU** (~35 mins on CPU).
  3. **Storage & Memory Footprint:**
     - Raw Dataset: ~2.1 GB download from Hugging Face mirror `foersben/mvtec-ad`.
     - Model Pickles: 662 MB per category $\implies$ 6.62 GB total across 10 categories.
     - In-Memory RAM/VRAM: Lazy loading prevents loading all models into memory; LRU cache (capacity = 3) bounds active VRAM to ~1.98 GB.
  4. **Architecture Compatibility:**
     - Dynamic discovery via `InspectionService.available_categories()` automatically indexes any subfolder with `padim.pkl` and `threshold.json`.
     - Core API routes (`GET /api/categories`, `POST /api/inspect`) and frontend dropdown require zero code modifications.
- **Context Updates:** Documented in `ARCHITECTURE.md`, `DECISIONS.md`, `TASK_TRACKER.md`, and `SESSION_LOG.md`.
- **Preservation Check:** All existing models and evaluation benchmarks strictly untouched; no training executed in this turn.

---

### Session 7: Scalable Model Registry Implementation & MVTec AD `screw` Integration
- **Date & Local Time:** 2026-11-07T14:27:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:** Implement Phase 1 (ModelRegistry, LRU cache capacity=3, memory tracking) and Phase 2 (MVTec AD `screw` category download, PaDiM fit, threshold calibration, CV evaluation, validation, and dashboard integration).
- **Phase 1 Implementation:**
  - Created `src/app/vision/registry.py` with `ModelRegistry`, `LRUModelCache`, `CategoryMetadata`, and `get_process_memory_mb()`.
  - Updated `InspectionService` (`services.py`) to manage models through the registry cache.
  - Added API endpoints: `GET /api/categories`, `GET /api/categories/metadata`, `GET /api/system/cache`.
  - Added unit test suite `tests/unit/test_model_registry.py` (8 passing tests).
- **Phase 2 Implementation & Actual Measured Metrics:**
  - **Dataset:** Downloaded 599 PNG files for MVTec AD `screw` from `foersben/mvtec-ad` (320 train normal, 160 test: 41 good, 119 defective).
  - **Training:** Fitted PaDiM on 320 train images in **103.5s on CUDA GPU** (normalization: [11.30, 15.49], saved to `data/index/vision/screw/padim.pkl`, 661.9 MB).
  - **Calibration:** Executed seeded 50/50 split and 50-seed Monte Carlo sweep in **19.7s on CUDA GPU** (`threshold_raw` = 24.1733, balanced accuracy = 0.7333 calibration / 0.5076 evaluation, FPR = 0.7143).
  - **Computer Vision Benchmarking:** Evaluated 160 test images in **17.3s on CUDA GPU**:
    - **Image AUROC: 0.6280**
    - **Pixel AUROC: 0.9726**
  - **Validation Report Across 160 Test Samples:**
    - Confusion Matrix: TP = 86, FP = 20, TN = 21, FN = 33
    - Sensitivity: 0.7227 | Specificity: 0.5122 | FPR: 0.4878 | Balanced Accuracy: 0.6174
    - Breakdown by Defect: `manipulated_front` 24/24 (100% recall, 0 errors, mean 32.70), `thread_top` 18/23 (mean 27.56), `scratch_head` 19/24 (mean 26.54), `thread_side` 12/23 (mean 26.14), `scratch_neck` 13/25 (mean 24.62). Clean screws (`good`: 21/41 normal, 20 false positives, mean 25.12).
    - Domain Finding: Arbitrary screw thread rotational orientation elevates patch feature Mahalanobis distances on normal screws, while pixel localization is outstanding at 0.9726.
  - **Empirical Memory & LRU Eviction Measurements:**
    - Baseline: RAM 650.45 MB | CUDA 0.0 MB
    - 1 Model (`bottle`): RAM 1431.28 MB | CUDA 42.69 MB
    - 2 Models (`bottle`, `cable`): RAM 2065.50 MB | CUDA 87.12 MB
    - 3 Models (`bottle`, `cable`, `metal_nut` — Full Cache): RAM 2660.59 MB | CUDA 130.68 MB
    - 4th Model (`screw`): **Oldest model (`bottle`) safely evicted!**
    - Active Cache: `['cable', 'metal_nut', 'screw']`
    - Post-eviction working set: RAM 2688.72 MB | CUDA 131.56 MB (bounded at ~2.68 GB RAM and ~131 MB VRAM).
- **Verification:**
  - Tested live end-to-end inspection on sample `data/raw/mvtec_ad/screw/test/manipulated_front/000.png`: correctly flagged anomalous with raw score `31.60` vs threshold `24.1733`, 2 regions localized, and heatmap generated.
  - Full unit test suite: **56 / 56 passed in 37.02s** across 9 test suites.
### Session 8: Calibration Integrity Audit, Protocol Separation & Screw Root Cause Investigation
- **Date & Local Time:** 2026-11-07T14:45:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Audit calibration image sources and verify overlap with reported final validation metrics.
  2. Classify calibration-overlap metrics and establish strict, leak-free held-out evaluation protocol.
  3. Investigate the disparity between Screw Pixel AUROC = 0.9726 and Image AUROC = 0.6280 (held-out specificity = 28.57%).
  4. Preserve all existing benchmark files, add unit tests, and update project context.
- **Audit Findings (Calibration Overlap & Contamination Classification):**
  - In `scripts/calibrate_thresholds.py`, the test set is split 50/50 into calibration (`cal`) and evaluation (`ev`) subsets using seed 42. `pick_threshold` selects the threshold strictly on `cal`.
  - However, `scripts/validate_anomaly_detection.py` originally evaluated all test images (`cal` + `ev`). Exactly 50% of images evaluated in `runs/validation_results.json` were in the calibration pool used to pick thresholds!
  - The historical full-test confusion matrix for screw (TP=86, FP=20, TN=21, FN=33, specificity 51.22%) is literally the sum of the calibration split (TP=43, FP=5, TN=15, FN=17, specificity 75.0%) and the held-out split (TP=43, FP=15, TN=6, FN=16, specificity 28.57%).
  - Formally classified all full-test confusion matrices as **threshold-selection-contaminated (apparent metrics)**.
- **Strict Separation Protocol Implemented:**
  - Updated `scripts/validate_anomaly_detection.py` with explicit `--split` options:
    - `--split held_out`: Evaluates strictly untouched evaluation half (`ev`, seed 42). Writes `runs/held_out_evaluation.json`.
    - `--split full`: Evaluates full test set with explicit contamination warning. Writes `runs/validation_results.json`.
    - `--split calibration`: Evaluates calibration split (`cal`).
  - Strict Held-Out Evaluation Results (254 Untouched Images):
    - **Bottle (41 imgs: 10G / 31D):** TP=31, FP=0, TN=10, FN=0 | Sens: 1.0000 | Spec: 1.0000 | FPR: 0.0000 | Bal Acc: 1.0000
    - **Cable (75 imgs: 29G / 46D):** TP=28, FP=5, TN=24, FN=18 | Sens: 0.6087 | Spec: 0.8276 | FPR: 0.1724 | Bal Acc: 0.7181
    - **Metal Nut (58 imgs: 11G / 47D):** TP=45, FP=0, TN=11, FN=2 | Sens: 0.9574 | Spec: 1.0000 | FPR: 0.0000 | Bal Acc: 0.9787
    - **Screw (80 imgs: 21G / 59D):** TP=43, FP=15, TN=6, FN=16 | Sens: 0.7288 | Spec: 0.2857 | FPR: 0.7143 | Bal Acc: 0.5073
  - All historical benchmark files (`runs/calibration.json`, `runs/eval_cv.json`, `runs/validation_results.json`) preserved intact.
- **Screw Disparity Empirical Investigation & Root Cause:**
  - Tested pooling alternatives (Top-10 mean, Top-50 mean, 99.5%, 99.0% percentiles): all yielded AUROC ~0.62. Pooling alone does not resolve the overlap.
  - Proved clean screws exhibit max scores averaging 25.12, exceeding the threshold of 24.1733, directly overlapping subtle defect scores (scratch_neck avg 24.62, thread_side avg 26.14, scratch_head avg 26.54, thread_top avg 27.56).
  - Root cause: PaDiM models the normal visual distribution per patch position $\mathcal{N}(\mu_{ij}, \Sigma_{ij})$, requiring spatial registration. 3D helical screw rotation shifts threads across the spatial grid. At unseen rotation angles, clean threads generate Mahalanobis spikes (25–35) along metallic boundaries, numerically indistinguishable from subtle scratches and missing threads at the image level.
  - Pixel AUROC is 0.9726 because over 8 million test pixels, 99.7% are normal background (score ~8) and defect pixels (~30) rank higher. But Image AUROC relies on the worst single patch per image, triggering false alarms on rotating clean screws.
- **Unit Tests Added:**
  - `test_all_categories_calibration_and_evaluation_splits_strictly_disjoint`: proves `set(cal_paths).isdisjoint(set(ev_paths))` across all 4 categories.
  - `test_split_sizes_match_certified_protocol`: verifies exact counts in `cal` and `ev` splits.
  - Test Suite: **58 / 58 passed in 39.26s**.
- **Preservation Check:** All 4 trained PaDiM models (`bottle`, `cable`, `metal_nut`, `screw`), thresholds, and historical benchmark files strictly preserved.

---

### Session 9: Category-Specific Strategy, Rotation Normalization & PatchCore Evaluation
- **Date & Local Time:** 2026-11-07T15:05:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Investigate 2D rotation normalization for screw images without modifying production models.
  2. Research and implement PatchCore as an experimental alternative model architecture.
  3. Design a fair comparison using identical calibration and held-out splits (seed 42, 50/50).
  4. Investigate cable's relatively low held-out sensitivity (60.87%).
  5. Preserve all historical benchmark files and production models.
- **Experimental Findings & Measured Comparisons:**
  1. **Rotation Normalization for Screw:**
     - Developed PCA-based centroid and major-axis orientation detector with head-at-top alignment.
     - Fitted PaDiM on 320 rotation-normalized training images and evaluated on identical calibration and held-out splits.
     - **Result:** Held-out Image AUROC collapsed from 0.5843 to **0.4374**, and Sensitivity collapsed to **1.69%** (1 TP, 58 FN).
     - **Mechanism:** In-plane 2D rotation introduces border interpolation padding, causes 180° orientation flip ambiguities on machine screws with subtle head/tip width differences, and fails to align the 3D helical roll angle around the cylindrical axis.
  2. **PatchCore Experimental Architecture:**
     - Implemented `PatchCore` in `src/app/vision/patchcore.py` (ResNet-18 layers 2+3, 10,000 coreset memory bank, nearest-neighbor Euclidean scoring).
     - Evaluated on identical calibration and held-out splits:
       - **Held-out Specificity:** Increased from **28.57% (PaDiM)** to **47.62% (PatchCore)** (10 TN vs 6 TN).
       - **Held-out False Positive Rate:** Decreased from **71.43%** to **52.38%**.
       - **Held-out Pixel AUROC:** Maintained state-of-the-art at **0.9709**.
       - **Held-out Balanced Accuracy:** Increased from 50.73% to **53.47%**.
       - **Image AUROC:** Measured at 0.5730 held-out / 0.6433 calibration.
  3. **Cable Sensitivity Deep-Dive:**
     - Evaluated all 150 test cables against calibrated threshold 30.4218:
       - `cut_outer_insulation`: 10/10 detected (100% recall, mean score 42.14)
       - `missing_cable`: 11/12 detected (91.7% recall, mean score 33.55)
       - `combined`: 10/11 detected (90.9% recall, mean score 41.86)
       - `bent_wire`: 11/13 detected (84.6% recall, mean score 42.24)
       - `cable_swap`: 9/12 detected (75.0% recall, mean score 34.02)
       - `cut_inner_insulation`: 8/14 detected (57.1% recall, mean score 33.37)
       - **`missing_wire`:** **1/10 detected (10.0% recall, 90% missed!)**, mean score = 21.74 vs clean cables 22.77. (Negative anomaly: missing conductor exposes background, matching background features seen in training).
       - **`poke_insulation`:** **0/10 detected (0.0% recall, 100% missed!)**, mean score = 24.87 vs threshold 30.42. (Microscopic punctures downsampled by layer3 and diluted by $\sigma=4$ Gaussian smoothing).
- **Unit Tests Added:**
  - `tests/unit/test_patchcore.py`: 4 tests (initialization, feature shapes, fit memory bank, anomaly map, save/load cycle).
  - Total test suite: **62 / 62 passed in 47.53s** across 10 test suites.
- **Benchmarking Artifact:**
  - Written to `runs/experimental_benchmarks.json`.
- **Preservation Check:** All production PaDiM models (`bottle`, `cable`, `metal_nut`, `screw`), thresholds, and historical benchmarks strictly preserved. Zero tuning against held-out labels.

---

### Session 10: Specialized Cable Inspection Module & Multi-Detector Architecture
- **Date & Local Time:** 2026-11-07T15:35:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Address the two documented failure modes of standard PaDiM on cables:
     - `missing_wire` (baseline recall: 12.5% on held-out split)
     - `poke_insulation` (baseline recall: 0.0% on held-out split)
  2. Implement experimental multi-detector inspection modules:
     - Conductor continuity and transmission density across wire divergence channels.
     - High-frequency local patch residual analysis ($\sigma=1.0$) for sub-millimeter punctures.
  3. Benchmark head-to-head on identical calibration (`cal`, 75 imgs) and held-out evaluation (`ev`, 75 imgs) splits (seed 42, zero leakage).
  4. Categorize findings into: Confirmed visual defects, Potential anomalies, and Uncertain findings requiring human review.
  5. Add unit tests, preserve production models, and update project context.
- **Experimental Findings & Empirical Validations:**
  1. **Conductor Continuity Detector (`ConductorContinuityDetector`):**
     - Measures cross-sectional transmission density across top, left, and right harness divergence channels.
     - On seed 42 calibration subset: clean cables scored $\le 0.433$, missing wire cables scored $\ge 0.635$.
     - Setting threshold $T=0.50$ achieved **0 false alarms on clean calibration cables**.
     - On strict held-out evaluation, `missing_wire` recall jumped from **12.5% (1/8) to 62.5% (5/8)** (+50.0% gain).
  2. **Surface Puncture Detector (`SurfacePunctureDetector`):**
     - Evaluates local patch anomaly activations with light Gaussian smoothing ($\sigma=1.0$) instead of default $\sigma=4.0$.
     - High-frequency patch peak scores for clean cables average 28.5 (max 33.1), while `poke_insulation` punctures surge to **33.58–54.11** (avg 45.2).
     - Setting threshold $T=38.0$ on calibration split completely separates puncturing.
     - On strict held-out evaluation, `poke_insulation` recall jumped from **0.0% (0/6) to 100.0% (6/6)** (+100.0% gain, all detected).
  3. **Strict Held-Out Head-to-Head Comparison (75 Untouched Images, `runs/cable_specialized_evaluation.json`):**
     - **Baseline PaDiM:** 28 TP / 5 FP / 24 TN / 18 FN | Sensitivity: **60.87%** | Specificity: **82.76%** | Balanced Accuracy: **71.81%** | Latency: 156.11 ms.
     - **Specialized Cable Inspector:** 41 TP / 10 FP / 19 TN / 5 FN | Sensitivity: **89.13%** | Specificity: **65.52%** | Balanced Accuracy: **77.32%** | Latency: 358.39 ms.
     - **Key Gains:**
       - `poke_insulation`: 0/6 (0.0%) -> **6/6 (100.0%)** (+100.0%)
       - `missing_wire`: 1/8 (12.5%) -> **5/8 (62.5%)** (+50.0%)
       - `bent_wire`: 4/5 (80.0%) -> **5/5 (100.0%)** (+20.0%)
       - `combined`: 3/4 (75.0%) -> **4/4 (100.0%)** (+25.0%)
       - `cut_inner_insulation`: 3/6 (50.0%) -> **4/6 (66.7%)** (+16.7%)
       - False negatives slashed by **72.2%** (from 18 missed defects down to only 5).
  4. **Tripartite Finding Categorization Integration:**
     - Integrated `categorized_findings` into `InspectionReport` schema and `ReportGenerator`:
       - 🔴 `confirmed_visual_defects`: High-margin anomalies ($\Delta \ge 5.0$) or multi-detector consensus.
       - 🟡 `potential_anomalies`: Single detector moderate activation ($0 < \Delta < 5.0$).
       - 🔵 `uncertain_human_review`: Borderline proximity ($|\Delta| < 3.0$) or natural cable curvature risk.
     - Enhanced `frontend/index.html` (both inline and full report modal) with styled categorization cards.
- **Unit Tests Added:**
  - `tests/unit/test_cable_inspector.py`: 6 tests (initialization, continuity detection on black canvas, puncture detection on uniform canvas, clean composite, defect augmentation).
  - `tests/unit/test_report.py`: 9 tests (added tripartite categorization test case).
  - Total test suite: **69 / 69 passed in 79.09s** across 11 test suites.
- **Preservation Check:** All production models (`bottle`, `cable`, `metal_nut`, `screw`), thresholds, and historical benchmarks strictly preserved. Zero tuning against held-out labels.

---

### Session 12: Independent Held-Out Evaluation, Latency Profiling & API Verification
- **Date & Local Time:** 2026-11-07T17:30:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Validate the consensus and boundary-masked cable inspection strategy on the untouched held-out evaluation split (75 images: 29 good, 46 defective, seed 42).
  2. Measure detailed wall-clock latencies (Mean, Median, P95) across vision, sub-detectors, retrieval, DB, and reporting stages.
  3. Seamlessly integrate `SpecializedCableInspector` into production `InspectionService` for category `"cable"`.
  4. Perform operational verification on real images (`missing_wire`, `good`, `poke_insulation`) across backend endpoints.
  5. Verify mutual exclusivity, provenance tracking, coordinates, actionable guidance, JSON download, and non-probabilistic score representation.
  6. Run full unit test suite (73 / 73 passing) and maintain persistent project memory.
- **Key Empirical Results & Verifications:**
  1. **Strict Held-Out 3-Way Comparison (`runs/cable_consensus_evaluation.json`):**
     - **Baseline PaDiM:** 28 TP / 5 FP / 24 TN / 18 FN | Sensitivity: **60.87%** | Specificity: **82.76%** | FPR: 17.24% | Balanced Accuracy: **71.81%**
     - **Naive OR Multi-Detector:** 41 TP / 9 FP / 20 TN / 5 FN | Sensitivity: **89.13%** | Specificity: **68.97%** | FPR: 31.03% | Balanced Accuracy: **79.05%**
     - **Consensus & Masked Inspector:** 37 TP / 8 FP / 21 TN / 9 FN | Sensitivity: **80.43%** | Specificity: **72.41%** | FPR: 27.59% | Balanced Accuracy: **76.42%**
     - Defect-specific recall: `bent_wire`: 100%, `cable_swap`: 100%, `combined`: 100%, `cut_inner`: 66.7%, `cut_outer`: 100%, `missing_cable`: 100%, `missing_wire`: 50.0%, `poke_insulation`: 50.0% (remaining 50% routed safely to `uncertain_human_review`).
  2. **Detailed Wall-Clock Latency Profiling (20 test images):**
     - `padim_only`: Mean: 148.23 ms | Median: 147.47 ms | P95: 155.88 ms
     - `continuity_only`: Mean: 60.39 ms | Median: 59.57 ms | P95: 64.37 ms
     - `puncture_only`: Mean: 93.58 ms | Median: 93.43 ms | P95: 95.73 ms
     - `composite_vision`: Mean: 355.84 ms | Median: 352.08 ms | P95: 374.28 ms
     - `hybrid_retrieval`: Mean: 14.20 ms | Median: 14.17 ms | P95: 14.90 ms
     - `sqlite_database`: Mean: 0.62 ms | Median: 0.60 ms | P95: 0.83 ms
     - `report_generation`: Mean: 0.11 ms | Median: 0.09 ms | P95: 0.17 ms
     - `total_end_to_end`: Mean: 370.77 ms | Median: 367.40 ms | P95: 389.10 ms
     - Overlap analysis: Disjoint sequential sum is 302.19 ms; composite vision is 355.84 ms (+53.65 ms) due to shared bounding box generation, contour extraction, finding reconciliation, and provenance assembly.
  3. **Backend Integration (`src/app/services/services.py`):**
     - Enhanced `inspect()` to route category `"cable"` dynamically through `SpecializedCableInspector(inspector, consensus_mode=True)` while preserving standard `AnomalyInspector` for other categories (`bottle`, `screw`, `metal_nut`).
  4. **Operational Inspection & API Verification (`scratch/verify_live_cable_inspection.py`):**
     - Defective missing wire: conductor continuity alert, routed to human review with specific action instructions.
     - Clean normal cable: clean verdict (`anomalous=False`, score 18.95 < threshold 30.42), 0 false alarms.
     - Microscopic puncture: high margin anomaly detected (`raw score 54.11 >= threshold 30.42`), localized puncture coordinates identified, actionable instructions: "Physically probe cable jacket at candidate puncture coordinates."
     - Data integrity: pairwise disjoint category sets, `is_human_confirmed == False`, complete detector provenance (`detector`, `score`, `threshold`, `location`, `bbox`), database persistence, and valid 12,275-byte JSON download verified.
- **Unit Test Results:**
  - Full suite verified: **73 / 73 passed in 53.45s**.
- **Preservation Check:** All existing production models (`data/index/vision/`) and historical benchmark results (`runs/eval_cv.json`, `runs/calibration.json`, `runs/held_out_evaluation.json`, `runs/cable_specialized_evaluation.json`) strictly preserved. Zero leakage or tuning against held-out labels.

---

### Session 13: Final Decision Logic & Product Readiness Pass
- **Date & Local Time:** 2026-11-07T17:45:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Inspect and enforce strict 3-level finding categorization precedence in `src/app/reporting/report.py`: uncertain / human review evaluated first, then candidate defect, then potential anomaly.
  2. Guarantee pairwise disjoint subsets across all three finding categories with `is_human_confirmed=False` across all algorithmic predictions.
  3. Author boundary unit tests for margins $-3.0$, $2.99$ (just below 3.0), $3.0$, $4.99$ (just below 5.0), and $5.0$.
  4. Ensure UI reflects this 3-level hierarchy, prominently highlights mandatory review actions, clearly distinguishes calibration stability from strict held-out metrics, and highlights Naive OR (79.05% BalAcc) vs Consensus (76.42% BalAcc) trade-offs.
  5. Run complete end-to-end product readiness verification across all 4 categories (`bottle`, `cable`, `metal_nut`, `screw`), LRU caching/eviction, and run the complete unit test suite.
- **Key Implementation & Verification Results:**
  1. **Strict 3-Level Precedence in `src/app/reporting/report.py`:**
     - Precedence 1 (`uncertain_human_review`): Evaluated first. Captures borderline score proximity ($|\Delta| < 3.0$), detector disagreements (specialized detector triggers while base PaDiM is nominal), or documented category ambiguities.
     - Precedence 2 (`candidate_defects`): Evaluated second. Captures strong indications not flagged as uncertain: high margin ($\Delta \ge 5.0$) OR multi-detector consensus ($\Delta \ge 3.0$).
     - Precedence 3 (`potential_anomalies`): Evaluated third. Captures moderate single-detector activations ($3.0 \le \Delta < 5.0$).
     - Hardcoded `is_human_confirmed=False` across all algorithmic outputs. Populated actionable review instructions (`🔍 Action Required: ...` / `⚡ Action Required: ...`).
  2. **Boundary Unit Testing in `tests/unit/test_report.py`:**
     - Added `test_finding_categorization_boundary_conditions` covering margins $-3.0$, $2.99$, $3.0$, $4.99$, $5.0$, detector disagreements, and corroborated consensus near boundaries.
     - Proved pairwise disjoint category sets across all tested outputs ($\text{Uncertain} \cap \text{Candidate} = \emptyset$, $\text{Uncertain} \cap \text{Potential} = \emptyset$, $\text{Candidate} \cap \text{Potential} = \emptyset$).
     - Unit test count expanded to 14 in `test_report.py`.
  3. **Frontend Dashboard Polish (`frontend/index.html`):**
     - Aligned inline inspection results and modal view to strictly reflect the 3-level order: Uncertain first, Candidate Defects second, Potential Anomalies third.
     - Added dedicated Multi-Detector Cable Benchmark Comparison table clearly identifying ⭐ Naive OR as the best binary balanced-accuracy result on this split (79.05% BalAcc, 89.13% recall) and explaining Consensus & Masked (76.42% BalAcc) operational trade-off (cutting false alarms on clean twisted cables and routing ambiguous spikes to human review).
     - Clearly distinguished Calibration Pool Stability (50 seeds) from Strict Held-Out Evaluation (untouched evaluation split, zero threshold-selection leakage).
  4. **Product Readiness Verification (`scratch/verify_product_readiness.py`):**
     - Step 1: Health, dynamic category listing (`bottle`, `cable`, `metal_nut`, `screw`), metadata, benchmark loading, and hybrid KB query verified.
     - Step 2: Multi-detector cable inspections on clean cable (nominal, 0 false alarms), missing wire (continuity alert, routed to human review), and puncture (high-margin defect detected, score 54.11) verified.
     - Step 3: Model caching & category switching across all 4 categories verified. Cache bounded at capacity 3, LRU model eviction verified without memory leaks.
     - Step 4: SQLite persistence and report retrieval verified.
     - Step 5: Full JSON download export (14,561 bytes) with disclaimer verified.
  5. **Unit Test Suite:**
     - **74 / 74 tests passed in 48.38s** across 11 unit test suites.
- **Preservation Check:** All existing production models and historical benchmark records strictly preserved. Zero tuning against held-out labels.

---

### Session 11: Multi-Detector Decision Integrity, Provenance & Reporting System Polish
- **Date & Local Time:** 2026-11-07T15:55:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Inspect finding categories and enforce strict mathematical mutual exclusivity.
  2. Resolve overlapping conditions between Potential Anomalies and Human Review.
  3. Ensure algorithmic model outputs are never labeled confirmed physical defects (`is_human_confirmed: bool = False`).
  4. Track multi-detector evidence provenance (`FindingProvenance`) and prevent duplicate findings.
  5. Resolve detector disagreements and investigate FPR increase on calibration data only.
  6. Measure end-to-end composite pipeline latency.
  7. Add unit tests, update documentation, and commit verified work.
- **Key Implementation Improvements & Findings:**
  1. **Strict Mutual Exclusivity:**
     - Borderline zone ($|\Delta| < 3.0$) maps exclusively to `uncertain_human_review`.
     - Multi-detector corroboration or high margin ($\Delta \ge 5.0$) maps exclusively to `candidate_defects`.
     - Moderate single-detector activation ($3.0 \le \Delta < 5.0$) maps exclusively to `potential_anomalies`.
     - Unit tests verify pairwise disjoint category sets across all findings.
  2. **Authority Boundary & De-duplication:**
     - Added `FindingProvenance` (`detector`, `score`, `threshold`, `margin`, `location`, `bbox`) and `CategorizedFinding` (`id`, `category`, `title`, `description`, `provenance`, `is_human_confirmed=False`, `action_required`).
     - Added unique deterministic IDs preventing duplicate finding display.
     - Preserved clear distinction: model indications are candidate hypotheses, not confirmed physical defects.
  3. **Detector Disagreement & False Positive Resolution:**
     - Proved on calibration data only: unconstrained OR logic at $T=38.0$ across the full frame triggered on boundary edge noise and cable bend glints on twisted clean cables.
     - Resolved by masking outer 2 patches to suppress frame noise and routing uncorroborated single-detector boundary spikes where PaDiM is nominal strictly into `uncertain_human_review` with actionable inspection instructions.
     - On calibration data, this cuts false alarms in half (FPR $27.59\% \to 13.79\%$), increases specificity to $86.21\%$, and boosts calibration balanced accuracy to $85.49\%$. Zero tuning against held-out labels.
  4. **Latency Measurement:**
     - Composite vision inspection: 374.57 ms (PaDiM: 154.32 ms, Continuity: 74.22 ms, Puncture: 97.76 ms).
     - Hybrid technical document retrieval: 14.47 ms.
     - SQLite history lookup: 0.62 ms.
     - Deterministic report generation: 0.09 ms.
     - Total end-to-end composite pipeline latency: **~390 ms**.
  5. **Dashboard Polish:**
     - Displayed mutually exclusive categories with provenance tags (`[Detector: PaDiM | Score: ...]`) and actionable inspection actions (`⚡ Action Required: ...`) in `frontend/index.html`.
- **Unit Tests Added:**
  - `test_category_mutual_exclusivity_and_non_overlap`: asserts pairwise disjoint category ID sets.
  - `test_detector_disagreement_handling`: verifies uncorroborated puncture trigger with nominal PaDiM routes to uncertain review.
  - `test_duplicate_findings_prevention`: verifies de-duplication of duplicate notes and evidence items.
  - `test_evidence_provenance_and_human_unconfirmed`: asserts provenance and `is_human_confirmed == False`.
  - Total test suite: **73 / 73 passed in 43.35s** across 11 test suites.
- **Preservation Check:** All production models (`bottle`, `cable`, `metal_nut`, `screw`), thresholds, and historical benchmarks strictly preserved. Zero tuning against held-out labels.

---

### Session 12: Independent Held-Out Evaluation, Latency Profiling & API Verification
- **Date & Local Time:** 2026-11-07T17:30:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Validate the consensus and boundary-masked cable inspection strategy on the untouched held-out evaluation split (75 images: 29 good, 46 defective, seed 42).
  2. Measure detailed wall-clock latencies (Mean, Median, P95) across vision, sub-detectors, retrieval, DB, and reporting stages.
  3. Seamlessly integrate `SpecializedCableInspector` into production `InspectionService` for category `"cable"`.
  4. Perform operational verification on real images (`missing_wire`, `good`, `poke_insulation`) across backend endpoints.
  5. Verify mutual exclusivity, provenance tracking, coordinates, actionable guidance, JSON download, and non-probabilistic score representation.
  6. Run full unit test suite (73 / 73 passing) and maintain persistent project memory.
- **Key Empirical Results & Verifications:**
  1. **Strict Held-Out 3-Way Comparison (`runs/cable_consensus_evaluation.json`):**
     - **Baseline PaDiM:** 28 TP / 5 FP / 24 TN / 18 FN | Sensitivity: **60.87%** | Specificity: **82.76%** | FPR: 17.24% | Balanced Accuracy: **71.81%**
     - **Naive OR Multi-Detector:** 41 TP / 9 FP / 20 TN / 5 FN | Sensitivity: **89.13%** | Specificity: **68.97%** | FPR: 31.03% | Balanced Accuracy: **79.05%**
     - **Consensus & Masked Inspector:** 37 TP / 8 FP / 21 TN / 9 FN | Sensitivity: **80.43%** | Specificity: **72.41%** | FPR: 27.59% | Balanced Accuracy: **76.42%**
     - Defect-specific recall: `bent_wire`: 100%, `cable_swap`: 100%, `combined`: 100%, `cut_inner`: 66.7%, `cut_outer`: 100%, `missing_cable`: 100%, `missing_wire`: 50.0%, `poke_insulation`: 50.0% (remaining 50% routed safely to `uncertain_human_review`).
  2. **Detailed Wall-Clock Latency Profiling (20 test images):**
     - `padim_only`: Mean: 148.23 ms | Median: 147.47 ms | P95: 155.88 ms
     - `continuity_only`: Mean: 60.39 ms | Median: 59.57 ms | P95: 64.37 ms
     - `puncture_only`: Mean: 93.58 ms | Median: 93.43 ms | P95: 95.73 ms
     - `composite_vision`: Mean: 355.84 ms | Median: 352.08 ms | P95: 374.28 ms
     - `hybrid_retrieval`: Mean: 14.20 ms | Median: 14.17 ms | P95: 14.90 ms
     - `sqlite_database`: Mean: 0.62 ms | Median: 0.60 ms | P95: 0.83 ms
     - `report_generation`: Mean: 0.11 ms | Median: 0.09 ms | P95: 0.17 ms
     - `total_end_to_end`: Mean: 370.77 ms | Median: 367.40 ms | P95: 389.10 ms
     - Overlap analysis: Disjoint sequential sum is 302.19 ms; composite vision is 355.84 ms (+53.65 ms) due to shared bounding box generation, contour extraction, finding reconciliation, and provenance assembly.
  3. **Backend Integration (`src/app/services/services.py`):**
     - Enhanced `inspect()` to route category `"cable"` dynamically through `SpecializedCableInspector(inspector, consensus_mode=True)` while preserving standard `AnomalyInspector` for other categories (`bottle`, `screw`, `metal_nut`).
  4. **Operational Inspection & API Verification (`scratch/verify_live_cable_inspection.py`):**
     - Defective missing wire: conductor continuity alert, routed to human review with specific action instructions.
     - Clean normal cable: clean verdict (`anomalous=False`, score 18.95 < threshold 30.42), 0 false alarms.
     - Microscopic puncture: high margin anomaly detected (`raw score 54.11 >= threshold 30.42`), localized puncture coordinates identified, actionable instructions: "Physically probe cable jacket at candidate puncture coordinates."
     - Data integrity: pairwise disjoint category sets, `is_human_confirmed == False`, complete detector provenance (`detector`, `score`, `threshold`, `location`, `bbox`), database persistence, and valid 12,275-byte JSON download verified.
- **Unit Test Results:**
  - Full suite verified: **73 / 73 passed in 53.45s**.
- **Preservation Check:** All existing production models (`data/index/vision/`) and historical benchmark results (`runs/eval_cv.json`, `runs/calibration.json`, `runs/held_out_evaluation.json`, `runs/cable_specialized_evaluation.json`) strictly preserved. Zero leakage or tuning against held-out labels.

---

### Session 13: Final Decision Logic & Product Readiness Pass
- **Date & Local Time:** 2026-11-07T17:45:00+05:30
- **Agent:** Antigravity Engineering Assistant (Gemini 3.8 Flash / Pair Programming)
- **Objective:**
  1. Inspect and enforce strict 3-level finding categorization precedence in `src/app/reporting/report.py`: uncertain / human review evaluated first, then candidate defect, then potential anomaly.
  2. Guarantee pairwise disjoint subsets across all three finding categories with `is_human_confirmed=False` across all algorithmic predictions.
  3. Author boundary unit tests for margins $-3.0$, $2.99$ (just below 3.0), $3.0$, $4.99$ (just below 5.0), and $5.0$.
  4. Ensure UI reflects this 3-level hierarchy, prominently highlights mandatory review actions, clearly distinguishes calibration stability from strict held-out metrics, and highlights Naive OR (79.05% BalAcc) vs Consensus (76.42% BalAcc) trade-offs.
  5. Run complete end-to-end product readiness verification across all 4 categories (`bottle`, `cable`, `metal_nut`, `screw`), LRU caching/eviction, and run the complete unit test suite.
- **Key Implementation & Verification Results:**
  1. **Strict 3-Level Precedence in `src/app/reporting/report.py`:**
     - Precedence 1 (`uncertain_human_review`): Evaluated first. Captures borderline score proximity ($|\Delta| < 3.0$), detector disagreements (specialized detector triggers while base PaDiM is nominal), or documented category ambiguities.
     - Precedence 2 (`candidate_defects`): Evaluated second. Captures strong indications not flagged as uncertain: high margin ($\Delta \ge 5.0$) OR multi-detector consensus ($\Delta \ge 3.0$).
     - Precedence 3 (`potential_anomalies`): Evaluated third. Captures moderate single-detector activations ($3.0 \le \Delta < 5.0$).
     - Hardcoded `is_human_confirmed=False` across all algorithmic outputs. Populated actionable review instructions (`🔍 Action Required: ...` / `⚡ Action Required: ...`).
  2. **Boundary Unit Testing in `tests/unit/test_report.py`:**
     - Added `test_finding_categorization_boundary_conditions` covering margins $-3.0$, $2.99$, $3.0$, $4.99$, $5.0$, detector disagreements, and corroborated consensus near boundaries.
     - Proved pairwise disjoint category sets across all tested outputs ($\text{Uncertain} \cap \text{Candidate} = \emptyset$, $\text{Uncertain} \cap \text{Potential} = \emptyset$, $\text{Candidate} \cap \text{Potential} = \emptyset$).
     - Unit test count expanded to 14 in `test_report.py`.
  3. **Frontend Dashboard Polish (`frontend/index.html`):**
     - Aligned inline inspection results and modal view to strictly reflect the 3-level order: Uncertain first, Candidate Defects second, Potential Anomalies third.
     - Added dedicated Multi-Detector Cable Benchmark Comparison table clearly identifying ⭐ Naive OR as the best binary balanced-accuracy result on this split (79.05% BalAcc, 89.13% recall) and explaining Consensus & Masked (76.42% BalAcc) operational trade-off (cutting false alarms on clean twisted cables and routing ambiguous spikes to human review).
     - Clearly distinguished Calibration Pool Stability (50 seeds) from Strict Held-Out Evaluation (untouched evaluation split, zero threshold-selection leakage).
  4. **Product Readiness Verification (`scratch/verify_product_readiness.py`):**
     - Step 1: Health, dynamic category listing (`bottle`, `cable`, `metal_nut`, `screw`), metadata, benchmark loading, and hybrid KB query verified.
     - Step 2: Multi-detector cable inspections on clean cable (nominal, 0 false alarms), missing wire (continuity alert, routed to human review), and puncture (high-margin defect detected, score 54.11) verified.
     - Step 3: Model caching & category switching across all 4 categories verified. Cache bounded at capacity 3, LRU model eviction verified without memory leaks.
     - Step 4: SQLite persistence and report retrieval verified.
     - Step 5: Full JSON download export (14,561 bytes) with disclaimer verified.
  5. **Unit Test Suite:**
     - **74 / 74 tests passed in 48.38s** across 11 unit test suites.
- **Preservation Check:** All existing production models and historical benchmark records strictly preserved. Zero tuning against held-out labels.
