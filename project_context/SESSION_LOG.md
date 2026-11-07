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
