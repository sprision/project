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

- [x] **Unit Test Suite Verification (45 / 45 Passed)**
  - `tests/unit/test_schemas.py`: 4 passed (0.31s)
  - `tests/unit/test_db.py`: 2 passed (0.72s)
  - `tests/unit/test_ingest.py`: 5 passed (1.23s)
  - `tests/unit/test_report.py`: 7 passed (0.70s) [Added 2 regression tests for dict-serialized retrieval]
  - `tests/unit/test_vision_validation.py`: 5 passed (14.45s)
  - `tests/unit/test_threshold_calibration.py`: 8 passed (15.55s) [Added 2 tests for calibration threshold loading and metric arithmetic]
  - `tests/unit/test_retriever.py`: 4 passed (38.71s)
  - `tests/unit/test_agent.py`: 10 passed (32.76s) [Added 1 regression test for tool report assembly]
  - Total: **45 passing tests** across 8 unit test suites.

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

- [ ] Interactive live browser test verification (launching uvicorn outside sandbox or on user command).

---

## Pending Tasks

- [ ] Optional: Add Git LFS tracking configuration for remote repositories.
