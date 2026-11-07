# PROJECT_STATUS.md

## Current Implementation State (Verified)

### 1. Vision & Anomaly Detection (PaDiM)
- **Status:** Complete & Verified.
- **Fitted Models:**
  - `data/index/vision/bottle/padim.pkl` (662 MB) — Exists, verified.
  - `data/index/vision/cable/padim.pkl` (662 MB) — Exists, verified.
  - `data/index/vision/metal_nut/padim.pkl` (662 MB) — Exists, verified.
- **Backbone:** ResNet-18 (`timm`), PyTorch 2.14.1 with CUDA enabled.
- **Optimizations:** Cached batched matrix multiplication (`torch.bmm`) in `src/app/vision/model.py` verified; enables sub-second anomaly map generation.
- **Region Extraction:** Connected components with contour/bounding-box statistics via OpenCV.
- **Heatmap Generation:** JET colormap blending over resized original image, saving to `data/heatmaps/`.
- **Live Test:** Verified on real sample `data/raw/mvtec_ad/bottle/test/broken_large/000.png` -> Raw score `64.897` vs threshold `23.4613` -> correctly flagged anomalous with 1 defect region and overlay in 4.26s.

### 2. Threshold Calibration & Reliability Validation
- **Status:** Complete & Fully Validated across all 348 MVTec AD test samples.
- **Per-Category Thresholds:**
  - `bottle`: `threshold_raw` = 23.4613 (Balanced Accuracy: 1.0, FPR: 0.0). Verified in `data/index/vision/bottle/threshold.json`.
  - `cable`: `threshold_raw` = 30.4218 (Balanced Accuracy: 0.8306 calib / 0.7181 eval, FPR: 0.1724). Verified in `data/index/vision/cable/threshold.json`.
  - `metal_nut`: `threshold_raw` = 23.1698 (Balanced Accuracy: 0.8785 calib / 0.9787 eval, FPR: 0.0). Verified in `data/index/vision/metal_nut/threshold.json`.
- **Full Test Set Validation (All 348 Images in `runs/validation_results.json`):**
  - **Bottle (83 images: 20 good, 63 defective):**
    - TP=63, FP=0, TN=20, FN=0
    - Sensitivity: 1.0000 | Specificity: 1.0000 | Balanced Accuracy: 1.0000 | FPR: 0.0000
    - Image AUROC: 1.0000 (Matches `runs/eval_cv.json`)
  - **Cable (150 images: 58 good, 92 defective):**
    - TP=60, FP=6, TN=52, FN=32
    - Sensitivity: 0.6522 | Specificity: 0.8966 | Balanced Accuracy: 0.7744 | FPR: 0.1034
    - Image AUROC: 0.8413 (Matches `runs/eval_cv.json`)
    - False positives (6): clean samples scored between 31.42 and 35.80 (slight sheath irregularities).
    - False negatives (32): primarily `missing_wire` (9) and `poke_insulation` (10) due to low patch activation on tiny pinpricks/absent conductors.
  - **Metal Nut (115 images: 22 good, 93 defective):**
    - TP=84, FP=1, TN=21, FN=9
    - Sensitivity: 0.9032 | Specificity: 0.9545 | Balanced Accuracy: 0.9289 | FPR: 0.0455
    - Image AUROC: 0.9741 (Matches `runs/eval_cv.json`)
    - False positive (1): `metal_nut/test/good/006.png` scored 30.0865 due to an edge feature at image crop boundary (`y=0`).
    - False negatives (9): subtle surface scratches (`scratch`: 4) and minor bends (`bent`: 5) scoring close to threshold 23.1698.
- **Calibration Log:** Comprehensive 50-seed stability metrics recorded in `runs/calibration.json`.
- **Verdict Modes:** Both `calibrated_raw` and `normalised` supported across `model.py`, `services.py`, and `report.py`.

### 3. Retrieval-Augmented Generation (RAG)
- **Status:** Complete & Verified.
- **Indexed Chunks:** Verified in `data/index/chunks.json` (17 chunks across 4 documents).
- **Embedding Vectors:** Verified in `data/index/embeddings.npz` (17 normalized 384-d vectors).
- **BM25 Search:** `BM25Plus` tokenizer and scoring active.
- **Hybrid Fusion:** Reciprocal Rank Fusion (k=60) blending dense and lexical scores.
- **Offline Optimization:** `local_files_only=True` integrated into `retriever.py` to prevent DNS timeout delays.
- **Knowledge Base Source Docs:** 4 files present in `data/kb/`:
  - `maintenance_decision_support.txt`
  - `mvtec_ad_dataset.txt`
  - `padim_method.txt`
  - `surface_inspection_guidance.txt`

### 4. Inspection Agent & Tool Execution
- **Status:** Complete & Verified.
- **Tools Implemented (6/6):**
  - `inspect_image` (calls PaDiM AnomalyInspector)
  - `retrieve_technical_documents` (calls hybrid Retriever)
  - `search_inspection_history` (queries SQLite `inspections` table)
  - `compare_inspection_results` (computes score delta against previous inspection)
  - `summarize_evidence` (aggregates observations and facts)
  - `generate_inspection_report` (assembles cited InspectionReport)
- **Agent Policy:** Bounded rule-based/keyword tool selection in `src/app/agent/agent.py`.
- **Trace Logging:** Full tool execution trace with latency (`ms`), status (`ok`), and error strings.

### 5. Persistence & Database
- **Status:** Complete & Verified.
- **SQLite Engine:** `src/app/database/db.py` managing `data/index/inspections.sqlite`.
- **Schema:** Table `inspections` (columns: `id`, `image_id`, `category`, `question`, `image_anomaly_score`, `is_anomalous`, `n_regions`, `generator`, `report_json`, `created_at`) with index on `(category, created_at DESC)`.
- **Live Test:** Verified inspection write and retrieval through `reports()` endpoint.

### 6. Reporting Layer
- **Status:** Complete & Verified.
- **Deterministic Generator:** Verified in `src/app/reporting/report.py` (cites evidence, classifies items into visual observations, retrieved facts, hypotheses, recommendations, limitations, and mandatory disclaimer). Handles both object and dict serialized forms of `retrieval` and `evidence`.
- **Optional LLM Generator:** Present for OpenAI / Gemini with deterministic fallback if keys are missing or API fails.

### 7. FastAPI Backend & API Surface
- **Status:** Complete & Verified.
- **Entrypoint:** `src/app/main.py` mounting router from `src/app/api/routes.py` and serving static files from `frontend/`.
- **Routes:**
  - `GET /api/health`
  - `GET /api/categories`
  - `POST /api/inspect`
  - `GET /api/kb/query`
  - `POST /api/kb/ingest`
  - `GET /api/heatmap/{name}`
  - `GET /api/reports`
  - `GET /api/reports/{id}`
  - `GET /api/eval/cv` (New)
  - `GET /api/eval/calibration` (New)

### 8. Frontend Dashboard
- **Status:** Complete & Verified (`frontend/index.html`).
- **Features:** Dark industrial UI with overview dashboard, image upload, PaDiM inference trigger, heatmap viewing, defect region details, KB search, document ingestion, inspection history table, report viewer, and JSON download. Dynamically loads CV evaluation and calibration metrics from `/api/eval/`.

### 9. Test Suite Verification
- **Status:** 45 / 45 passing unit tests across all 8 suites:
  - `test_schemas.py`: 4 passed
  - `test_db.py`: 2 passed
  - `test_ingest.py`: 5 passed
  - `test_report.py`: 7 passed (includes 2 regression tests for dict retrieval)
  - `test_vision_validation.py`: 5 passed
  - `test_threshold_calibration.py`: 8 passed (includes 2 tests for calibration threshold presence and metric arithmetic)
  - `test_retriever.py`: 4 passed
  - `test_agent.py`: 10 passed (includes 1 regression test for agent report generation tool with dict retrieval)

### 10. Deployment Strategy
- **Status:** Complete.
- `Dockerfile`: Multi-stage, CPU PyTorch wheel, libgl1/libglib2.0-0, healthcheck probe.
- `.dockerignore`: Configured.
- `docker-compose.yml`: Configured with persistent volumes.
