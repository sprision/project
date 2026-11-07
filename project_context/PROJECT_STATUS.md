# PROJECT_STATUS.md

## Current Implementation State (Verified)

### 1. Vision & Anomaly Detection (PaDiM, Registry & Experimental PatchCore)
- **Status:** Complete & Verified.
- **Production Fitted Models (4 Categories):**
  - `data/index/vision/bottle/padim.pkl` (662 MB) — Exists, verified.
  - `data/index/vision/cable/padim.pkl` (662 MB) — Exists, verified.
  - `data/index/vision/metal_nut/padim.pkl` (662 MB) — Exists, verified.
  - `data/index/vision/screw/padim.pkl` (662 MB) — Exists, verified.
- **Experimental Models & Benchmarking:**
  - `src/app/vision/patchcore.py`: Experimental PatchCore architecture featuring ResNet-18 layer2+3 patch representations and memory-bank nearest-neighbor scoring.
  - `src/app/vision/cable_inspector.py`: Specialized multi-detector cable inspection architecture integrating `ConductorContinuityDetector` and `SurfacePunctureDetector` with consensus strategy and boundary noise masking.
  - Head-to-head cable evaluation recorded in `runs/cable_specialized_evaluation.json` and `runs/cable_consensus_evaluation.json`:
    - Strict held-out evaluation across 75 untouched test samples (seed 42, zero leakage):
      - **Baseline PaDiM:** 28 TP / 5 FP / 24 TN / 18 FN | Sensitivity: **60.87%** | Specificity: **82.76%** | FPR: 17.24% | Balanced Accuracy: **71.81%** | Latency: 148.23 ms.
      - **Naive OR Multi-Detector:** 41 TP / 9 FP / 20 TN / 5 FN | Sensitivity: **89.13%** | Specificity: **68.97%** | FPR: 31.03% | Balanced Accuracy: **79.05%**.
      - **Consensus & Masked Inspector:** 37 TP / 8 FP / 21 TN / 9 FN | Sensitivity: **80.43%** | Specificity: **72.41%** | FPR: 27.59% | Balanced Accuracy: **76.42%** | Latency: 355.84 ms.
      - Key Defect Recall: `bent_wire`: 100.0%, `cable_swap`: 100.0%, `combined`: 100.0%, `cut_inner`: 66.7%, `cut_outer`: 100.0%, `missing_cable`: 100.0%, `missing_wire`: 50.0%, `poke_insulation`: 50.0% (remaining 50% routed strictly to `uncertain_human_review`).
    - Detailed latency percentiles across 20 test images:
      - Composite Vision: Mean **355.84 ms**, Median **352.08 ms**, P95 **374.28 ms**.
      - Hybrid Retrieval: Mean **14.20 ms**, Median **14.17 ms**, P95 **14.90 ms**.
      - SQLite History: Mean **0.62 ms**, Median **0.60 ms**, P95 **0.83 ms**.
      - Report Generation: Mean **0.11 ms**, Median **0.09 ms**, P95 **0.17 ms**.
      - Total End-to-End Pipeline: Mean **370.77 ms**, Median **367.40 ms**, P95 **389.10 ms**.
  - Empirical comparison documented in `runs/experimental_benchmarks.json`:
    - On held-out screw evaluation, PatchCore increases specificity from 28.57% (PaDiM) to **47.62%** and reduces FPR from 71.43% to **52.38%** (Pixel AUROC = 0.9709).
    - Rotation-normalized PaDiM evaluated: in-plane 2D alignment collapsed held-out AUROC to 0.4374 and sensitivity to 1.69% due to border padding, 180° flip ambiguity, and unresolved 3D helical roll angle.
- **Architecture & Registry:**
  - `src/app/vision/registry.py`: Scalable `ModelRegistry` with dynamic category discovery and thread-safe in-memory `LRUModelCache` (capacity = 3).
  - Lazy loading with automatic LRU eviction bounded at ~2.0 GB active RAM / ~132 MB CUDA VRAM.
  - Pretrained ResNet-18 backbone (`timm`), cached batched matrix multiplication (`torch.bmm`), connected-component defect region extraction, and JET heatmap overlay.
- **Live Tests:** Verified end-to-end inspections across all 4 categories.

### 2. Threshold Calibration & Reliability Validation
- **Status:** Complete, Audited & Certified.
- **Evaluation Protocols Supported:**
  1. **Strict Held-Out Evaluation (`runs/held_out_evaluation.json`):** Evaluates strictly untouched test samples (`ev` split, seed 42). Zero threshold-selection leakage.
  2. **Apparent Full-Test Evaluation (`runs/validation_results.json`):** Evaluates all test samples. Classified as apparent / threshold-selection-contaminated because 50% of images were in the calibration pool.
- **Per-Category Thresholds:**
  - `bottle`: `threshold_raw` = 23.4613 (Balanced Accuracy: 1.0 calib / 1.0 held-out eval, FPR: 0.0).
  - `cable`: `threshold_raw` = 30.4218 (Balanced Accuracy: 0.8306 calib / 0.7181 held-out eval, FPR: 0.1724).
  - `metal_nut`: `threshold_raw` = 23.1698 (Balanced Accuracy: 0.8785 calib / 0.9787 held-out eval, FPR: 0.0).
  - `screw`: `threshold_raw` = 24.1733 (Balanced Accuracy: 0.7333 calib / 0.5073 held-out eval, FPR: 0.7143).
- **Strict Held-Out Validation (`runs/held_out_evaluation.json` - 254 Untouched Images):**
  - **Bottle (41 images: 10 good, 31 defective):** TP=31, FP=0, TN=10, FN=0. Sensitivity: 1.0000 | Specificity: 1.0000 | FPR: 0.0000 | Balanced Accuracy: 1.0000
  - **Cable (75 images: 29 good, 46 defective):** TP=28, FP=5, TN=24, FN=18. Sensitivity: 0.6087 | Specificity: 0.8276 | FPR: 0.1724 | Balanced Accuracy: 0.7181
  - **Metal Nut (58 images: 11 good, 47 defective):** TP=45, FP=0, TN=11, FN=2. Sensitivity: 0.9574 | Specificity: 1.0000 | FPR: 0.0000 | Balanced Accuracy: 0.9787
  - **Screw (80 images: 21 good, 59 defective):** TP=43, FP=15, TN=6, FN=16. Sensitivity: 0.7288 | Specificity: 0.2857 | FPR: 0.7143 | Balanced Accuracy: 0.5073
- **Apparent Full Test Set Validation (`runs/validation_results.json` - 508 Images, Contaminated):**
  - **Bottle (83 images: 20 good, 63 defective):** TP=63, FP=0, TN=20, FN=0. Sensitivity: 1.0000 | Specificity: 1.0000 | Balanced Accuracy: 1.0000 | AUROC: 1.0000
  - **Cable (150 images: 58 good, 92 defective):** TP=60, FP=6, TN=52, FN=32. Sensitivity: 0.6522 | Specificity: 0.8966 | Balanced Accuracy: 0.7744 | AUROC: 0.8413
  - **Metal Nut (115 images: 22 good, 93 defective):** TP=84, FP=1, TN=21, FN=9. Sensitivity: 0.9032 | Specificity: 0.9545 | Balanced Accuracy: 0.9289 | AUROC: 0.9741
  - **Screw (160 images: 41 good, 119 defective):** TP=86, FP=20, TN=21, FN=33. Sensitivity: 0.7227 | Specificity: 0.5122 | Balanced Accuracy: 0.6174 | Image AUROC: 0.6280 | Pixel AUROC: 0.9726.
- **Disjoint Split Proof:** Verified via unit tests in `test_threshold_calibration.py` that calibration paths and held-out evaluation paths have zero intersection across all 4 categories.
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
- **Decision-Support Enhancements:**
  - Persistent global authority notice banner establishing decision-support boundaries (not an autonomous quality authority).
  - Explicit non-probabilistic score presentation: raw Mahalanobis distances and threshold margins clearly explained.
  - Multi-detector finding categorization with strict mutual exclusivity, duplicate prevention, and detector provenance tracking (`FindingProvenance`, `CategorizedFinding`).
  - Tripartite finding classification across report schemas and dashboard:
    1. 🔴 *Candidate Visual Defects* (Strong Model Indication • Unconfirmed)
    2. 🟡 *Potential Anomalies* (Single Detector Moderate Indication)
    3. 🔵 *Uncertain Findings* (Borderline / Disagreement • Human Review Required)
  - Explicit actionable recommendations attached to human review items.
  - Category benchmark reliability pills (`bottle`: high, `metal_nut`: high, `cable`: moderate with documented overlap).
  - Automated borderline score warning cards ($|raw - threshold| < 3.0$) mandating secondary human visual review.
  - Category-specific limitation alerts for cable geometry overlap and metal nut boundary edge reflections.

### 9. Test Suite Verification
- **Status:** 73 / 73 passing unit tests across all 11 test suites (43.35s runtime):
  - `test_schemas.py`: 4 passed
  - `test_db.py`: 2 passed
  - `test_ingest.py`: 5 passed
  - `test_model_registry.py`: 8 passed (LRU cache accounting, eviction order, production discovery, memory reporting)
  - `test_patchcore.py`: 4 passed (initialization, embedding shapes, fit memory bank, anomaly map, save/load cycle)
  - `test_cable_inspector.py`: 6 passed (conductor continuity detector, surface puncture detector, composite inspection, finding categorization)
  - `test_report.py`: 13 passed (regression tests for dict retrieval, borderline score warnings, category limitations, category mutual exclusivity, detector disagreement handling, duplicate prevention, evidence provenance, and unconfirmed physical defect distinction)
  - `test_vision_validation.py`: 5 passed
  - `test_threshold_calibration.py`: 12 passed (disjoint split verification, certified sample counts, calibrated threshold presence, metric arithmetic, seeded reproducibility)
  - `test_retriever.py`: 4 passed
  - `test_agent.py`: 10 passed (agent loop with dict retrieval and tool dispatch)

### 10. Deployment Strategy
- **Status:** Complete.
- `Dockerfile`: Multi-stage, CPU PyTorch wheel, libgl1/libglib2.0-0, healthcheck probe.
- `.dockerignore`: Configured.
- `docker-compose.yml`: Configured with persistent volumes.
