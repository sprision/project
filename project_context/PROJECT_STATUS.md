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

### 2. Threshold Calibration
- **Status:** Complete & Verified.
- **Per-Category Thresholds:**
  - `bottle`: `threshold_raw` = 23.4613 (Balanced Accuracy: 1.0, FPR: 0.0). Verified in `data/index/vision/bottle/threshold.json`.
  - `cable`: `threshold_raw` = 30.4218 (Balanced Accuracy: 0.8306 calib / 0.7181 eval, FPR: 0.1724). Verified in `data/index/vision/cable/threshold.json`.
  - `metal_nut`: `threshold_raw` = 23.1698 (Balanced Accuracy: 0.8785 calib / 0.9787 eval, FPR: 0.0). Verified in `data/index/vision/metal_nut/threshold.json`.
- **Calibration Log:** Comprehensive 50-seed stability metrics recorded in `runs/calibration.json`.
- **Verdict Modes:** Both `calibrated_raw` and `normalised` supported across `model.py`, `services.py`, and `report.py`.

### 3. Retrieval-Augmented Generation (RAG)
- **Status:** Complete & Verified.
- **Indexed Chunks:** Verified in `data/index/chunks.json` (17 chunks across 4 documents).
- **Embedding Vectors:** Verified in `data/index/embeddings.npz` (17 normalized 384-d vectors).
- **BM25 Search:** `BM25Plus` tokenizer and scoring active.
- **Hybrid Fusion:** Reciprocal Rank Fusion ($k=60$) blending dense and lexical scores.
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

### 6. Reporting Layer
- **Status:** Complete & Verified.
- **Deterministic Generator:** Verified in `src/app/reporting/report.py` (cites evidence, classifies items into visual observations, retrieved facts, hypotheses, recommendations, limitations, and mandatory disclaimer).
- **Optional LLM Generator:** Present for OpenAI / Gemini with deterministic fallback if keys are missing or API fails.

### 7. FastAPI Backend
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

### 8. Frontend
- **Status:** Implemented (`frontend/index.html`).
- **Features:** Dark industrial UI with overview dashboard, image upload, PaDiM inference trigger, heatmap viewing, defect region details, KB search, document ingestion, inspection history table, report viewer, and JSON download.

---

## Known Incomplete or Ongoing Work

1. **End-to-End Local Server Verification:**
   - FastAPI server startup on live port and interactive testing with sample images from `data/raw/mvtec_ad/` to ensure end-to-end user experience in browser.
2. **Production Deployment Strategy:**
   - Formal deployment packaging (Docker containerization, environment configuration, reverse proxy/production runner setup such as Uvicorn with Gunicorn or systemd/Nginx).
3. **LLM Provider API Integration:**
   - LLM generation currently defaults to deterministic because `APP_LLM_PROVIDER=none` in `.env`. External API keys can be provided if non-deterministic summarization is desired.

